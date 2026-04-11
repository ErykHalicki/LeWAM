import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention as _flex_attention_eager

_USE_FLEX = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
if _USE_FLEX:
    flex_attention = torch.compile(_flex_attention_eager, dynamic=True)
else:
    flex_attention = _flex_attention_eager


# ── Position ID creation ─────────────────────────────────────────────────────
#
# These functions create (t_ids, h_ids, w_ids) tuples for 3D RoPE.
# Temporal IDs are in seconds (frame_index / fps) so they stay meaningful
# across frame rate changes. Spatial IDs are integer grid coordinates.
#
# To experiment with different position schemes (e.g. log-spaced temporal,
# learned offsets, camera-aware spatial), replace these functions.

def make_video_pos_ids(num_frames, patch_h, patch_w, fps, t_offset=0):
    """
    3D position IDs for video patch tokens (raw, unscaled).
    Returns (t_ids, h_ids, w_ids), each shape (num_frames * patch_h * patch_w,).
    """
    t_ids = ((torch.arange(num_frames).float() + t_offset) / fps).repeat_interleave(patch_h * patch_w)
    h_ids = torch.arange(patch_h).float().repeat_interleave(patch_w).repeat(num_frames)
    w_ids = torch.arange(patch_w).float().repeat(patch_h).repeat(num_frames)
    return t_ids, h_ids, w_ids


def make_action_pos_ids(action_horizon, fps, t_offset=0):
    """
    1D temporal position IDs for action tokens (h=0, w=0).
    Returns (t_ids, h_ids, w_ids), each shape (action_horizon,).
    """
    t_ids = (torch.arange(action_horizon).float() + t_offset) / fps
    h_ids = torch.zeros(action_horizon)
    w_ids = torch.zeros(action_horizon)
    return t_ids, h_ids, w_ids


def concat_pos_ids(*pos_tuples):
    """Concatenate multiple (t_ids, h_ids, w_ids) tuples along the token dimension."""
    return tuple(torch.cat([p[i] for p in pos_tuples]) for i in range(3))


class PatchPositionIds(nn.Module):
    """
    Stores 3D RoPE position IDs as buffers (auto device/dtype tracking).
    fps and patch grid can be changed at runtime via set_fps / set_patch_grid.
    """
    def __init__(self, num_frames, patch_h, patch_w, fps, t_offset=0):
        super().__init__()
        self._num_frames = num_frames
        self._patch_h = patch_h
        self._patch_w = patch_w
        self._fps = fps
        self._t_offset = t_offset
        t, h, w = make_video_pos_ids(num_frames, patch_h, patch_w, fps, t_offset)
        self.register_buffer('t_ids', t, persistent=False)
        self.register_buffer('h_ids', h, persistent=False)
        self.register_buffer('w_ids', w, persistent=False)

    def _recompute(self):
        device = self.t_ids.device
        t, h, w = make_video_pos_ids(
            self._num_frames, self._patch_h, self._patch_w,
            self._fps, self._t_offset,
        )
        self.t_ids = t.to(device)
        self.h_ids = h.to(device)
        self.w_ids = w.to(device)

    def set_fps(self, fps):
        self._fps = fps
        self._recompute()

    def set_patch_grid(self, H, W):
        self._patch_h = H
        self._patch_w = W
        self._recompute()

    @property
    def pos(self):
        return (self.t_ids, self.h_ids, self.w_ids)


# ── Utilities ─────────────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLULinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
        self.w_gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.w(x) * F.silu(self.w_gate(x))


def make_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(SwiGLULinear(in_dim, hidden_dim), nn.Linear(hidden_dim, out_dim))


def split_heads(x, num_heads, head_dim):
    B, N, _ = x.shape
    return x.reshape(B, N, num_heads, head_dim).transpose(1, 2)


# ── RoPE ──────────────────────────────────────────────────────────────────────

PRETRAINED_GRID_SIZE = 14  # TODO: make configurable / loadable from checkpoint

class RoPE3D(nn.Module):
    """
    3D Rotary Position Embeddings for spatiotemporal tokens.
    Head dim split: d//4 temporal, d//4 height, d//2 width.

    When interpolate=True, spatial position IDs are scaled so the current
    grid maps to the pretrained 14x14-per-camera range. This allows the
    model to handle different crop sizes and camera counts at fine-tuning
    or inference time.

    Action tokens use this with h=0, w=0 for effective 1D temporal RoPE
    (spatial bands become identity rotations).
    """
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4"
        self.dim_t = head_dim // 4
        self.dim_h = head_dim // 4
        self.dim_w = head_dim // 2
        self.interpolate = False
        self.pretrained_h = PRETRAINED_GRID_SIZE
        self.pretrained_w = PRETRAINED_GRID_SIZE
        self.register_buffer('freqs_t', self._freqs(self.dim_t, base), persistent=False)
        self.register_buffer('freqs_h', self._freqs(self.dim_h, base), persistent=False)
        self.register_buffer('freqs_w', self._freqs(self.dim_w, base), persistent=False)

    def set_interpolation(self, patch_h, patch_w, num_cameras):
        self.interpolate = True
        self.pretrained_h = PRETRAINED_GRID_SIZE
        self.pretrained_w = PRETRAINED_GRID_SIZE * num_cameras
        self.current_h = patch_h
        self.current_w = patch_w

    @staticmethod
    def _freqs(dim, base):
        return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rot(self, x, positions, freqs):
        angles = positions.float()[:, None] * freqs.float()[None, :]
        angles = torch.cat([angles, angles], dim=-1).to(x.dtype)
        cos = angles.cos()[None, None]
        sin = angles.sin()[None, None]
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x, t_ids, h_ids, w_ids):
        """x: (B, heads, N, head_dim) -> rotated x."""
        if self.interpolate:
            if self.current_h > 1:
                h_ids = h_ids * (self.pretrained_h - 1) / (self.current_h - 1)
            if self.current_w > 1:
                w_ids = w_ids * (self.pretrained_w - 1) / (self.current_w - 1)
        xt, xh, xw = x.split([self.dim_t, self.dim_h, self.dim_w], dim=-1)
        return torch.cat([
            self._rot(xt, t_ids, self.freqs_t),
            self._rot(xh, h_ids, self.freqs_h),
            self._rot(xw, w_ids, self.freqs_w),
        ], dim=-1)


# ── Attention ─────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, rope=None, pos=None, attn_mask=None):
        B, N, _ = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = split_heads(Q, self.num_heads, self.head_dim)
        K = split_heads(K, self.num_heads, self.head_dim)
        V = split_heads(V, self.num_heads, self.head_dim)
        if rope is not None and pos is not None:
            Q = rope(Q, *pos)
            K = rope(K, *pos)
        Q = self.q_norm(Q).to(V.dtype)
        K = self.k_norm(K).to(V.dtype)
        if _USE_FLEX:
            out = flex_attention(Q, K, V, block_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, -1))


class CrossAttention(nn.Module):
    """
    Cross-attention with per-source K/V projections.
    RoPE on Q only: queries know their spatiotemporal position,
    sources (language, VLM image tokens, state) don't share that coordinate system.
    """
    def __init__(self, hidden_size, num_heads, num_sources):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_sources)])
        self.k_norms = nn.ModuleList([nn.LayerNorm(self.head_dim) for _ in range(num_sources)])
        self.v_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_sources)])
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, sources, source_masks=None, rope=None, q_pos=None, use_kv_cache=False):
        B, N, _ = q.shape
        Q = split_heads(self.q_proj(q), self.num_heads, self.head_dim)
        if rope is not None and q_pos is not None:
            Q = rope(Q, *q_pos)
        Q = self.q_norm(Q)

        if use_kv_cache and self._cached_k is not None:
            K = self._cached_k
            V = self._cached_v
            attn_mask = self._cached_mask
        else:
            Ks, Vs = [], []
            combined_mask = None
            for i, src in enumerate(sources):
                if src is None:
                    continue
                Ks.append(self.k_norms[i](split_heads(self.k_projs[i](src), self.num_heads, self.head_dim)))
                Vs.append(split_heads(self.v_projs[i](src), self.num_heads, self.head_dim))
                if source_masks is not None:
                    m = source_masks[i]
                    chunk = m if m is not None else torch.zeros(B, src.shape[1], dtype=torch.bool, device=q.device)
                    combined_mask = chunk if combined_mask is None else torch.cat([combined_mask, chunk], dim=1)

            K = torch.cat(Ks, dim=2)
            V = torch.cat(Vs, dim=2)
            attn_mask = ~combined_mask.unsqueeze(1).unsqueeze(1) if combined_mask is not None else None

            if use_kv_cache:
                self._cached_k = K
                self._cached_v = V
                self._cached_mask = attn_mask

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, -1))


# ── Action Preprocessor ─────────────────────────────────────────────────────

class ActionPreprocessor(nn.Module):
    """
    Normalizes and unnormalizes relative actions and state using precomputed stats.

    Stats file stores q1..q99, mean, std for both rel_action and state.
    The norm_strategy selects which percentiles to clip to before z-scoring:
        "q1_q99"  -> clip to 1st/99th percentile (default)
        "q5_q95"  -> clip to 5th/95th percentile
        "q10_q90" -> clip to 10th/90th percentile
        "none"    -> no clipping, just z-score
    """

    def __init__(self, stats: dict, norm_strategy: str = "q1_q99"):
        super().__init__()
        self.norm_strategy = norm_strategy

        if norm_strategy != "none":
            lo_key, hi_key = norm_strategy.split("_")
        else:
            lo_key, hi_key = "q1", "q99"

        for key in ("rel_action", "state"):
            self.register_buffer(f"{key}_lo", stats[key][lo_key].float())
            self.register_buffer(f"{key}_hi", stats[key][hi_key].float())
            self.register_buffer(f"{key}_mean", stats[key]["mean"].float())
            self.register_buffer(f"{key}_std", stats[key]["std"].float())

    def normalize_rel_action(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_strategy != "none":
            x = x.clamp(self.rel_action_lo, self.rel_action_hi)
        return (x - self.rel_action_mean) / self.rel_action_std

    def unnormalize_rel_action(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.rel_action_std + self.rel_action_mean

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_strategy != "none":
            x = x.clamp(self.state_lo, self.state_hi)
        return (x - self.state_mean) / self.state_std

    def unnormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.state_std + self.state_mean


# ── Transformer Block ────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    SA (3D RoPE + block-causal mask, adaLN-Zero) -> CA (RoPE on Q) -> MLP (adaLN-Zero)
    """
    def __init__(self, hidden_size, num_heads, num_sources=0, mlp_ratio=4.0,
                 use_adaln=True, dropout=0.0, sources_dim=None):
        super().__init__()
        self.use_adaln = use_adaln
        self.rope = RoPE3D(hidden_size // num_heads)

        affine = not use_adaln
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=affine, eps=1e-6)
        self.sa = SelfAttention(hidden_size, num_heads)

        self.has_ca = num_sources > 0
        self.norm_ca = nn.LayerNorm(hidden_size, eps=1e-6) if self.has_ca else None
        self.ca = CrossAttention(hidden_size, num_heads, num_sources) if self.has_ca else None
        if sources_dim is not None:
            assert len(sources_dim) == num_sources
            self.source_projs = nn.ModuleList([
                nn.Linear(d, hidden_size) for d in sources_dim
            ])
        else:
            self.source_projs = None

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=affine, eps=1e-6)
        self.mlp = make_mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size)
        self.drop = nn.Dropout(dropout)

        if use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(self, x, sources=None, source_masks=None, cond=None, attn_mask=None, pos=None,
                use_kv_cache=False):
        """
        x:            (B, N, hidden_size)
        sources:      list of (B, N_i, source_dim_i) tensors (projected internally if source_projs set)
        source_masks: list of (B, N_i) bool (True=ignore) or None per source
        cond:         (B, hidden_size) adaLN-Zero conditioning
        attn_mask:    (1, 1, N, N) bool, True=attend
        pos:          (t_ids, h_ids, w_ids) for 3D RoPE on all tokens
        use_kv_cache: if True, cache/reuse cross-attention K/V for static sources
        """
        if not self.use_adaln and cond is not None:
            warnings.warn("cond passed to Block with use_adaln=False, ignored", stacklevel=2)

        shift_msa = scale_msa = gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        if self.use_adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(cond).chunk(6, dim=1)

        h = modulate(self.norm1(x), shift_msa, scale_msa) if self.use_adaln else self.norm1(x)
        sa_out = self.drop(self.sa(h, rope=self.rope, pos=pos, attn_mask=attn_mask))
        x = x + (gate_msa.unsqueeze(1) * sa_out if self.use_adaln else sa_out)

        cache_hit = use_kv_cache and self.ca is not None and self.ca._cached_k is not None
        if self.has_ca and (sources is not None or cache_hit):
            if not cache_hit and self.source_projs is not None:
                sources = [
                    proj(s) if s is not None else None
                    for proj, s in zip(self.source_projs, sources)
                ]
            ca_out = self.drop(self.ca(
                self.norm_ca(x), sources, source_masks=source_masks,
                rope=self.rope, q_pos=pos, use_kv_cache=use_kv_cache,
            ))
            x = x + ca_out

        h = modulate(self.norm2(x), shift_mlp, scale_mlp) if self.use_adaln else self.norm2(x)
        mlp_out = self.drop(self.mlp(h))
        x = x + (gate_mlp.unsqueeze(1) * mlp_out if self.use_adaln else mlp_out)

        return x
