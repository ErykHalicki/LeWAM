import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_patch_ids(t_start, num_frames, H, W, fps):
    t_ids = (torch.arange(t_start, t_start + num_frames).float() / fps).repeat_interleave(H * W)
    h_ids = torch.arange(H).repeat_interleave(W).repeat(num_frames)
    w_ids = torch.arange(W).repeat(H * num_frames)
    return t_ids, h_ids, w_ids


class PatchPositionIds(nn.Module):
    """
    Holds 3D RoPE position ids (t, h, w) for a fixed spatiotemporal grid.
    fps can be updated at runtime via set_fps() without re-creating the module.

    pos property returns (t_ids, h_ids, w_ids) tuple for use in RoPE.
    """
    def __init__(self, t_start, num_frames, H, W, fps):
        super().__init__()
        self._t_start = t_start
        self._num_frames = num_frames
        self._H = H
        self._W = W
        t_ids, h_ids, w_ids = make_patch_ids(t_start, num_frames, H, W, fps)
        self.register_buffer('t_ids', t_ids, persistent=False)
        self.register_buffer('h_ids', h_ids, persistent=False)
        self.register_buffer('w_ids', w_ids, persistent=False)

    def set_fps(self, fps: float):
        t_ids, h_ids, w_ids = make_patch_ids(self._t_start, self._num_frames, self._H, self._W, fps)
        self.t_ids = t_ids.to(self.t_ids.device)
        self.h_ids = h_ids.to(self.h_ids.device)
        self.w_ids = w_ids.to(self.w_ids.device)

    @property
    def pos(self):
        return (self.t_ids, self.h_ids, self.w_ids)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLULinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w      = nn.Linear(in_dim, out_dim)
        self.w_gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.w(x) * F.silu(self.w_gate(x))


def make_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        SwiGLULinear(in_dim, hidden_dim),
        nn.Linear(hidden_dim, out_dim),
    )

def split_heads(x, num_heads, head_dim):
    B, N, _ = x.shape
    return x.reshape(B, N, num_heads, head_dim).transpose(1, 2)


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embeddings: (temporal, height, width).
    Head dim split: d//4 temporal, d//4 height, d//2 width.
    Requires head_dim % 4 == 0.
    """
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 3D RoPE"
        self.dim_t = head_dim // 4
        self.dim_h = head_dim // 4
        self.dim_w = head_dim // 2
        self.register_buffer('freqs_t', self._freqs(self.dim_t, base), persistent=False)
        self.register_buffer('freqs_h', self._freqs(self.dim_h, base), persistent=False)
        self.register_buffer('freqs_w', self._freqs(self.dim_w, base), persistent=False)

    @staticmethod
    def _freqs(dim, base):
        return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rot(self, x, positions, freqs):
        angles = positions.float()[:, None] * freqs[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        cos = angles.cos()[None, None]
        sin = angles.sin()[None, None]
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x, t_ids, h_ids, w_ids):
        """x: (B, heads, N, head_dim) — returns rotated x."""
        xt, xh, xw = x.split([self.dim_t, self.dim_h, self.dim_w], dim=-1)
        return torch.cat([
            self._rot(xt, t_ids, self.freqs_t),
            self._rot(xh, h_ids, self.freqs_h),
            self._rot(xw, w_ids, self.freqs_w),
        ], dim=-1)


class SelfAttention(nn.Module):
    """Self-attention with optional 3D RoPE. Uses fused QKV projection."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, rope=None, pos=None):
        B, N, _ = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = split_heads(Q, self.num_heads, self.head_dim)
        K = split_heads(K, self.num_heads, self.head_dim)
        V = split_heads(V, self.num_heads, self.head_dim)
        if rope is not None and pos is not None:
            Q = rope(Q, *pos)
            K = rope(K, *pos)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, -1))


class CrossAttention(nn.Module):
    """
    Cross-attention with independent K/V projections per source.
    Sources are concatenated before attention, with optional per-source RoPE and masking.
    """
    def __init__(self, hidden_size, num_heads, num_sources):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_sources)])
        self.v_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_sources)])
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, sources, rope=None, q_pos=None, source_positions=None, source_masks=None):
        """
        q:                (B, N, hidden_size)
        sources:          list of (B, N_i, hidden_size), one per source
        rope:             RoPE3D or None
        q_pos:            (t_ids, h_ids, w_ids) for Q, or None
        source_positions: list of (t_ids, h_ids, w_ids) or None per source
        source_masks:     list of (B, N_i) bool masks (True=ignore) or None per source
        """
        B, N, _ = q.shape

        Q = split_heads(self.q_proj(q), self.num_heads, self.head_dim)
        if rope is not None and q_pos is not None:
            Q = rope(Q, *q_pos)

        Ks, Vs = [], []
        combined_mask = None
        for i, src in enumerate(sources):
            if src is None:
                continue
            K_i = split_heads(self.k_projs[i](src), self.num_heads, self.head_dim)
            V_i = split_heads(self.v_projs[i](src), self.num_heads, self.head_dim)
            pos_i = source_positions[i] if source_positions is not None else None
            if rope is not None and pos_i is not None:
                K_i = rope(K_i, *pos_i)
            Ks.append(K_i)
            Vs.append(V_i)
            if source_masks is not None:
                m = source_masks[i]
                chunk = m if m is not None else torch.zeros(B, src.shape[1], dtype=torch.bool, device=q.device)
                combined_mask = chunk if combined_mask is None else torch.cat([combined_mask, chunk], dim=1)

        K = torch.cat(Ks, dim=2)
        V = torch.cat(Vs, dim=2)
        attn_mask = ~combined_mask.unsqueeze(1).unsqueeze(1) if combined_mask is not None else None

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, -1))


class Block(nn.Module):
    """
    Unified transformer block used by both DiT (flow matching) and IDM (regression).

    SA (optional 3D RoPE, optional adaLN-Zero) → CA (variable sources) → MLP (optional adaLN-Zero)

    use_adaln=True:  adaLN-Zero conditioning via `cond` (B, hidden_size) — for DiT.
    use_adaln=False: standard LayerNorm, no gating — for IDM.

    sa_first=True:  SA → CA → MLP (DiT: queries have content from the start).
    sa_first=False: CA → SA → MLP (IDM: queries are learned tokens with no initial content,
                    so CA must run first to give them something meaningful to self-attend over).

    q_pos passed to forward controls whether queries get 3D RoPE in SA and CA.
    source_positions controls 3D RoPE on each source's K/V in CA.
    """
    def __init__(self, hidden_size, num_heads, num_sources, mlp_ratio=4.0, use_adaln=True, sa_first=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.rope = RoPE3D(hidden_size // num_heads)

        affine = not use_adaln
        self.norm1  = nn.LayerNorm(hidden_size, elementwise_affine=affine, eps=1e-6)
        self.sa     = SelfAttention(hidden_size, num_heads)

        self.norm_ca = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ca      = CrossAttention(hidden_size, num_heads, num_sources)

        self.norm2    = nn.LayerNorm(hidden_size, elementwise_affine=affine, eps=1e-6)
        self.mlp      = make_mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size)
        self.sa_first = sa_first

        if use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def _run_sa(self, x, shift=None, scale=None, gate=None, q_pos=None):
        h = modulate(self.norm1(x), shift, scale) if self.use_adaln else self.norm1(x)
        out = self.sa(h, rope=self.rope if q_pos is not None else None, pos=q_pos)
        return x + (gate.unsqueeze(1) * out if self.use_adaln else out)

    def _run_ca(self, x, sources, source_positions, source_masks, q_pos):
        return x + self.ca(self.norm_ca(x), sources, rope=self.rope, q_pos=q_pos,
                           source_positions=source_positions, source_masks=source_masks)

    def _run_mlp(self, x, shift=None, scale=None, gate=None):
        h = modulate(self.norm2(x), shift, scale) if self.use_adaln else self.norm2(x)
        return x + (gate.unsqueeze(1) * self.mlp(h) if self.use_adaln else self.mlp(h))

    def forward(self, x, sources, source_positions=None, source_masks=None, cond=None, q_pos=None):
        """
        x:                (B, N, hidden_size)
        sources:          list of (B, N_i, hidden_size)
        source_positions: list of (t_ids, h_ids, w_ids) or None per source
        source_masks:     list of (B, N_i) bool (True=ignore) or None per source
        cond:             (B, hidden_size) adaLN conditioning — required when use_adaln=True
        q_pos:            (t_ids, h_ids, w_ids) for 3D RoPE on queries, or None
        """
        if not self.use_adaln and cond is not None:
            warnings.warn("cond passed to a Block with use_adaln=False — it will be ignored", stacklevel=2)

        shift_msa = scale_msa = gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        if self.use_adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)

        if self.sa_first:
            x = self._run_sa(x, shift_msa, scale_msa, gate_msa, q_pos)
            x = self._run_ca(x, sources, source_positions, source_masks, q_pos)
        else:
            # CA before SA: used when queries have no input-dependent content (e.g. learned
            # action query tokens in the IDM). Running SA first would be a no-op in block 0
            # since all queries are near-identical — CA must populate them first.
            x = self._run_ca(x, sources, source_positions, source_masks, q_pos)
            x = self._run_sa(x, shift_msa, scale_msa, gate_msa, q_pos)

        return self._run_mlp(x, shift_mlp, scale_mlp, gate_mlp)
