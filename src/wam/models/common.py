import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def make_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(approximate="tanh"),
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
