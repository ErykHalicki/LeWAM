# --------------------------------------------------------
# Inverse Dynamics Model (IDM)
#
# IDM forward(current_frames, future_frames, state):
#   current_frames (B, T*H*W, in_dim)   current frame patch embeddings
#   future_frames  (B, K*H*W, in_dim)   predicted (or GT) future patch embeddings
#   state          (B, state_dim)       proprioceptive state vector
#   → returns      (B, chunk_len, action_dim)   predicted action chunk
#
# Architecture: Transformer with learned action query tokens.
#   Each IDMBlock: SA (bidirectional, no RoPE) → CA → MLP
#   CA cross-attends to: current_frames (3D RoPE), future_frames (3D RoPE), state (no RoPE)
#
# Positional encoding:
#   - current/future patches: 3D RoPE over (T, H, W) applied in cross-attention K/V
#   - action queries: learned 1D position embeddings (temporal order within chunk)
#   - state token: no positional encoding
# --------------------------------------------------------

import torch
import torch.nn as nn

from wam.models.common import RoPE3D, SelfAttention, CrossAttention, make_mlp


class IDMBlock(nn.Module):
    """IDM block: SA (bidirectional) → CA (current, future, state) → MLP."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.rope = RoPE3D(hidden_size // num_heads)

        self.norm1  = nn.LayerNorm(hidden_size, eps=1e-6)
        self.sa     = SelfAttention(hidden_size, num_heads)

        self.norm_ca = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ca      = CrossAttention(hidden_size, num_heads, num_sources=3)

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp   = make_mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size)

    def forward(self, q, current, future, state_token, cur_pos, fut_pos):
        """
        q:           (B, chunk_len, hidden_size)
        current:     (B, T*H*W, hidden_size)
        future:      (B, K*H*W, hidden_size)
        state_token: (B, 1, hidden_size)
        cur_pos:     (t_ids, h_ids, w_ids) for current frames
        fut_pos:     (t_ids, h_ids, w_ids) for future frames
        """
        q = q + self.sa(self.norm1(q))
        q = q + self.ca(
            self.norm_ca(q),
            sources=[current, future, state_token],
            rope=self.rope,
            q_pos=None,
            source_positions=[cur_pos, fut_pos, None],
            source_masks=None,
        )
        q = q + self.mlp(self.norm2(q))
        return q


class IDM(nn.Module):
    """
    Inverse Dynamics Model: predicts an action chunk given current and future patch embeddings.

    p(a_t | z_t, z_{t+1}, s_t)

    current_frames: (B, T*H*W, in_dim)
    future_frames:  (B, K*H*W, in_dim)
    state:          (B, state_dim)
    """
    def __init__(
        self,
        num_past_frames,
        num_future_frames,
        patch_h,
        patch_w,
        in_dim=768,
        hidden_size=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        state_dim=64,
        action_dim=7,
        action_chunk_len=16,
    ):
        super().__init__()
        self.current_proj = nn.Linear(in_dim, hidden_size)
        self.future_proj  = nn.Linear(in_dim, hidden_size)
        self.state_proj   = nn.Linear(state_dim, hidden_size)

        self.action_queries = nn.Parameter(torch.randn(1, action_chunk_len, hidden_size) * 0.02)
        self.action_pos_emb = nn.Parameter(torch.randn(1, action_chunk_len, hidden_size) * 0.02)

        self.blocks = nn.ModuleList([
            IDMBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(hidden_size, eps=1e-6)
        self.out_proj = nn.Linear(hidden_size, action_dim)

        self.register_buffer('cur_t_ids', self._t_ids(0,               num_past_frames,   patch_h, patch_w), persistent=False)
        self.register_buffer('cur_h_ids', self._h_ids(num_past_frames,                    patch_h, patch_w), persistent=False)
        self.register_buffer('cur_w_ids', self._w_ids(num_past_frames,                    patch_h, patch_w), persistent=False)
        self.register_buffer('fut_t_ids', self._t_ids(num_past_frames, num_future_frames,  patch_h, patch_w), persistent=False)
        self.register_buffer('fut_h_ids', self._h_ids(num_future_frames,                   patch_h, patch_w), persistent=False)
        self.register_buffer('fut_w_ids', self._w_ids(num_future_frames,                   patch_h, patch_w), persistent=False)

        self._init_weights()

    @staticmethod
    def _t_ids(t_start, num_frames, H, W):
        return torch.arange(t_start, t_start + num_frames).repeat_interleave(H * W)

    @staticmethod
    def _h_ids(num_frames, H, W):
        return torch.arange(H).repeat_interleave(W).repeat(num_frames)

    @staticmethod
    def _w_ids(num_frames, H, W):
        return torch.arange(W).repeat(H * num_frames)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, current_frames, future_frames, state):
        B = current_frames.shape[0]

        current     = self.current_proj(current_frames)
        future      = self.future_proj(future_frames)
        state_token = self.state_proj(state).unsqueeze(1)

        cur_pos = (self.cur_t_ids, self.cur_h_ids, self.cur_w_ids)
        fut_pos = (self.fut_t_ids, self.fut_h_ids, self.fut_w_ids)

        q = self.action_queries.expand(B, -1, -1) + self.action_pos_emb

        for block in self.blocks:
            q = block(q, current, future, state_token, cur_pos, fut_pos)

        return self.out_proj(self.norm(q))


def IDM_L(**kwargs):
    return IDM(depth=12, hidden_size=1024, num_heads=16, **kwargs)

def IDM_B(**kwargs):
    return IDM(depth=6, hidden_size=768, num_heads=12, **kwargs)

def IDM_S(**kwargs):
    return IDM(depth=4, hidden_size=384, num_heads=6, **kwargs)

def IDM_Baby(**kwargs):
    return IDM(depth=2, hidden_size=64, num_heads=4, **kwargs)

IDM_models = {
    'IDM-L':    IDM_L,
    'IDM-B':    IDM_B,
    'IDM-S':    IDM_S,
    'IDM-Baby': IDM_Baby,
}
