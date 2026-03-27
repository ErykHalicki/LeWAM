# --------------------------------------------------------
# Inverse Dynamics Model (IDM)
#
# IDM forward(past_frames, future_frames, state, aux_frames):
#   past_frames  (B, T*H*W, in_dim)              past frame patch embeddings (main camera)
#   future_frames (B, K*H*W, in_dim)             predicted (or GT) future patch embeddings
#   state         (B, state_dim)|None            proprioceptive state vector
#   aux_frames    (B, C*T*H*W, in_dim)|None      auxiliary camera past frames (not predicted)
#   → returns     (B, chunk_len, action_latent_dim)  action latents (pass to ActionDecoder for actions)
#
# Each Block: SA (bidirectional, no RoPE, no adaLN) → CA (past+RoPE, future+RoPE, state, aux+RoPE) → MLP
# Loss (not in this file): MSE(a_pred, a_gt)
# --------------------------------------------------------

import torch
import torch.nn as nn

from wam.models.common import Block, make_mlp


class IDM(nn.Module):
    """
    Inverse Dynamics Model: predicts an action chunk given past and future patch embeddings.

    p(a_latent | z_t, z_{t+1}, s_t)

    Outputs action latents (B, K, action_latent_dim). Pass through ActionDecoder to get actions.

    past_frames:  (B, T*H*W, in_dim)        main camera
    future_frames: (B, K*H*W, in_dim)
    state:         (B, state_dim)|None
    aux_frames:    (B, C*T*H*W, in_dim)|None
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
        action_latent_dim=64,
    ):
        super().__init__()
        self.num_past_frames = num_past_frames
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.past_proj   = nn.Linear(in_dim, hidden_size)
        self.future_proj = nn.Linear(in_dim, hidden_size)
        self.state_proj  = nn.Linear(state_dim, hidden_size)
        self.aux_proj    = make_mlp(in_dim, hidden_size, hidden_size)

        self.action_queries = nn.Parameter(torch.randn(1, num_future_frames, hidden_size) * 0.02)
        self.action_pos_emb = nn.Parameter(torch.randn(1, num_future_frames, hidden_size) * 0.02)

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, num_sources=4, mlp_ratio=mlp_ratio, use_adaln=False, sa_first=False)
            for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(hidden_size, eps=1e-6)
        self.out_proj = nn.Linear(hidden_size, action_latent_dim)

        self.register_buffer('past_t_ids', self._t_ids(0,               num_past_frames,   patch_h, patch_w), persistent=False)
        self.register_buffer('past_h_ids', self._h_ids(num_past_frames,                    patch_h, patch_w), persistent=False)
        self.register_buffer('past_w_ids', self._w_ids(num_past_frames,                    patch_h, patch_w), persistent=False)
        self.register_buffer('fut_t_ids',  self._t_ids(num_past_frames, num_future_frames,  patch_h, patch_w), persistent=False)
        self.register_buffer('fut_h_ids',  self._h_ids(num_future_frames,                   patch_h, patch_w), persistent=False)
        self.register_buffer('fut_w_ids',  self._w_ids(num_future_frames,                   patch_h, patch_w), persistent=False)

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

    def forward(self, past_frames, future_frames, state=None, aux_frames=None):
        """
        past_frames:  (B, T*H*W, in_dim)        main camera — always required
        future_frames: (B, K*H*W, in_dim)        main camera predicted future — always required
        state:         (B, state_dim)|None
        aux_frames:    (B, C*T*H*W, in_dim)|None auxiliary cameras (not predicted)
        """
        B = past_frames.shape[0]

        past        = self.past_proj(past_frames)
        future      = self.future_proj(future_frames)
        state_token = self.state_proj(state).unsqueeze(1) if state is not None else None

        past_pos = (self.past_t_ids, self.past_h_ids, self.past_w_ids)
        fut_pos  = (self.fut_t_ids,  self.fut_h_ids,  self.fut_w_ids)

        if aux_frames is not None:
            num_aux = aux_frames.shape[1] // (self.num_past_frames * self.patch_h * self.patch_w)
            aux = self.aux_proj(aux_frames)
            aux_pos = (
                self.past_t_ids.repeat(num_aux),
                self.past_h_ids.repeat(num_aux),
                self.past_w_ids.repeat(num_aux),
            )
        else:
            aux     = None
            aux_pos = None

        q = self.action_queries.expand(B, -1, -1) + self.action_pos_emb

        for block in self.blocks:
            q = block(
                q,
                sources=[past, future, state_token, aux],
                source_positions=[past_pos, fut_pos, None, aux_pos],
                cond=None,
                q_pos=None,
            )

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
