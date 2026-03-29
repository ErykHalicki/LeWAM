# --------------------------------------------------------
# Inverse Dynamics Model (IDM)
#
# IDM forward(past_frames, future_frames, state):
#   past_frames   (B, C*T*H*W, in_dim)  past frame patch embeddings (all cameras concatenated)
#   future_frames (B, C*K*H*W, in_dim)  predicted (or GT) future patch embeddings (all cameras)
#   state         (B, state_dim)|None   proprioceptive state vector
#   → returns     (B, chunk_len, action_latent_dim)  action latents (pass to ActionDecoder for actions)
#
# For multi-camera, concatenate all cameras' tokens along dim=1 before calling forward.
# RoPE position ids are repeated per camera (same spatial/temporal ids for each camera).
#
# Each Block: CA (past+RoPE, future+RoPE, state) → SA (bidirectional, no RoPE, no adaLN) → MLP
# Loss (not in this file): MSE(a_pred, a_gt)
# --------------------------------------------------------

import torch
import torch.nn as nn

from wam.models.common import Block, PatchPositionIds


class IDM(nn.Module):
    """
    Inverse Dynamics Model: predicts an action chunk given past and future patch embeddings.

    p(a_latent | z_t, z_{t+1}, s_t)

    Outputs action latents (B, K, action_latent_dim). Pass through ActionDecoder to get actions.

    past_frames:   (B, C*T*H*W, in_dim)  all cameras concatenated along token dim
    future_frames: (B, C*K*H*W, in_dim)  all cameras concatenated along token dim
    state:         (B, state_dim)|None
    """
    def __init__(
        self,
        num_past_frames,
        num_future_frames,
        patch_h,
        patch_w,
        fps=30.0,
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
        self.num_future_frames = num_future_frames
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.past_proj   = nn.Linear(in_dim, hidden_size)
        self.future_proj = nn.Linear(in_dim, hidden_size)
        self.state_proj  = nn.Linear(state_dim, hidden_size)

        self.action_queries = nn.Parameter(torch.randn(1, num_future_frames, hidden_size) * 0.02)
        self.action_pos_emb = nn.Parameter(torch.randn(1, num_future_frames, hidden_size) * 0.02)

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, num_sources=3, mlp_ratio=mlp_ratio, use_adaln=False, sa_first=False)
            for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(hidden_size, eps=1e-6)
        self.out_proj = nn.Linear(hidden_size, action_latent_dim)

        self.past_pos   = PatchPositionIds(0, num_past_frames, patch_h, patch_w, fps)
        self.future_pos = PatchPositionIds(num_past_frames, num_future_frames, patch_h, patch_w, fps)

        self._init_weights()

    def set_fps(self, fps: float):
        self.past_pos.set_fps(fps)
        self.future_pos.set_fps(fps)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, past_frames, future_frames, state=None):
        """
        past_frames:   (B, C*T*H*W, in_dim)  all cameras concatenated (C=1 for single camera)
        future_frames: (B, C*K*H*W, in_dim)  all cameras concatenated
        state:         (B, state_dim)|None
        """
        B = past_frames.shape[0]

        tokens_per_past   = self.num_past_frames * self.patch_h * self.patch_w
        tokens_per_future = self.num_future_frames * self.patch_h * self.patch_w
        num_cameras_past   = past_frames.shape[1] // tokens_per_past
        num_cameras_future = future_frames.shape[1] // tokens_per_future
        assert num_cameras_past == num_cameras_future, (
            f"Camera count mismatch: past has {num_cameras_past}, future has {num_cameras_future}"
        )

        past        = self.past_proj(past_frames)
        future      = self.future_proj(future_frames)
        state_token = self.state_proj(state).unsqueeze(1) if state is not None else None

        t, h, w = self.past_pos.pos
        past_pos = (t.repeat(num_cameras_past), h.repeat(num_cameras_past), w.repeat(num_cameras_past))

        t, h, w = self.future_pos.pos
        fut_pos = (t.repeat(num_cameras_future), h.repeat(num_cameras_future), w.repeat(num_cameras_future))

        q = self.action_queries.expand(B, -1, -1) + self.action_pos_emb

        for block in self.blocks:
            q = block(
                q,
                sources=[past, future, state_token],
                source_positions=[past_pos, fut_pos, None],
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
