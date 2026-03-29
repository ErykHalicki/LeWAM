# --------------------------------------------------------
# Inverse Dynamics Model (IDM)
#
# Architecture: CLS + state token + past tokens + future tokens → SA → CLS → action latent
#
# IDM forward(past_frames, future_frames, state):
#   past_frames   (B, C*T*H*W, in_dim)  past frame patch embeddings (all cameras concatenated)
#   future_frames (B, C*K*H*W, in_dim)  predicted (or GT) future patch embeddings (all cameras)
#   state         (B, state_dim)|None   proprioceptive state vector
#   → returns     (B, num_future_frames, action_latent_dim)
#
# For multi-camera, concatenate all cameras' tokens along dim=1 before calling forward.
# RoPE position ids are repeated per camera (same spatial/temporal ids for each camera).
# CLS and state tokens receive zero RoPE positions (identity rotation).
# --------------------------------------------------------

import torch.nn as nn
import torch

from wam.models.common import Block, PatchPositionIds, RoPE3D


class IDM(nn.Module):
    """
    Inverse Dynamics Model: predicts action latents given past and future patch embeddings.

    p(a_latent | z_past, z_future, s)

    Input sequence: [CLS, state_token, past_tokens(RoPE), future_tokens(RoPE)]
    Output: CLS token projected to action latents → (B, num_future_frames, action_latent_dim)
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
        lang_dim=512,
    ):
        super().__init__()
        self.num_past_frames   = num_past_frames
        self.num_future_frames = num_future_frames
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.cls_token   = nn.Parameter(torch.zeros(1, num_future_frames, hidden_size))
        self.past_proj   = nn.Linear(in_dim, hidden_size)
        self.future_proj = nn.Linear(in_dim, hidden_size)
        self.state_proj  = nn.Linear(state_dim, hidden_size)

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_adaln=False, sa_first=True, no_ca=True)
            for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(hidden_size, eps=1e-6)
        self.out_proj = nn.Linear(hidden_size, action_latent_dim)

        self.rope      = RoPE3D(hidden_size // num_heads)
        self.past_pos  = PatchPositionIds(0, num_past_frames, patch_h, patch_w, fps)
        self.future_pos = PatchPositionIds(num_past_frames, num_future_frames, patch_h, patch_w, fps)

    def set_fps(self, fps: float):
        self.past_pos.set_fps(fps)
        self.future_pos.set_fps(fps)

    def set_patch_grid(self, H: int, W: int):
        self.patch_h = H
        self.patch_w = W
        self.past_pos.set_patch_grid(H, W)
        self.future_pos.set_patch_grid(H, W)

    def forward(self, past_frames, future_frames, state=None, lang=None, l_mask=None):
        """
        past_frames:   (B, C*T*H*W, in_dim)
        future_frames: (B, C*K*H*W, in_dim)
        state:         (B, state_dim)|None
        → (B, num_future_frames, action_latent_dim)
        """
        B = past_frames.shape[0]
        tokens_per_past   = self.num_past_frames   * self.patch_h * self.patch_w
        tokens_per_future = self.num_future_frames * self.patch_h * self.patch_w
        num_cameras = past_frames.shape[1] // tokens_per_past

        past        = self.past_proj(past_frames)
        future      = self.future_proj(future_frames)
        state_token = self.state_proj(state).unsqueeze(1) if state is not None else None
        class_token_batch = self.cls_token.expand(B, -1, -1)

        # Build per-token RoPE position ids for the full sequence.
        # CLS and state get zero positions (RoPE identity). Past/future get spatial-temporal ids.
        t_past, h_past, w_past = self.past_pos.pos
        t_fut,  h_fut,  w_fut  = self.future_pos.pos
        t_past = t_past.repeat(num_cameras)
        h_past = h_past.repeat(num_cameras)
        w_past = w_past.repeat(num_cameras)
        t_fut  = t_fut.repeat(num_cameras)
        h_fut  = h_fut.repeat(num_cameras)
        w_fut  = w_fut.repeat(num_cameras)

        n_special = self.num_future_frames + (1 if state is not None else 0)
        zeros = torch.zeros(n_special, device=past_frames.device)
        t_all = torch.cat([zeros, t_past, t_fut])
        h_all = torch.cat([zeros, h_past, h_fut])
        w_all = torch.cat([zeros, w_past, w_fut])

        parts = [class_token_batch]
        if state_token is not None:
            parts.append(state_token)
        parts.extend([past, future])
        x = torch.cat(parts, dim=1)

        for block in self.blocks:
            x = block(x, sources=[], q_pos=(t_all, h_all, w_all))

        return self.out_proj(self.norm(x[:, :self.num_future_frames]))


def IDM_L(**kwargs):
    return IDM(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def IDM_B(**kwargs):
    return IDM(depth=12, hidden_size=768, num_heads=12, **kwargs)

def IDM_S(**kwargs):
    return IDM(depth=12, hidden_size=384, num_heads=6, **kwargs)

def IDM_Baby(**kwargs):
    return IDM(depth=2, hidden_size=64, num_heads=4, **kwargs)

IDM_models = {
    'IDM-L':    IDM_L,
    'IDM-B':    IDM_B,
    'IDM-S':    IDM_S,
    'IDM-Baby': IDM_Baby,
}
