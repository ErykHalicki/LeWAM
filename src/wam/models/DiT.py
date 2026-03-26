# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
#
# Architecture: Flow Matching Transformer for video world modelling.
#
# DiT forward(x, t, past_frames, l, state, l_mask):
#   x           (B, K*H*W, in_dim)   noisy future VJEPA2 patch embeddings
#   t           (B,)                 flow matching timestep in [0, 1]
#   past_frames (B, T*H*W, in_dim)   clean past/current frame patch embeddings
#   l           (B, S, lang_dim)     pre-computed frozen language token embeddings
#   state       (B, state_dim)       proprioceptive state vector
#   l_mask      (B, S)               key padding mask for language (True = ignore)
#   → returns   (B, K*H*W, in_dim)   predicted velocity field (x1 - x0)
#
# Positional encoding:
#   - Past frames + noisy future x: 3D RoPE over (T, H, W), applied post-projection
#   - Language tokens: no positional encoding (encoder already encodes order)
#   - State token: no positional encoding (single vector, no sequence structure)
#
# Each DiTBlock: SA (3D RoPE) → CA → MLP
#   SA:  self-attention on noisy future patches with 3D RoPE
#   CA:  cross-attention to [past_frames (3D RoPE), language (no RoPE), state (no RoPE)]
#        each source has independent K/V projections inside CrossAttention
#   MLP: standard feedforward; adaLN-Zero conditioning from timestep on SA and MLP
#
# Flow matching loss (not in this file): MSE(v_pred, x1 - x0)
#   where x_t = (1 - t) * x0 + t * x1, x0 ~ N(0, I)
# --------------------------------------------------------

import math

import torch
import torch.nn as nn

from wam.models.common import RoPE3D, SelfAttention, CrossAttention, make_mlp, modulate


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size=768, frequency_embedding_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DiTBlock(nn.Module):
    """DiT block: SA (3D RoPE) -> CA (past_frames with RoPE, language, state) -> MLP."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.rope = RoPE3D(hidden_size // num_heads)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sa = SelfAttention(hidden_size, num_heads)

        self.norm_ca = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ca = CrossAttention(hidden_size, num_heads, num_sources=3)  # past_frames, language, state

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = make_mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, t, past_frames, l, state_token, x_pos, pf_pos, l_mask=None):
        """
        x:           (B, N, hidden_size)
        t:           (B, hidden_size)       timestep conditioning
        past_frames: (B, M, hidden_size)
        l:           (B, S, hidden_size)
        state_token: (B, 1, hidden_size)
        x_pos:       (t_ids, h_ids, w_ids) each (N,)
        pf_pos:      (t_ids, h_ids, w_ids) each (M,)
        l_mask:      (B, S) True = ignore
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.sa(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=self.rope, pos=x_pos,
        )
        x = x + self.ca(
            self.norm_ca(x),
            sources=[past_frames, l, state_token],
            rope=self.rope,
            q_pos=x_pos,
            source_positions=[pf_pos, None, None],
            source_masks=[None, l_mask, None],
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """
    Flow Matching Transformer on pre-extracted VJEPA2 patch embeddings.

    Predicts velocity field v(x_t, t, past_frames, l, state) for flow matching:
        x_t = (1 - t) * x0 + t * x1,  target = x1 - x0

    3D RoPE (T, H, W) applied to self-attention and past-frame cross-attention.
    Language tokens and state token have no positional encoding.

    x:           (B, K*H*W, in_dim)
    t:           (B,)
    past_frames: (B, T*H*W, in_dim)
    l:           (B, S, lang_dim)
    state:       (B, state_dim)
    """
    def __init__(
        self,
        num_frames,
        num_past_frames,
        patch_h,
        patch_w,
        in_dim=768,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        lang_dim=768,
        state_dim=64,
        language_dropout_prob=0.15,
        state_dropout_prob=0.15,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.language_dropout_prob = language_dropout_prob
        self.state_dropout_prob = state_dropout_prob

        self.x_embedder    = nn.Linear(in_dim, hidden_size)
        self.t_embedder    = TimestepEmbedder(hidden_size)
        self.context_proj  = nn.Linear(in_dim, hidden_size)
        self.lang_proj     = nn.Linear(lang_dim, hidden_size)
        self.state_proj    = nn.Linear(state_dim, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, in_dim)

        self.register_buffer('future_t_ids', self._t_ids(num_past_frames, num_frames, patch_h, patch_w), persistent=False)
        self.register_buffer('future_h_ids', self._h_ids(num_frames, patch_h, patch_w),                  persistent=False)
        self.register_buffer('future_w_ids', self._w_ids(num_frames, patch_h, patch_w),                  persistent=False)
        self.register_buffer('past_t_ids',   self._t_ids(0, num_past_frames, patch_h, patch_w),          persistent=False)
        self.register_buffer('past_h_ids',   self._h_ids(num_past_frames, patch_h, patch_w),             persistent=False)
        self.register_buffer('past_w_ids',   self._w_ids(num_past_frames, patch_h, patch_w),             persistent=False)

        self.initialize_weights()

    @staticmethod
    def _t_ids(t_start, num_frames, H, W):
        return torch.arange(t_start, t_start + num_frames).repeat_interleave(H * W)

    @staticmethod
    def _h_ids(num_frames, H, W):
        return torch.arange(H).repeat_interleave(W).repeat(num_frames)

    @staticmethod
    def _w_ids(num_frames, H, W):
        return torch.arange(W).repeat(H * num_frames)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, past_frames, l, state, l_mask=None):
        """
        x:           (B, K*H*W, in_dim)
        t:           (B,)
        past_frames: (B, T*H*W, in_dim)
        l:           (B, S, lang_dim)
        state:       (B, state_dim)
        l_mask:      (B, S) True = ignore
        """
        if self.training and self.language_dropout_prob > 0:
            drop = torch.rand(l.shape[0], device=l.device) < self.language_dropout_prob
            l = l.masked_fill(drop.view(-1, 1, 1), 0.0)
        if self.training and self.state_dropout_prob > 0:
            drop = torch.rand(state.shape[0], device=state.device) < self.state_dropout_prob
            state = state.masked_fill(drop.unsqueeze(1), 0.0)

        x           = self.x_embedder(x)                        # (B, K*H*W, hidden_size)
        past_frames = self.context_proj(past_frames)             # (B, T*H*W, hidden_size)
        l           = self.lang_proj(l)                          # (B, S, hidden_size)
        state_token = self.state_proj(state).unsqueeze(1)        # (B, 1, hidden_size)
        t_emb       = self.t_embedder(t)                         # (B, hidden_size)

        x_pos  = (self.future_t_ids, self.future_h_ids, self.future_w_ids)
        pf_pos = (self.past_t_ids,   self.past_h_ids,   self.past_w_ids)

        for block in self.blocks:
            x = block(x, t_emb, past_frames, l, state_token, x_pos, pf_pos, l_mask)

        return self.final_layer(x, t_emb)                        # (B, K*H*W, in_dim)


def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_Baby(**kwargs):
    return DiT(depth=2, hidden_size=64, num_heads=4, **kwargs)

DiT_models = {
    'DiT-XL':   DiT_XL,
    'DiT-L':    DiT_L,
    'DiT-B':    DiT_B,
    'DiT-S':    DiT_S,
    'DiT-Baby': DiT_Baby,
}
