import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wam.models.common import (
    Block, PatchPositionIds, concat_pos_ids,
    modulate, make_mlp,
)
from wam.models.video_encoder import build_vjepa2_encoder_arch
from wam.models.vlm_encoder import VLMEncoder
from wam.models.action_encoders import StateEncoder, ActionEncoder, ActionPreprocessor


class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = make_mlp(dim, dim, dim)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.dim).to(t.dtype))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class LeWAM(nn.Module):
    """
    Joint video-latent + action flow matching model.

    Sequence: [context_video | future_video | action_tokens]
      Context (clean):  causal self-attention within context.
      Future  (noisy):  sees all context + causal within future temporal blocks.
      Actions (noisy):  sees all context + aligned future blocks + causal actions.

    Cross-attention sources: state token + language tokens (from VLM).
    adaLN-Zero conditioning: timestep embedding only.
    3D RoPE: video get (t, h, w), actions get (t, 0, 0).
    """

    VJEPA_DIM = 768
    VJEPA_TUBELET_SIZE = 2
    VJEPA_PATCH_SIZE = 16

    SIZES = {
        "large": {"model_dim": 1024, "depth": 16, "num_heads": 16},
        "base":  {"model_dim": 768,  "depth": 12, "num_heads": 12},
        "small": {"model_dim": 512,  "depth": 8,  "num_heads": 8},
        "baby":  {"model_dim": 256,  "depth": 6,  "num_heads": 4},
    }

    def __init__(
        self,
        model_dim=512,
        depth=8,
        num_heads=8,
        num_context_frames=16,
        num_future_frames=32,
        frame_latent_h=24,
        frame_latent_w=24,
        fps=15.0,
        action_fps=None,
        action_dim=6,
        state_dim=6,
        vlm_model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        vlm_num_layers=8,
        stats_path=None,
        norm_strategy="q1_q99",
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.config = {
            "model_dim": model_dim, "depth": depth, "num_heads": num_heads,
            "num_context_frames": num_context_frames,
            "num_future_frames": num_future_frames,
            "frame_latent_h": frame_latent_h, "frame_latent_w": frame_latent_w,
            "fps": fps, "action_fps": action_fps,
            "action_dim": action_dim, "state_dim": state_dim,
            "vlm_model_id": vlm_model_id, "vlm_num_layers": vlm_num_layers,
            "stats_path": stats_path, "norm_strategy": norm_strategy,
            "mlp_ratio": mlp_ratio,
        }

        self.model_dim = model_dim
        self.action_dim = action_dim
        self.num_context_frames = num_context_frames
        self.num_future_frames = num_future_frames
        self.frame_latent_h = frame_latent_h
        self.frame_latent_w = frame_latent_w
        self.fps = fps
        if action_fps is None:
            action_fps = 30
        self.action_fps = action_fps

        num_ctx_t = num_context_frames // self.VJEPA_TUBELET_SIZE
        num_fut_t = num_future_frames // self.VJEPA_TUBELET_SIZE
        assert num_ctx_t % 2 == 0, f"Context tubelets ({num_ctx_t}) must be even"
        assert num_fut_t % 2 == 0, f"Future tubelets ({num_fut_t}) must be even"
        self.num_context_tubelets = num_ctx_t
        self.num_future_tubelets = num_fut_t

        future_duration = num_future_frames / fps
        action_horizon = int(future_duration * action_fps)
        assert action_horizon % 2 == 0, f"Action horizon ({action_horizon}) must be even"
        self.action_horizon = action_horizon

        spatial = frame_latent_h * frame_latent_w
        self.N_ctx = num_ctx_t * spatial
        self.N_fut = num_fut_t * spatial
        self.N_act = action_horizon

        # ── Encoders ──────────────────────────────────────────────────────
        self.video_encoder = build_vjepa2_encoder_arch(
            crop_size=frame_latent_h * self.VJEPA_PATCH_SIZE,
        )
        self.video_encoder.set_frozen(True)

        self.vlm_encoder = (
            VLMEncoder(model_id=vlm_model_id, num_layers=vlm_num_layers,
                       model_dim=model_dim)
            if vlm_model_id else None
        )

        # ── Projections ───────────────────────────────────────────────────
        self.video_proj = nn.Linear(self.VJEPA_DIM + 1, model_dim)
        self.action_encoder = ActionEncoder(action_dim, model_dim)
        self.state_encoder = StateEncoder(state_dim, model_dim)
        self.t_embedder = TimestepEmbedder(model_dim)

        # ── Transformer ──────────────────────────────────────────────────
        # Cross-attention sources: state (always) + language (when VLM present)
        num_ca_sources = 2 if self.vlm_encoder is not None else 1
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_sources=num_ca_sources,
                  mlp_ratio=mlp_ratio, use_adaln=True)
            for _ in range(depth)
        ])

        # ── Output heads ─────────────────────────────────────────────────
        self.video_final = FinalLayer(model_dim, self.VJEPA_DIM)
        self.action_final = FinalLayer(model_dim, action_dim)

        # ── Action normalization ─────────────────────────────────────────
        self.action_preprocessor = (
            ActionPreprocessor(stats_path, norm_strategy) if stats_path else None
        )

        # ── Position IDs ─────────────────────────────────────────────────
        self.context_pos = PatchPositionIds(
            num_ctx_t, frame_latent_h, frame_latent_w, fps, t_offset=0,
        )
        self.future_pos = PatchPositionIds(
            num_fut_t, frame_latent_h, frame_latent_w, fps, t_offset=num_ctx_t,
        )
        self._register_action_pos(num_ctx_t, fps, action_horizon, action_fps)

        # ── Attention mask ───────────────────────────────────────────────
        self.register_buffer(
            "attn_mask",
            self._build_attn_mask(self.N_ctx, self.N_fut, action_horizon, spatial),
            persistent=False,
        )

        self._init_weights()

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_size(cls, size="small", **kwargs):
        """Build from a named size preset: 'base', 'small', or 'baby'."""
        return cls(**cls.SIZES[size], **kwargs)

    @classmethod
    def from_checkpoint(cls, path, **overrides):
        """Rebuild from a saved checkpoint. Pass overrides to fix paths etc."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = ckpt["config"] | overrides
        model = cls(**config)
        model.load_state_dict(ckpt["model"])
        return model

    def load_vjepa2_weights(self, checkpoint_path):
        """Load pretrained VJEPA2 encoder weights from a .pt file."""
        from wam.models.video_encoder import load_vjepa2_encoder
        loaded = load_vjepa2_encoder(
            checkpoint_path,
            crop_size=self.frame_latent_h * self.VJEPA_PATCH_SIZE,
        )
        self.video_encoder.backbone.load_state_dict(loaded.backbone.state_dict())
        self.video_encoder.set_frozen(True)

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode_video(self, frames):
        """
        frames: (B, T, C, H, W)       single camera
                (B, N, T, C, H, W)    multi-camera
                list of N (B, T, C, H, W) tensors

        Each camera is center-cropped to crop_size x crop_size and stitched
        along the width before encoding. Returns:
            (B, T//tubelet * frame_latent_h * frame_latent_w * N, vjepa_dim)
        """
        preprocessed = self.video_encoder.preprocessor(frames)
        with torch.no_grad():
            return self.video_encoder(preprocessed)

    def encode_language(self, texts, images=None):
        """
        Returns (tokens, mask) or (None, None) if no VLM.
        tokens: (B, S, model_dim)   mask: (B, S) bool True=ignore
        """
        if self.vlm_encoder is None:
            return None, None
        return self.vlm_encoder(texts, images=images)

    # ── Normalization ─────────────────────────────────────────────────────

    def normalize_state(self, state):
        if self.action_preprocessor is not None:
            return self.action_preprocessor.normalize_state(state)
        return state

    def normalize_actions(self, actions):
        if self.action_preprocessor is not None:
            return self.action_preprocessor.normalize_rel_action(actions)
        return actions

    def unnormalize_actions(self, actions):
        if self.action_preprocessor is not None:
            return self.action_preprocessor.unnormalize_rel_action(actions)
        return actions

    # ── Forward ───────────────────────────────────────────────────────────

    def _pos_ids(self):
        return concat_pos_ids(
            self.context_pos.pos,
            self.future_pos.pos,
            (self.action_t_ids, self.action_h_ids, self.action_w_ids),
        )

    def forward(self, x_t_video, x_t_action, context_tokens, t, state,
                lang_tokens=None, lang_mask=None):
        """
        x_t_video:      (B, N_fut, vjepa_dim)  noisy future tokens
        x_t_action:     (B, N_act, action_dim) noisy actions
        context_tokens: (B, N_ctx, vjepa_dim)  clean context tokens
        t:              (B,)                   flow-matching timestep [0,1]
        state:          (B, state_dim)         normalized state
        lang_tokens:    (B, S, model_dim) | None
        lang_mask:      (B, S) bool True=ignore | None

        Returns (video_vel, action_vel):
            video_vel:  (B, N_fut, vjepa_dim)
            action_vel: (B, N_act, action_dim)
        """
        B = t.shape[0]

        ctx_flag = context_tokens.new_ones(B, self.N_ctx, 1)
        fut_flag = x_t_video.new_zeros(B, self.N_fut, 1)
        video_in = torch.cat([
            torch.cat([context_tokens, ctx_flag], dim=-1),
            torch.cat([x_t_video, fut_flag], dim=-1),
        ], dim=1)

        x = torch.cat([
            self.video_proj(video_in),
            self.action_encoder(x_t_action),
        ], dim=1)

        cond = self.t_embedder(t)
        state_token = self.state_encoder(state).unsqueeze(1)

        pos = self._pos_ids()

        # Cross-attention: state (always idx 0) + language (idx 1 if VLM)
        if self.vlm_encoder is not None:
            sources = [state_token, lang_tokens]
            source_masks = [None, lang_mask]
        else:
            sources = [state_token]
            source_masks = [None]

        for block in self.blocks:
            x = block(
                x, sources=sources, source_masks=source_masks,
                cond=cond, attn_mask=self.attn_mask, pos=pos,
            )

        video_out = x[:, self.N_ctx : self.N_ctx + self.N_fut]
        action_out = x[:, self.N_ctx + self.N_fut :]

        return self.video_final(video_out, cond), self.action_final(action_out, cond)

    # ── Inference ─────────────────────────────────────────────────────────

    @staticmethod
    def smooth_actions(actions):
        """
        DreamZero-style action smoothing: upsample 2x, Savitzky-Golay filter, downsample.
        actions: (B, N_act, action_dim) tensor
        """
        from scipy.signal import savgol_filter

        device = actions.device
        dtype = actions.dtype
        B, N, D = actions.shape

        upsampled = F.interpolate(
            actions.float().permute(0, 2, 1),
            size=N * 2,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)

        up_np = upsampled.cpu().numpy()
        window = min(21, N * 2)
        if window % 2 == 0:
            window -= 1
        window = max(window, 5)
        poly_order = min(3, window - 1)

        for b in range(B):
            up_np[b] = savgol_filter(up_np[b], window, poly_order, axis=0)

        smoothed = torch.from_numpy(up_np).to(device=device)

        downsampled = F.interpolate(
            smoothed.permute(0, 2, 1),
            size=N,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)

        return downsampled.to(dtype=dtype)

    @torch.no_grad()
    def ode_solve(self, context_tokens, state,
                  lang_tokens=None, lang_mask=None, num_steps=10,
                  smooth=False):
        """
        Euler ODE integration from noise -> predicted future.
        Returns (pred_video, pred_actions).
        """
        B, device, dtype = (
            context_tokens.shape[0],
            context_tokens.device,
            context_tokens.dtype,
        )
        x_vid = torch.randn(B, self.N_fut, self.VJEPA_DIM, device=device, dtype=dtype)
        x_act = torch.randn(B, self.N_act, self.action_dim, device=device, dtype=dtype)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device, dtype=dtype)
            v_vid, v_act = self(
                x_vid, x_act, context_tokens, t, state, lang_tokens, lang_mask,
            )
            x_vid = x_vid + v_vid * dt
            x_act = x_act + v_act * dt

        if smooth:
            x_act = self.smooth_actions(x_act)

        return x_vid, x_act

    # ── Attention mask ────────────────────────────────────────────────────

    @staticmethod
    def _build_attn_mask(N_ctx, N_fut, N_act, spatial, block_size=2):
        """
        Block-causal boolean mask, True = attend.

        Sequence layout: [context C | future F | actions A]
        block_size: temporal block size in tubelets (default 2).
        Video blocks are block_size * spatial tokens.
        Action blocks are block_size tokens.

            Context[b]: context blocks 0..b.
            Future[b]:  all context + future blocks 0..b + action blocks 0..b.
            Action[b]:  all context + future blocks 0..b + action blocks 0..b.
        """
        C, F, A = N_ctx, N_fut, N_act
        N = C + F + A
        vbs = block_size * spatial
        abs_ = block_size
        num_ctx_blocks = C // vbs if vbs > 0 else 0
        num_fut_blocks = F // vbs
        num_act_blocks = A // abs_

        mask = torch.zeros(N, N, dtype=torch.bool)

        for i in range(num_ctx_blocks):
            r0 = i * vbs
            r1 = (i + 1) * vbs
            mask[r0:r1, :r1] = True

        for i in range(num_fut_blocks):
            r0 = C + i * vbs
            r1 = C + (i + 1) * vbs
            act_end = -(-((i + 1) * num_act_blocks) // num_fut_blocks) * abs_
            mask[r0:r1, :C] = True
            mask[r0:r1, C:r1] = True
            mask[r0:r1, C + F : C + F + act_end] = True

        for i in range(num_act_blocks):
            r0 = C + F + i * abs_
            r1 = C + F + (i + 1) * abs_
            fut_end_block = -(-((i + 1) * num_fut_blocks) // num_act_blocks)
            fut_end = C + fut_end_block * vbs
            mask[r0:r1, :C] = True
            mask[r0:r1, C:fut_end] = True
            mask[r0:r1, C + F : r1] = True

        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def visualize_attn_mask(N_ctx, N_fut, N_act, block_size=2):
        mask = LeWAM._build_attn_mask(N_ctx, N_fut, N_act, spatial=1, block_size=block_size).squeeze()
        labels = ['C'] * N_ctx + ['F'] * N_fut + ['A'] * N_act
        N = mask.shape[0]
        print('      ' + ' '.join(labels))
        for i in range(N):
            print(f'{labels[i]:<2} {i:>2}  ' + ' '.join('#' if mask[i, j] else '.' for j in range(N)))

    # ── Weight init ──────────────────────────────────────────────────────

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for module in (self.video_proj, self.action_encoder, self.state_encoder,
                       self.t_embedder, self.blocks, self.video_final,
                       self.action_final):
            module.apply(_basic)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        for final in (self.video_final, self.action_final):
            nn.init.zeros_(final.adaLN_modulation[-1].weight)
            nn.init.zeros_(final.adaLN_modulation[-1].bias)
            nn.init.zeros_(final.linear.weight)
            nn.init.zeros_(final.linear.bias)

    # ── Runtime config ────────────────────────────────────────────────────

    def set_action_fps(self, action_fps):
        action_horizon = int((self.num_future_frames / self.fps) * action_fps)
        assert action_horizon % 2 == 0, f"Action horizon ({action_horizon}) must be even"
        self.action_fps = action_fps
        self.action_horizon = action_horizon
        self.N_act = action_horizon
        self._register_action_pos(
            self.num_context_tubelets, self.fps, action_horizon, action_fps,
        )
        spatial = self.frame_latent_h * self.frame_latent_w
        self.attn_mask = self._build_attn_mask(
            self.N_ctx, self.N_fut, action_horizon, spatial,
        ).to(self.attn_mask.device)

    def set_fps(self, fps):
        self.fps = fps
        self.context_pos.set_fps(fps)
        self.future_pos.set_fps(fps)
        self._register_action_pos(
            self.num_context_tubelets, fps, self.action_horizon, self.action_fps,
        )

    def set_patch_grid(self, H, W):
        self.frame_latent_h, self.frame_latent_w = H, W
        self.context_pos.set_patch_grid(H, W)
        self.future_pos.set_patch_grid(H, W)
        spatial = H * W
        self.N_ctx = self.num_context_tubelets * spatial
        self.N_fut = self.num_future_tubelets * spatial
        self.attn_mask = self._build_attn_mask(
            self.N_ctx, self.N_fut, self.action_horizon, spatial,
        ).to(self.attn_mask.device)

    def count_params(self, millions=True, trainable_only=True):
        n = sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)
        return int(round(n * (1e-6 if millions else 1)))

    # ── Internal ─────────────────────────────────────────────────────────

    def _register_action_pos(self, num_ctx_t, fps, action_horizon, action_fps):
        ctx_duration = num_ctx_t / fps
        t = torch.tensor([
            ctx_duration + i / action_fps
            for i in range(action_horizon)
        ])
        self.register_buffer("action_t_ids", t, persistent=False)
        self.register_buffer("action_h_ids", torch.zeros(action_horizon),
                             persistent=False)
        self.register_buffer("action_w_ids", torch.zeros(action_horizon),
                             persistent=False)
