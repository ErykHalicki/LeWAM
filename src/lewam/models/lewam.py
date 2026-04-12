import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewam.models.common import (
    ActionPreprocessor, Block, PatchPositionIds, concat_pos_ids,
    modulate, make_mlp, _USE_FLEX,
)
from lewam.models.video_encoder import build_vjepa2_encoder_arch
from lewam.models.vlm_encoder import VLMEncoder

from torch.nn.attention.flex_attention import create_block_mask


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


class InputLayer(nn.Module):
    """Project input tokens to model_dim with adaLN timestep conditioning."""
    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, cond):
        x = self.linear(x)
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        return modulate(self.norm(x), shift, scale)


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

    def __init__(
        self,
        model_dim=512,
        depth=8,
        num_heads=8,
        num_context_frames=24,
        num_future_frames=8,
        frame_latent_h=16,
        frame_latent_w=16,
        fps=5.0,
        action_fps=30.0,
        action_dim=6,
        state_dim=6,
        vlm_model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        vlm_num_layers=8,
        norm_stats=None,
        norm_strategy="q1_q99",
        mlp_ratio=4.0,
        attn_block_size=1,
        action_only=False,
        interpolate_rope=True,
        _pretrained_vlm=True,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.interpolate_rope = interpolate_rope

        self.config = {
            "model_dim": model_dim, "depth": depth, "num_heads": num_heads,
            "num_context_frames": num_context_frames,
            "num_future_frames": num_future_frames,
            "frame_latent_h": frame_latent_h, "frame_latent_w": frame_latent_w,
            "fps": fps, "action_fps": action_fps,
            "action_dim": action_dim, "state_dim": state_dim,
            "vlm_model_id": vlm_model_id, "vlm_num_layers": vlm_num_layers,
            "norm_strategy": norm_strategy,
            "mlp_ratio": mlp_ratio,
            "attn_block_size": attn_block_size,
            "action_only": action_only,
        }

        self.model_dim = model_dim
        self.action_dim = action_dim
        self.attn_block_size = attn_block_size
        self.action_only = action_only
        self._video_only = False
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
        if not action_only:
            assert num_fut_t % 2 == 0, f"Future tubelets ({num_fut_t}) must be even"
        self.num_context_tubelets = num_ctx_t
        self.num_future_tubelets = num_fut_t

        future_duration = num_future_frames / fps
        action_horizon = int(future_duration * action_fps)
        assert action_horizon % 2 == 0, f"Action horizon ({action_horizon}) must be even"
        self.action_horizon = action_horizon

        spatial = frame_latent_h * frame_latent_w
        self.N_ctx = num_ctx_t * spatial
        self.N_fut = 0 if action_only else num_fut_t * spatial
        self.N_act = action_horizon

        # ── Encoders ──────────────────────────────────────────────────────
        self.video_encoder = build_vjepa2_encoder_arch(
            crop_size=frame_latent_h * self.VJEPA_PATCH_SIZE,
        )
        self.video_encoder.set_frozen(True)

        self.vlm_encoder = (
            VLMEncoder(model_id=vlm_model_id, num_layers=vlm_num_layers,
                       pretrained=_pretrained_vlm)
            if vlm_model_id else None
        )

        if self.vlm_encoder is not None:
            assert vlm_num_layers == depth, (
                f"LeWAM v0.3 requires vlm_num_layers == depth, got {vlm_num_layers} vs {depth}. "
                f"Each DiT block cross-attends to the corresponding VLM layer."
            )

        # ── Projections ───────────────────────────────────────────────────
        self.context_proj = nn.Linear(self.VJEPA_DIM, model_dim)
        self.future_proj = None if action_only else InputLayer(self.VJEPA_DIM, model_dim)
        self.action_encoder = InputLayer(action_dim, model_dim)
        self.t_embedder = TimestepEmbedder(model_dim)

        # ── Transformer ──────────────────────────────────────────────────
        if self.vlm_encoder is not None:
            num_ca_sources = 2
            sources_dim = [state_dim, self.vlm_encoder.vlm_hidden_dim]
        else:
            num_ca_sources = 1
            sources_dim = [state_dim]

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_sources=num_ca_sources,
                  mlp_ratio=mlp_ratio, use_adaln=True, dropout=0.0,
                  sources_dim=sources_dim)
            for _ in range(depth)
        ])

        # ── Output heads ─────────────────────────────────────────────────
        self.video_final = None if action_only else FinalLayer(model_dim, self.VJEPA_DIM)
        self.action_final = FinalLayer(model_dim, action_dim)

        # ── Action normalization ─────────────────────────────────────────
        if norm_stats is None:
            raise ValueError("norm_stats is required. Pass a stats dict or use from_checkpoint/from_pretrained.")
        self.action_preprocessor = ActionPreprocessor(norm_stats, norm_strategy)

        # ── Position IDs ─────────────────────────────────────────────────
        tubelet_fps = fps / self.VJEPA_TUBELET_SIZE
        self.context_pos = PatchPositionIds(
            num_ctx_t, frame_latent_h, frame_latent_w, tubelet_fps, t_offset=0,
        )
        self.future_pos = None if action_only else PatchPositionIds(
            num_fut_t, frame_latent_h, frame_latent_w, tubelet_fps, t_offset=num_ctx_t,
        )
        self._register_action_pos(num_ctx_t, tubelet_fps, action_horizon, action_fps)

        # ── Attention mask ───────────────────────────────────────────────
        self.build_flex_mask()

        self._init_weights()


    @staticmethod
    def _dummy_norm_stats(action_dim, state_dim):
        """Placeholder norm stats so ActionPreprocessor buffers exist for state_dict loading."""
        def _make(dim):
            z, o = torch.zeros(dim), torch.ones(dim)
            return {f"q{q}": z.clone() for q in range(1, 6)} | \
                   {f"q{q}": o.clone() for q in range(95, 100)} | \
                   {"mean": z.clone(), "std": o.clone()}
        return {"rel_action": _make(action_dim), "state": _make(state_dim)}

    @classmethod
    def from_checkpoint(cls, path_or_ckpt, **overrides):
        """Rebuild from a saved checkpoint (path or pre-loaded dict). Pass overrides to fix paths etc."""
        if isinstance(path_or_ckpt, dict):
            ckpt = path_or_ckpt
        else:
            ckpt = torch.load(path_or_ckpt, map_location="cpu", weights_only=False)
        if ckpt.get("norm_stats") is None:
            raise ValueError(
                "Checkpoint does not have baked norm stats. "
                "Run bake_norm_stats.py on the checkpoint first."
            )
        config = ckpt["config"] | overrides
        config.pop("stats_path", None)
        config["norm_stats"] = ckpt["norm_stats"]
        config["_pretrained_vlm"] = False
        model = cls(**config)
        model.load_state_dict(ckpt["model"])
        return model

    def load_vjepa2_weights(self, checkpoint_path):
        """Load pretrained VJEPA2 encoder weights from a .pt file."""
        from lewam.models.video_encoder import load_vjepa2_encoder
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
        Returns (per_layer, mask) or (None, None) if no VLM.
        per_layer: list of depth tensors, each (B, S, vlm_hidden_dim)
        mask: (B, S) bool True=ignore
        """
        if self.vlm_encoder is None:
            return None, None
        return self.vlm_encoder(texts, images=images)

    # ── Normalization ─────────────────────────────────────────────────────

    def normalize_state(self, state):
        return self.action_preprocessor.normalize_state(state)

    def normalize_actions(self, actions):
        return self.action_preprocessor.normalize_rel_action(actions)

    def unnormalize_actions(self, actions):
        return self.action_preprocessor.unnormalize_rel_action(actions)

    # ── Forward ───────────────────────────────────────────────────────────

    def _pos_ids(self):
        parts = [self.context_pos.pos]
        if self.future_pos is not None:
            parts.append(self.future_pos.pos)
        if not self._video_only:
            parts.append((self.action_t_ids, self.action_h_ids, self.action_w_ids))
        return concat_pos_ids(*parts)

    def _clear_ode_cache(self):
        self._cached_context_proj = None
        for block in self.blocks:
            if block.ca is not None:
                block.ca._cached_k = None
                block.ca._cached_v = None
                block.ca._cached_mask = None

    def forward(self, x_t_video, x_t_action, context_tokens, t, state,
                lang_tokens=None, lang_mask=None, ode_cache=False):
        """
        x_t_video:      (B, N_fut, vjepa_dim)  noisy future video tokens (ignored if action_only)
        x_t_action:     (B, N_act, action_dim) noisy actions
        context_tokens: (B, N_ctx, vjepa_dim)  clean context video tokens
        t:              (B,)                   flow-matching timestep [0,1]
        state:          (B, state_dim)         normalized state
        lang_tokens:    list[(B, S, vlm_hidden_dim)] | None  per-layer VLM hidden states
        lang_mask:      (B, S) bool True=ignore | None
        ode_cache:      if True, cache/reuse context projection and cross-attention K/V

        Returns (video_vel, action_vel):
            video_vel:  (B, N_fut, vjepa_dim) or None if action_only
            action_vel: (B, N_act, action_dim)
        """
        if self._flex_mask.kv_num_blocks.device != t.device:
            self.build_flex_mask()

        cond = self.t_embedder(t)

        if ode_cache and self._cached_context_proj is not None:
            ctx_proj = self._cached_context_proj
        else:
            ctx_proj = self.context_proj(context_tokens)
            if ode_cache:
                self._cached_context_proj = ctx_proj

        parts = [ctx_proj]
        if self.future_proj is not None:
            parts.append(self.future_proj(x_t_video, cond))
        if not self._video_only:
            parts.append(self.action_encoder(x_t_action, cond))
        x = torch.cat(parts, dim=1)
        state_token = state.unsqueeze(1)

        pos = self._pos_ids()

        mask = self._flex_mask if _USE_FLEX else self._dense_mask
        for i, block in enumerate(self.blocks):
            if lang_tokens is not None:
                sources = [state_token, lang_tokens[i]]
                source_masks = [None, lang_mask]
            else:
                sources = [state_token]
                source_masks = [None]

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, sources, source_masks,
                    cond, mask, pos,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x, sources=sources, source_masks=source_masks,
                    cond=cond, attn_mask=mask, pos=pos,
                    use_kv_cache=ode_cache,
                )

        if self._video_only:
            video_out = x[:, self.N_ctx : self.N_ctx + self.N_fut]
            return self.video_final(video_out, cond), None

        action_out = x[:, self.N_ctx + self.N_fut :]

        if self.action_only:
            return None, self.action_final(action_out, cond)

        video_out = x[:, self.N_ctx : self.N_ctx + self.N_fut]
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
                  smooth=False, cfg_scale=1.0):
        """
        Euler ODE integration from noise -> predicted future.
        Returns (pred_video, pred_actions).

        cfg_scale: classifier-free guidance scale. 1.0 = no guidance,
                   >1.0 amplifies the language-conditioned prediction.
        """
        B, device, dtype = (
            context_tokens.shape[0],
            context_tokens.device,
            context_tokens.dtype,
        )
        x_vid = None if self.action_only else torch.randn(B, self.N_fut, self.VJEPA_DIM, device=device, dtype=dtype)
        x_act = None if self._video_only else torch.randn(B, self.N_act, self.action_dim, device=device, dtype=dtype)

        use_cfg = cfg_scale != 1.0 and lang_tokens is not None

        self._clear_ode_cache()

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device, dtype=dtype)
            v_vid, v_act = self(
                x_vid, x_act, context_tokens, t, state, lang_tokens, lang_mask,
                ode_cache=True,
            )
            if use_cfg:
                v_vid_uncond, v_act_uncond = self(
                    x_vid, x_act, context_tokens, t, state, None, None,
                    ode_cache=True,
                )
                v_vid = v_vid_uncond + cfg_scale * (v_vid - v_vid_uncond) if v_vid is not None else None
                v_act = v_act_uncond + cfg_scale * (v_act - v_act_uncond) if v_act is not None else None
            if v_vid is not None:
                x_vid = x_vid + v_vid * dt
            if v_act is not None:
                x_act = x_act + v_act * dt

        if smooth and x_act is not None:
            x_act = self.smooth_actions(x_act)

        return x_vid, x_act

    # ── Attention mask ────────────────────────────────────────────────────

    @staticmethod
    def _build_flex_mask(N_ctx, N_fut, N_act, spatial, block_size=2, device="cpu", flex_block_size=128):
        C, F, A = N_ctx, N_fut, N_act
        N = C + F + A
        vbs = block_size * spatial
        abs_ = block_size
        num_fut_blocks = max(F // vbs, 1)
        num_act_blocks = max(A // abs_, 1)

        def mask_fn(b, h, q_idx, kv_idx):
            is_ctx = q_idx < C
            is_fut = (q_idx >= C) & (q_idx < C + F)

            ctx_block = q_idx // vbs
            ctx_ok = kv_idx < (ctx_block + 1) * vbs

            fut_block = torch.clamp((q_idx - C) // vbs, min=0)
            act_end = ((fut_block + 1) * num_act_blocks + num_fut_blocks - 1) // num_fut_blocks * abs_
            fut_ok = (
                (kv_idx < C) |
                ((kv_idx >= C) & (kv_idx < C + (fut_block + 1) * vbs)) |
                ((kv_idx >= C + F) & (kv_idx < C + F + act_end))
            )

            act_block = torch.clamp((q_idx - C - F) // abs_, min=0)
            fut_end_block = ((act_block + 1) * num_fut_blocks + num_act_blocks - 1) // num_act_blocks
            act_ok = (
                (kv_idx < C) |
                ((kv_idx >= C) & (kv_idx < C + fut_end_block * vbs)) |
                ((kv_idx >= C + F) & (kv_idx < C + F + (act_block + 1) * abs_))
            )

            return torch.where(is_ctx, ctx_ok, torch.where(is_fut, fut_ok, act_ok))

        return create_block_mask(mask_fn, B=None, H=None, Q_LEN=N, KV_LEN=N, device=device, BLOCK_SIZE=flex_block_size)

    def build_flex_mask(self):
        spatial = self.frame_latent_h * self.frame_latent_w
        device = next(self.parameters()).device
        self._flex_mask = self._build_flex_mask(
            self.N_ctx, self.N_fut, self.N_act, spatial,
            block_size=self.attn_block_size, device=device,
        )
        self._dense_mask = self._build_flex_mask(
            self.N_ctx, self.N_fut, self.N_act, spatial,
            block_size=self.attn_block_size, device="cpu", flex_block_size=1,
        ).to_dense().squeeze().bool().to(device)

    @staticmethod
    def visualize_attn_mask(N_ctx, N_fut, N_act, block_size=2):
        flex = LeWAM._build_flex_mask(N_ctx, N_fut, N_act, spatial=1, block_size=block_size, flex_block_size=1)
        mask = flex.to_dense().squeeze()
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

        for module in (self.context_proj, self.future_proj, self.action_encoder,
                       self.t_embedder, self.blocks, self.video_final,
                       self.action_final):
            if module is not None:
                module.apply(_basic)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        for final in (self.video_final, self.action_final):
            if final is not None:
                nn.init.zeros_(final.adaLN_modulation[-1].weight)
                nn.init.zeros_(final.adaLN_modulation[-1].bias)
                nn.init.zeros_(final.linear.weight)
                nn.init.zeros_(final.linear.bias)

    # ── Runtime config ────────────────────────────────────────────────────

    def set_video_only_mode(self, enabled: bool):
        """Drop action tokens from the transformer sequence entirely.

        When enabled: forward() skips action_encoder/action_final, the flex
        mask is rebuilt with N_act=0, and action_encoder/action_final params
        are frozen so weight decay does not drift them. Reversible.
        """
        if self.action_only and enabled:
            raise ValueError("Cannot enable video_only on an action_only model.")
        self._video_only = bool(enabled)
        self.N_act = 0 if enabled else self.action_horizon
        for p in self.action_encoder.parameters():
            p.requires_grad = not enabled
        for p in self.action_final.parameters():
            p.requires_grad = not enabled
        self.build_flex_mask()

    def set_action_fps(self, action_fps):
        action_horizon = int((self.num_future_frames / self.fps) * action_fps)
        assert action_horizon % 2 == 0, f"Action horizon ({action_horizon}) must be even"
        self.action_fps = action_fps
        self.action_horizon = action_horizon
        self.N_act = 0 if self._video_only else action_horizon
        tubelet_fps = self.fps / self.VJEPA_TUBELET_SIZE
        self._register_action_pos(
            self.num_context_tubelets, tubelet_fps, action_horizon, action_fps,
        )
        self.build_flex_mask()

    def set_fps(self, fps):
        self.fps = fps
        tubelet_fps = fps / self.VJEPA_TUBELET_SIZE
        self.context_pos.set_fps(tubelet_fps)
        if self.future_pos is not None:
            self.future_pos.set_fps(tubelet_fps)
        self._register_action_pos(
            self.num_context_tubelets, tubelet_fps, self.action_horizon, self.action_fps,
        )

    def set_patch_grid(self, H, W, num_cameras=None):
        self.frame_latent_h, self.frame_latent_w = H, W
        self.context_pos.set_patch_grid(H, W)
        if self.future_pos is not None:
            self.future_pos.set_patch_grid(H, W)
        if num_cameras is not None and self.interpolate_rope:
            for block in self.blocks:
                block.rope.set_interpolation(H, W, num_cameras)
        spatial = H * W
        self.N_ctx = self.num_context_tubelets * spatial
        self.N_fut = 0 if self.action_only else self.num_future_tubelets * spatial
        self.build_flex_mask()

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
