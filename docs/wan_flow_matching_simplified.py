"""
Simplified reference implementation of WANPolicyHead.
Source: dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py

Omits: KV caching, LoRA, TensorRT, VRAM management, distributed inference,
        timing instrumentation, debug prints, and all loading logic.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from einops import rearrange


# ---------------------------------------------------------------------------
# Submodule stubs (stand-ins for the real implementations)
# ---------------------------------------------------------------------------

class T5TextEncoder(nn.Module):
    """UMT5-XXL text encoder. Produces per-token embeddings."""
    def __init__(self, d_model: int = 4096):
        super().__init__()
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        return torch.zeros(B, L, self.d_model)


class CLIPImageEncoder(nn.Module):
    """Open-CLIP ViT-H/14 image encoder. Returns global CLIP features."""
    def __init__(self, d_model: int = 1280):
        super().__init__()
        self.d_model = d_model

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        return torch.zeros(B, self.d_model)


class VideoVAE(nn.Module):
    """
    Causal 3-D VAE. Spatial 8x downsampling, temporal 4x compression.
    Input:  [B, C=3, T, H, W]  (pixel space, normalised to [-1, 1])
    Output: [B, z_dim, T', H', W']  where T' = 1 + (T-1)/4, H' = H/8, W' = H/8
    """
    def __init__(self, z_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim

    def encode(self, video: torch.Tensor, **kwargs) -> torch.Tensor:
        B, _, T, H, W = video.shape
        T_lat = 1 + (T - 1) // 4
        return torch.zeros(B, self.z_dim, T_lat, H // 8, W // 8)

    def decode(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        B, _, T_lat, H_lat, W_lat = latents.shape
        T = 1 + (T_lat - 1) * 4
        return torch.zeros(B, 3, T, H_lat * 8, W_lat * 8)


class StateEncoder(nn.Module):
    """Linear projection from proprioceptive state to model dim."""
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(state_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.proj(state)


class ActionEncoder(nn.Module):
    """Linear projection from noisy action tokens to model dim."""
    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.proj(action)


class ActionDecoder(nn.Module):
    """Linear projection from model dim back to action space."""
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class WanDiT(nn.Module):
    """
    WAN 2.1/2.2 video diffusion transformer adapted for joint video+action denoising.

    Core additions over the base video DiT:
      - state_encoder:   maps robot state tokens into the sequence
      - action_encoder:  maps noisy action tokens into the sequence
      - action_decoder:  reads out action predictions from the sequence

    The transformer attends jointly over:
      - video latent tokens  (patch-embedded, RoPE-positioned)
      - CLIP image context   (cross-attention)
      - T5 text context      (cross-attention)
      - robot state tokens
      - action tokens

    Returns (video_velocity_pred, action_velocity_pred) — the flow-matching
    velocity field (x1 - x0) for video and actions respectively.
    """
    def __init__(
        self,
        dim: int = 5120,
        num_layers: int = 40,
        num_heads: int = 40,
        action_dim: int = 7,
        state_dim: int = 9,
        num_action_per_block: int = 1,
        num_state_per_block: int = 1,
        frame_seqlen: int = 200,
        local_attn_size: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.frame_seqlen = frame_seqlen
        self.local_attn_size = local_attn_size

        self.state_encoder = StateEncoder(state_dim, dim)
        self.action_encoder = ActionEncoder(action_dim, dim)
        self.action_decoder = ActionDecoder(dim, action_dim)

        # Patch embedding: stride (1, 2, 2) over (T, H, W) latent
        z_dim = 16
        self.patch_embed = nn.Linear(z_dim * 1 * 2 * 2, dim)

        # Timestep embedding (sinusoidal -> MLP)
        self.time_embed = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

        # Transformer blocks (simplified; real WAN uses RoPE + SwiGLU + cross-attn)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, num_heads, batch_first=True) for _ in range(num_layers)])

        # Output projection for video tokens
        self.video_out = nn.Linear(dim, z_dim * 1 * 2 * 2)

    def forward(
        self,
        noisy_video: torch.Tensor,        # [B, z_dim, T_lat, H_lat, W_lat]
        timestep: torch.Tensor,            # [B, T_lat] integer timestep ids
        clip_feature: torch.Tensor,        # [B, d_clip]   CLIP image features
        y: torch.Tensor,                   # [B, z_dim+4, T_lat, H_lat, W_lat] VAE image condition + mask
        context: torch.Tensor,             # [B, L, d_t5]  T5 text embeddings
        seq_len: int,
        state: torch.Tensor | None = None, # [B, n_state, state_dim]
        action: torch.Tensor | None = None,# [B, n_action, action_dim]  noisy actions
        timestep_action: torch.Tensor | None = None,  # [B, n_action]
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        B, C, T, H, W = noisy_video.shape

        # Patch-embed video latents: [B, T*h*w, dim]
        x = rearrange(noisy_video, "b c t (h p1) (w p2) -> b (t h w) (c p1 p2)", p1=2, p2=2)
        x = self.patch_embed(x)

        # Build sequence: video tokens + optional state tokens + optional action tokens
        extra_tokens = []
        if state is not None:
            extra_tokens.append(self.state_encoder(state))
        if action is not None:
            extra_tokens.append(self.action_encoder(action))

        if extra_tokens:
            x = torch.cat([x, *extra_tokens], dim=1)

        # Transformer forward (real impl uses RoPE, SwiGLU, per-block cross-attn)
        for block in self.blocks:
            x = block(x)

        n_video = T * (H // 2) * (W // 2)
        video_tokens = x[:, :n_video]
        video_pred = self.video_out(video_tokens)
        video_pred = rearrange(video_pred, "b (t h w) (c p1 p2) -> b c t (h p1) (w p2)", t=T, h=H//2, w=W//2, p1=2, p2=2)

        action_pred = None
        if action is not None:
            n_state = state.shape[1] if state is not None else 0
            action_tokens = x[:, n_video + n_state:]
            action_pred = self.action_decoder(action_tokens)

        return video_pred, action_pred


# ---------------------------------------------------------------------------
# Flow-matching scheduler stub
# ---------------------------------------------------------------------------

class FlowMatchScheduler:
    """
    Linear flow matching: noisy = (1-t)*x0 + t*x1, velocity target = x1 - x0.
    Timestep 0 = clean, timestep 1000 = pure noise.
    """
    num_train_timesteps: int = 1000

    def __init__(self, shift: float = 5.0):
        self.shift = shift
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1)

    def set_timesteps(self, n: int, training: bool = False):
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, n).long()

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha = t.float() / self.num_train_timesteps
        alpha = alpha.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - alpha) * x0 + alpha * noise

    def training_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return noise - x0

    def training_weight(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t.float())

    def step(self, model_output, timestep, sample, step_index, **kwargs):
        dt = 1.0 / self.num_train_timesteps
        denoised = sample - model_output * dt
        return (denoised,)


# ---------------------------------------------------------------------------
# Main policy head
# ---------------------------------------------------------------------------

@dataclass
class WANPolicyHeadConfig:
    action_dim: int = 7
    action_horizon: int = 16
    state_dim: int = 9
    hidden_size: int = 5120
    num_frames: int = 17
    num_frame_per_block: int = 4
    num_inference_steps: int = 16
    cfg_scale: float = 5.0
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999


class WANPolicyHead(nn.Module):
    """
    Joint video + action generation policy using WAN flow-matching DiT.

    Encoders (all frozen during training unless specified):
      text_encoder  : T5 UMT5-XXL -> per-token embeddings for language conditioning
      image_encoder : CLIP ViT-H/14 -> global clip_feature fed to DiT cross-attention
      vae           : causal 3-D VAE -> video latent space

    Trainable core:
      model         : WAN DiT with appended state_encoder, action_encoder, action_decoder

    Training objective (flow matching):
      Sample t ~ Beta(alpha, beta) for both video and action.
      Add noise: noisy = (1-t)*x0 + t*eps.
      Predict velocity: v = eps - x0.
      Loss = MSE(v_pred, v_target), weighted by scheduler weight.

    Inference:
      Encode first frame with CLIP and VAE.
      Initialise noisy video + action with Gaussian noise.
      Run classifier-free guidance denoising loop (16 steps default).
      Return denoised action sequence and generated video latent.
    """

    def __init__(self, config: WANPolicyHeadConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        self.text_encoder = T5TextEncoder()
        self.image_encoder = CLIPImageEncoder()
        self.vae = VideoVAE()

        self.model = WanDiT(
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            num_action_per_block=config.num_frame_per_block,
        )

        self.scheduler = FlowMatchScheduler(shift=5.0)
        self.scheduler.set_timesteps(1000, training=True)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.normalize_video = lambda v: v * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """T5 encode -> zero-pad past real tokens."""
        seq_lens = attention_mask.gt(0).sum(dim=1)
        emb = self.text_encoder(input_ids, attention_mask)
        for i, v in enumerate(seq_lens):
            emb[i, v:] = 0.0
        return emb

    def encode_image(self, image: torch.Tensor, num_frames: int, H: int, W: int):
        """
        Encode the conditioning (first) frame:
          1. CLIP global feature (clip_feature)
          2. VAE encode: first frame + zero future frames, prepend binary mask (ys)
        Returns (clip_feature, ys, image_latent).
        """
        B = image.shape[0]

        clip_feature = self.image_encoder.encode_image(image)

        image_t = image.transpose(1, 2)
        zeros = torch.zeros(B, 3, num_frames - 1, H, W, dtype=image.dtype, device=image.device)
        y_latent = self.vae.encode(torch.cat([image_t, zeros], dim=2))

        T_lat, H_lat, W_lat = y_latent.shape[2], y_latent.shape[3], y_latent.shape[4]
        mask = torch.zeros(B, 4, T_lat, H_lat, W_lat, dtype=y_latent.dtype, device=y_latent.device)
        mask[:, :, 0:1] = 1.0
        ys = torch.cat([mask, y_latent], dim=1)

        return clip_feature, ys, y_latent[:, :, 0:1]

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(video)

    def sample_time(self, batch_size: int, device, dtype):
        s = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - s) / self.config.noise_s

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """
        batch keys:
          images            [B, T, H, W, C]  uint8 or float
          text              [B, L]  token ids
          text_attention_mask [B, L]
          state             [B, n_state, state_dim]
          action            [B, n_action, action_dim]  in [-1, 1]
          action_mask       [B, n_action, action_dim]  binary
          embodiment_id     [B]
          has_real_action   [B]  bool
        """
        videos = rearrange(batch["images"].float() / 255.0, "b t h w c -> b c t h w")
        videos = self.normalize_video(videos)

        state = batch["state"]
        actions = batch["action"]
        action_mask = batch["action_mask"]
        has_real_action = batch["has_real_action"]

        B = videos.shape[0]
        device = videos.device

        # --- Encode conditioning ---
        prompt_embs = self.encode_prompt(batch["text"], batch["text_attention_mask"])

        latents = self.encode_video(videos)

        _, _, num_frames, H, W = videos.shape
        first_frame = videos[:, :, :1].transpose(1, 2)
        clip_feas, ys, _ = self.encode_image(first_frame, num_frames, H, W)

        # --- Sample noise and timesteps ---
        noise = torch.randn_like(latents)
        noise_action = torch.randn_like(actions)

        # Per-frame timestep ids, shape [B, T_lat]
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (B, latents.shape[2]), device=device)
        timestep_action_id = torch.randint(0, self.scheduler.num_train_timesteps, actions.shape[:2], device=device)

        timestep = self.scheduler.timesteps[timestep_id].to(device)
        timestep_action = self.scheduler.timesteps[timestep_action_id].to(device)

        noisy_latents = self.scheduler.add_noise(
            latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (B, latents.shape[2]))

        noisy_actions = self.scheduler.add_noise(
            actions.flatten(0, 1), noise_action.flatten(0, 1), timestep_action.flatten(0, 1)
        ).unflatten(0, (B, actions.shape[1]))

        training_target_video = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)
        training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)

        T_lat, H_lat, W_lat = latents.shape[2], latents.shape[3], latents.shape[4]
        seq_len = T_lat * (H_lat // 2) * (W_lat // 2)

        # --- DiT forward ---
        video_pred, action_pred = self.model(
            noisy_video=noisy_latents,
            timestep=timestep,
            clip_feature=clip_feas,
            y=ys,
            context=prompt_embs,
            seq_len=seq_len,
            state=state,
            action=noisy_actions,
            timestep_action=timestep_action,
        )

        # --- Losses ---
        dynamics_loss = F.mse_loss(video_pred.float(), training_target_video.float())

        action_loss_raw = F.mse_loss(action_pred.float(), training_target_action.float(), reduction="none")
        action_loss_raw = action_loss_raw * action_mask
        action_loss_raw = has_real_action[:, None, None].float() * action_loss_raw
        action_loss = action_loss_raw.mean()

        loss = dynamics_loss + action_loss

        return {"loss": loss, "dynamics_loss": dynamics_loss, "action_loss": action_loss}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, batch: dict) -> dict:
        """
        Classifier-free guidance denoising loop.
        Returns {"action_pred": [B, action_horizon, action_dim],
                 "video_pred":  [B, C, T, H, W]}.
        """
        videos = rearrange(batch["images"].float() / 255.0, "b t h w c -> b c t h w")
        videos = self.normalize_video(videos)
        state = batch["state"]
        B, _, _, H, W = videos.shape
        device = videos.device

        # Encode text (positive + negative for CFG)
        prompt_emb = self.encode_prompt(batch["text"], batch["text_attention_mask"])
        prompt_emb_neg = self.encode_prompt(batch["text_negative"], batch["text_attention_mask_negative"])

        _, num_frames_vid, H_vid, W_vid = videos.shape[1], videos.shape[2], videos.shape[3], videos.shape[4]
        first_frame = videos[:, :, :1].transpose(1, 2)
        clip_feas, ys, image_latent = self.encode_image(first_frame, num_frames_vid, H_vid, W_vid)

        latents = self.encode_video(videos)
        T_lat, H_lat, W_lat = latents.shape[2], latents.shape[3], latents.shape[4]
        seq_len = T_lat * (H_lat // 2) * (W_lat // 2)

        # Initialise with pure noise
        noisy_video = torch.randn_like(latents)
        noisy_action = torch.randn(B, self.action_horizon, self.action_dim, device=device, dtype=latents.dtype)

        self.scheduler.set_timesteps(self.config.num_inference_steps)

        for t in self.scheduler.timesteps:
            t_val = int(t.item())
            timestep = torch.full((B, T_lat), t_val, device=device, dtype=torch.long)
            timestep_action = torch.full((B, self.action_horizon), t_val, device=device, dtype=torch.long)

            # Conditioned prediction
            pred_cond, pred_action_cond = self.model(
                noisy_video=noisy_video, timestep=timestep,
                clip_feature=clip_feas, y=ys, context=prompt_emb, seq_len=seq_len,
                state=state, action=noisy_action, timestep_action=timestep_action,
            )

            # Unconditioned prediction (CFG)
            pred_uncond, _ = self.model(
                noisy_video=noisy_video, timestep=timestep,
                clip_feature=clip_feas, y=ys, context=prompt_emb_neg, seq_len=seq_len,
                state=state, action=noisy_action, timestep_action=timestep_action,
            )

            # Classifier-free guidance
            video_pred = pred_uncond + self.config.cfg_scale * (pred_cond - pred_uncond)

            noisy_video = self.scheduler.step(video_pred, t, noisy_video, step_index=0)[0]
            noisy_action = self.scheduler.step(pred_action_cond, t, noisy_action, step_index=0)[0]

        return {"action_pred": noisy_action, "video_pred": noisy_video}
