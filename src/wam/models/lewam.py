import torch.nn as nn

from wam.models.DiT import DiT_models
from wam.models.IDM import IDM_models
from wam.models.encoders import StateEncoder, ActionDecoder, load_vjepa2_encoder, load_t5gemma_encoder


class LeWAM(nn.Module):
    """
    Full LeWAM model.

    Components:
        video_encoder    VJEPA2 backbone → patch embeddings
        language_encoder Frozen T5Gemma  → token embeddings (None if using pre-computed)
        state_encoder    Shared MLP: raw state → state embedding fed to DiT and IDM
        dit              Flow-matching transformer: predicts future patch embedding velocity field
        idm              Inverse dynamics model: past+future → action latents
        action_decoder   Shared MLP: action latents → actions (weights shared across horizon)

    Training paths:
        World model:   predict_future(x_t, t, past_frames, state, ...)  → velocity field (DiT loss)
        Action policy: infer_actions(past_frames, future_frames, state)  → actions (MSE loss)
    """
    def __init__(self, video_encoder, language_encoder, state_encoder, dit, idm, action_decoder):
        super().__init__()
        self.video_encoder    = video_encoder
        self.language_encoder = language_encoder
        self.state_encoder    = state_encoder
        self.dit              = dit
        self.idm              = idm
        self.action_decoder   = action_decoder

    def set_fps(self, fps: float):
        self.dit.set_fps(fps)
        self.idm.set_fps(fps)

    def encode_video(self, video):
        """video: (B, C, T, H, W) → (B, T*H*W, D)"""
        assert self.video_encoder is not None, "No video encoder attached"
        return self.video_encoder(video)

    def encode_language(self, texts):
        """→ embeddings (B, S, D), key_padding_mask (B, S)"""
        assert self.language_encoder is not None, "No language encoder attached"
        return self.language_encoder(texts)

    def predict_future(self, x_t, t, past_frames, state=None, lang=None, aux_frames=None, l_mask=None):
        """
        DiT forward for world model training.

        x_t:         (B, K*H*W, in_dim)   noisy future embeddings
        t:           (B,)                  flow matching timestep in [0, 1]
        past_frames: (B, T*H*W, in_dim)
        state:       (B, raw_state_dim)
        lang:        (B, S, lang_dim)|None pre-computed language embeddings
        aux_frames:  (B, C*T*H*W, in_dim)|None
        l_mask:      (B, S) True=ignore
        → (B, K*H*W, in_dim)  predicted velocity field
        """
        if state is not None:
            state_emb = self.state_encoder(state)
        else:
            state_emb = None
        return self.dit(x_t, t, past_frames, lang, state_emb, aux_frames, l_mask)

    def infer_actions(self, past_frames, future_frames, state, aux_frames=None):
        """
        IDM + ActionDecoder forward for action prediction.

        past_frames:   (B, T*H*W, in_dim)
        future_frames: (B, K*H*W, in_dim)
        state:         (B, raw_state_dim)
        aux_frames:    (B, C*T*H*W, in_dim)|None
        → (B, K, action_dim)
        """
        state_emb = self.state_encoder(state)
        latents   = self.idm(past_frames, future_frames, state_emb, aux_frames)
        return self.action_decoder(latents)

    def forward(self, x_t, t, past_frames, future_frames, state, lang=None, aux_frames=None, l_mask=None):
        """
        Joint forward for combined world model + action policy training.

        → v_pred (B, K*H*W, in_dim), actions (B, K, action_dim)
        """
        v_pred  = self.predict_future(x_t, t, past_frames, state, lang, aux_frames, l_mask)
        actions = self.infer_actions(past_frames, future_frames, state, aux_frames)
        return v_pred, actions


def build_lewam(
    num_past_frames, 
    num_future_frames,
    video_encoder=None,
    language_encoder=None, 
    patch_h=24,
    patch_w=24,
    in_dim=768,
    lang_dim=512,
    raw_state_dim=64,
    state_enc_dim=128,
    action_dim=7,
    action_latent_dim=64,
    fps=30.0,
    dit_size='B',
    idm_size='B',
):
    """
    Build a full LeWAM model.

    num_past_frames: number of frames to condition on
    num_future_frames: number of frames to predict

    video_encoder:    a VideoEncoder subclass instance (e.g. VJEPA2VideoEncoder), or None
    language_encoder: a LanguageEncoder subclass instance (e.g. GemmaLanguageEncoder), or None
                      (pass None when using pre-computed language embeddings)
    dit_size / idm_size: one of 'XL', 'L', 'B', 'S', 'Baby'
    """
    state_encoder  = StateEncoder(raw_state_dim, state_enc_dim * 2, state_enc_dim)
    action_decoder = ActionDecoder(action_latent_dim, action_latent_dim * 2, action_dim)

    dit = DiT_models[f'DiT-{dit_size}'](
        num_frames=num_future_frames,
        num_past_frames=num_past_frames,
        patch_h=patch_h,
        patch_w=patch_w,
        fps=fps,
        in_dim=in_dim,
        lang_dim=lang_dim,
        state_dim=state_enc_dim,
    )

    idm = IDM_models[f'IDM-{idm_size}'](
        num_past_frames=num_past_frames,
        num_future_frames=num_future_frames,
        patch_h=patch_h,
        patch_w=patch_w,
        fps=fps,
        in_dim=in_dim,
        state_dim=state_enc_dim,
        action_latent_dim=action_latent_dim,
    )

    return LeWAM(
        video_encoder=video_encoder,
        language_encoder=language_encoder,
        state_encoder=state_encoder,
        dit=dit,
        idm=idm,
        action_decoder=action_decoder,
    )


def build_lewam_with_encoders(
    vjepa2_checkpoint: str,
    t5gemma_checkpoint: str = None,
    t5gemma_model_id: str = "google/t5gemma-s-s-prefixlm",
    crop_size: int = 384,
    t5gemma_device_map: str = "auto",
    **kwargs,
) -> LeWAM:
    """
    Build a full LeWAM with real VJEPA2 and T5Gemma weights loaded.

    vjepa2_checkpoint: path to VJEPA2 .pt checkpoint
    t5gemma_model_id:  HuggingFace model ID for T5Gemma
    crop_size:         spatial crop size fed to VJEPA2 (384 = training resolution)
    t5gemma_device_map: device placement for T5Gemma (e.g. "auto", "cpu", "cuda")
    **kwargs:          forwarded to build_lewam (dit_size, idm_size, num_past_frames, etc.)

    Both encoders are frozen by default.
    """
    video_enc = load_vjepa2_encoder(vjepa2_checkpoint, crop_size=crop_size)
    lang_enc  = load_t5gemma_encoder(t5gemma_model_id, path=t5gemma_checkpoint, device_map=t5gemma_device_map)
    return build_lewam(video_encoder=video_enc, language_encoder=lang_enc, **kwargs)
