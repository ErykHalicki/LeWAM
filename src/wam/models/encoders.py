from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from wam.models.common import make_mlp

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class VideoEncoder(nn.Module, ABC):
    """
    Abstract base class for video encoders.

    Subclasses must implement forward(frames) → (B, T*H*W, D).
    """
    def set_frozen(self, frozen: bool):
        for p in self.backbone.parameters():
            p.requires_grad_(not frozen)

    @abstractmethod
    def forward(self, frames):
        ...


class LanguageEncoder(nn.Module, ABC):
    """
    Abstract base class for language encoders.
    Frozen by default.

    Subclasses must implement forward(texts) → (embeddings, key_padding_mask).
    """
    def set_frozen(self, frozen: bool):
        for p in self.backbone.parameters():
            p.requires_grad_(not frozen)

    @abstractmethod
    def forward(self, texts):
        ...


class VJEPA2VideoPreprocessor(nn.Module):
    """
    Replicates the VJEPA2 eval-mode preprocessing pipeline without the VJEPA2 dependency.

    Pipeline:
        1. Resize short side to crop_size (upscales if needed, minimal crop)
        2. Center-crop to (crop_size, crop_size) to handle non-square frames
        3. Scale uint8 [0, 255] → float [0, 1]
        4. ImageNet normalize
        5. Rearrange (B, T, C, H, W) → (B, C, T, H, W) for VJEPA2

    forward(frames):
        frames: (B, T, C, H, W)  uint8 or float in [0, 255]
        → (B, C, T, H, W)  float, ImageNet-normalized
    """
    def __init__(self, crop_size=224):
        super().__init__()
        self.crop_size = crop_size
        self.register_buffer('mean', torch.tensor(_IMAGENET_MEAN).view(3, 1, 1, 1))
        self.register_buffer('std',  torch.tensor(_IMAGENET_STD).view(3, 1, 1, 1))

    def unnormalize(self, frames):
        """
        Invert ImageNet normalization.

        frames: (..., C, H, W)  ImageNet-normalized float
        → (..., C, H, W)  float in [0, 1]
        """
        mean = self.mean.view(3, 1, 1)
        std  = self.std.view(3, 1, 1)
        return (frames * std + mean).clamp(0, 1)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W).float()
        #x = TF.resize(x, self.crop_size, antialias=True)
        #x = TF.center_crop(x, self.crop_size)
        #x = x / 255.0
        x = x.view(B, T, C, self.crop_size, self.crop_size)
        x = x.permute(0, 2, 1, 3, 4)               # (B, C, T, H, W)
        x = (x - self.mean) / self.std
        return x


class VJEPA2VideoEncoder(VideoEncoder):
    """
    VJEPA2 video encoder.

    preprocess(frames):
        frames: (T, C, H, W) or (B, T, C, H, W)  uint8 or float in [0, 255]
        → (C, T, H, W) or (B, C, T, H, W)  float, ImageNet-normalized

    forward(frames):
        frames: (B, C, T, H, W)  already preprocessed
        → (B, T*H*W, D)
    """
    def __init__(self, backbone, crop_size=224):
        super().__init__()
        self.backbone     = backbone
        self.preprocessor = VJEPA2VideoPreprocessor(crop_size)

    def preprocess(self, frames):
        batched = frames.ndim == 5
        if not batched:
            frames = frames.unsqueeze(0)
        out = self.preprocessor(frames)
        return out if batched else out.squeeze(0)

    def forward(self, frames):
        return self.backbone(frames)


class GemmaLanguageEncoder(LanguageEncoder):
    """
    Gemma language encoder. Tokenization is handled internally. Frozen by default.

    forward(texts):
        texts: list[str]
        → embeddings (B, S, D), key_padding_mask (B, S) where True = ignore
    """
    def __init__(self, backbone, tokenizer):
        super().__init__()
        self.backbone  = backbone
        self.tokenizer = tokenizer
        self.set_frozen(True)

    def forward(self, texts):
        device = next(self.backbone.parameters()).device
        enc    = self.tokenizer(texts, return_tensors="pt", padding=True).to(device)
        out    = self.backbone(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        mask   = ~enc.attention_mask.bool()   # True = ignore
        return out.last_hidden_state, mask


def build_vjepa2_encoder_arch(crop_size: int = 384) -> "VJEPA2VideoEncoder":
    """
    Build a VJEPA2-B encoder with random weights (architecture only).
    Use this when weights will be loaded via load_state_dict afterward.
    """
    from vjepa2.app.vjepa_2_1.models.vision_transformer import vit_base

    backbone = vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=16,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    return VJEPA2VideoEncoder(backbone, crop_size=crop_size)


def load_vjepa2_encoder(checkpoint_path: str, crop_size: int = 384) -> "VJEPA2VideoEncoder":
    """
    Load a VJEPA2-B encoder from a checkpoint file.

    checkpoint_path: path to .pt file (e.g. weights/vjepa2_1_vitb_dist_vitG_384.pt)
    crop_size:       spatial resolution fed to the backbone (384 matches training resolution)
    → VJEPA2VideoEncoder ready for eval (frozen by default)
    """
    from vjepa2.app.vjepa_2_1.models.vision_transformer import vit_base

    backbone = vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=16,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in ckpt["ema_encoder"].items()
    }
    backbone.load_state_dict(sd, strict=True)
    backbone.eval()
    enc = VJEPA2VideoEncoder(backbone, crop_size=crop_size)
    enc.set_frozen(True)
    return enc


def load_t5gemma_encoder(
    model_id: str = "google/t5gemma-s-s-prefixlm",
    path=None,
    torch_dtype=None,
    device_map: str | None = "auto",
) -> "GemmaLanguageEncoder":
    """
    Load a T5Gemma encoder from HuggingFace.

    model_id:    HuggingFace model ID
    torch_dtype: dtype for weights (defaults to bfloat16)
    device_map:  device placement strategy
    → GemmaLanguageEncoder (frozen by default)
    """
    import torch
    from transformers import T5GemmaEncoderModel, AutoTokenizer

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    backbone = T5GemmaEncoderModel.from_pretrained(
        model_id if path is None else path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        is_encoder_decoder=False, # im not sure why this works
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id if path is None else path)
    enc = GemmaLanguageEncoder(backbone, tokenizer)
    enc.eval()
    enc.set_frozen(True)
    return enc


class StateEncoder(nn.Module):
    """
    Shared MLP encoder for proprioceptive state.
    Used by both DiT and IDM so state representation is consistent.

    forward(state):
        state: (B, state_dim)
        → (B, out_dim)
    """
    def __init__(self, state_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = make_mlp(state_dim, hidden_dim, out_dim)

    def forward(self, state):
        return self.mlp(state)


class ActionDecoder(nn.Module):
    """
    Shared MLP decoder: maps IDM action latents → actions.
    Applied identically to each timestep (weights shared across horizon).

    forward(latents):
        latents: (B, chunk_len, latent_dim)
        → (B, chunk_len, action_dim)
    """
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.mlp = make_mlp(latent_dim, hidden_dim, action_dim)

    def forward(self, latents):
        return self.mlp(latents)


class ActionPreprocessor(nn.Module):
    """
    Normalizes and unnormalizes relative actions and state using precomputed per-dim stats.

    Stats are loaded from a norm_stats.pt file produced by precompute_norm_stats.py:
        {
            'rel_action': {'p5', 'p95', 'mean', 'std'},
            'state':      {'p5', 'p95', 'mean', 'std'},
        }

    Normalization:
        1. Clamp to [p5, p95]
        2. Subtract mean, divide by std

    All buffers move with the module (to(), cuda(), etc.).
    """

    rel_action_p5:   torch.Tensor
    rel_action_p95:  torch.Tensor
    rel_action_mean: torch.Tensor
    rel_action_std:  torch.Tensor
    state_p5:        torch.Tensor
    state_p95:       torch.Tensor
    state_mean:      torch.Tensor
    state_std:       torch.Tensor

    def __init__(self, stats_path: str):
        super().__init__()
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        for key in ("rel_action", "state"):
            for stat in ("p5", "p95", "mean", "std"):
                self.register_buffer(f"{key}_{stat}", stats[key][stat].float())

    def normalize_rel_action(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.maximum(torch.minimum(x, self.rel_action_p95), self.rel_action_p5) - self.rel_action_mean) / self.rel_action_std

    def unnormalize_rel_action(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.rel_action_std + self.rel_action_mean

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.maximum(torch.minimum(x, self.state_p95), self.state_p5) - self.state_mean) / self.state_std

    def unnormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.state_std + self.state_mean
