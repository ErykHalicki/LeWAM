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
        1. Resize short side to int(crop_size * 256/224)
        2. Center-crop to (crop_size, crop_size)
        3. Scale uint8 [0, 255] → float [0, 1]
        4. ImageNet normalize
        5. Rearrange (B, T, C, H, W) → (B, C, T, H, W) for VJEPA2

    forward(frames):
        frames: (B, T, C, H, W)  uint8 or float in [0, 255]
        → (B, C, T, H, W)  float, ImageNet-normalized
    """
    def __init__(self, crop_size=224):
        super().__init__()
        self.crop_size  = crop_size
        self.short_side = int(crop_size * 256 / 224)
        self.register_buffer('mean', torch.tensor(_IMAGENET_MEAN).view(3, 1, 1, 1))
        self.register_buffer('std',  torch.tensor(_IMAGENET_STD).view(3, 1, 1, 1))

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W).float()
        x = TF.resize(x, self.short_side, antialias=True)
        x = TF.center_crop(x, self.crop_size)
        x = x / 255.0
        x = x.view(B, T, C, self.crop_size, self.crop_size)
        x = x.permute(0, 2, 1, 3, 4)               # (B, C, T, H, W)
        x = (x - self.mean) / self.std
        return x


class VJEPA2VideoEncoder(VideoEncoder):
    """
    VJEPA2 video encoder. Preprocessing is handled internally.

    forward(frames):
        frames: (B, T, C, H, W)  uint8 or float in [0, 255]
        → (B, T*H*W, D)
    """
    def __init__(self, backbone, crop_size=224):
        super().__init__()
        self.backbone     = backbone
        self.preprocessor = VJEPA2VideoPreprocessor(crop_size)

    def forward(self, frames):
        video = self.preprocessor(frames)   # (B, C, T, H, W)
        return self.backbone(video)


class GemmaLanguageEncoder(LanguageEncoder):
    """
    Gemma language encoder. Tokenization is handled internally. Frozen by default.

    forward(texts):
        texts: list[str]
        → embeddings (B, S, D), key_padding_mask (B, S) where True = ignore
    """
    def __init__(self, backbone, model_id: str):
        super().__init__()
        from transformers import AutoTokenizer
        self.backbone  = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.set_frozen(True)

    def forward(self, texts):
        device = next(self.backbone.parameters()).device
        enc    = self.tokenizer(texts, return_tensors="pt", padding=True).to(device)
        out    = self.backbone(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        mask   = ~enc.attention_mask.bool()   # True = ignore
        return out.last_hidden_state, mask


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
