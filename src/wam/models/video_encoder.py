import torch
import torch.nn as nn
from einops import rearrange
from torchvision.transforms.v2.functional import center_crop

from wam.models.common import make_mlp

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class VJEPA2VideoPreprocessor(nn.Module):
    """
    Preprocesses video frames for VJEPA2.

    Accepts either:
        (B, T, C, H, W)       single camera
        (B, N, T, C, H, W)    multi-camera
        list of N (B, T, C, H, W) tensors

    Each camera is independently center-cropped to crop_size x crop_size,
    then cameras are stitched side-by-side along the width dimension.

    Returns: (B, C, T, crop_size, crop_size * N)  float, ImageNet-normalized
    """
    def __init__(self, crop_size=384):
        super().__init__()
        self.crop_size = crop_size
        self.register_buffer('mean', torch.tensor(_IMAGENET_MEAN).view(3, 1, 1, 1))
        self.register_buffer('std',  torch.tensor(_IMAGENET_STD).view(3, 1, 1, 1))

    def forward(self, frames):
        if isinstance(frames, list):
            frames = torch.stack(frames, dim=1)
        if frames.ndim == 5:
            frames = frames.unsqueeze(1)

        B, N, T, C, H, W = frames.shape
        cropped = center_crop(frames.float(), [self.crop_size, self.crop_size])
        stitched = rearrange(cropped, 'b n t c h w -> b c t h (n w)')
        return (stitched - self.mean) / self.std

    def unnormalize(self, frames):
        """frames: (..., C, H, W) ImageNet-normalized → float in [0, 1]"""
        mean = self.mean.view(3, 1, 1)
        std  = self.std.view(3, 1, 1)
        return (frames * std + mean).clamp(0, 1)


class VJEPA2VideoEncoder(nn.Module):
    """
    Wraps VJEPA2 backbone.

    forward(frames):
        frames: (B, C, T, H, W)  already preprocessed (ImageNet-normalized)
        → (B, T//tubelet_size * H_patches * W_patches, embed_dim)
    """
    def __init__(self, backbone, crop_size=384):
        super().__init__()
        self.backbone     = backbone
        self.preprocessor = VJEPA2VideoPreprocessor(crop_size)
        self.set_frozen(True)

    def set_frozen(self, frozen: bool):
        for p in self.backbone.parameters():
            p.requires_grad_(not frozen)

    def forward(self, frames):
        return self.backbone(frames.to(dtype=next(self.backbone.parameters()).dtype))


def build_vjepa2_encoder_arch(crop_size: int = 384) -> VJEPA2VideoEncoder:
    """Build a VJEPA2-B encoder with random weights (architecture only)."""
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


def load_vjepa2_encoder(checkpoint_path: str, crop_size: int = 384) -> VJEPA2VideoEncoder:
    """Load a VJEPA2-B encoder from a checkpoint file."""
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
