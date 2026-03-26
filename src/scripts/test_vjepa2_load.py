import torch

from vjepa2.app.vjepa_2_1.models.vision_transformer import vit_base

encoder = vit_base(
    patch_size=16,
    img_size=(256, 256),
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
encoder.eval()

B, C, T, H, W = 1, 3, 16, 256, 256
x = torch.randn(B, C, T, H, W)

with torch.no_grad():
    out = encoder(x)

print(f"input:  {list(x.shape)}")
print(f"output: {list(out.shape)}")
