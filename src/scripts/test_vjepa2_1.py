import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

from vjepa2.evals.hub.preprocessor import vjepa2_preprocessor
from vjepa2.app.vjepa_2_1.models.vision_transformer import vit_base

crop_size = 512
processor = vjepa2_preprocessor(crop_size=crop_size)

model = vit_base(
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

state_dict = torch.load("weights/vjepa2_1_vitb_dist_vitG_384.pt", map_location="cpu")
encoder_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict["ema_encoder"].items()}
model.load_state_dict(encoder_sd, strict=True)
model.eval()

img = Image.open('src/scripts/test.avif').convert('RGB')
frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # C x H x W
raw_frames = frame.unsqueeze(0).repeat(8, 1, 1, 1)  # T x C x H x W

video = processor(raw_frames)[0]  # C x T x H x W
print(video.shape)

with torch.no_grad():
    video_embeddings = model(video.unsqueeze(0))

print(video_embeddings.shape)

tokens = video_embeddings[0].float().numpy()  # (N, D)
N = tokens.shape[0]
H_patches = W_patches = int((crop_size / 16))
T = int(N / (H_patches*W_patches))  

pca = PCA(n_components=3)
components = pca.fit_transform(tokens)

components -= components.min(axis=0)
components /= components.max(axis=0)

frames_pca = components.reshape(T, H_patches, W_patches, 3)

orig = np.array(img)

fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))
for t in range(T):
    axes[0, t].imshow(orig)
    axes[0, t].axis('off')
    axes[0, t].set_title(f't={t}', fontsize=8)

    pca_up = np.array(Image.fromarray((frames_pca[t] * 255).astype(np.uint8)))
    axes[1, t].imshow(pca_up)
    axes[1, t].axis('off')
    axes[1, t].set_title(f'pca t={t}', fontsize=8)

plt.suptitle('V-JEPA2 patch embeddings PCA (3 components → RGB)')
plt.tight_layout()
plt.show()
