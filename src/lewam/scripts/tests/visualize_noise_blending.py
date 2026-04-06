import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

le_wam_root = os.environ.get('LE_WAM_ROOT')
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")

test_img_path = os.path.join(le_wam_root, 'src/lelewam/scripts/tests/test.avif')
img = Image.open(test_img_path).convert("RGB")
x0 = np.array(img).astype(np.float32) / 255.0

x1 = np.random.randn(*x0.shape).astype(np.float32)
x1_vis = (x1 - x1.min()) / (x1.max() - x1.min())

timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, axes = plt.subplots(1, len(timesteps), figsize=(4 * len(timesteps), 4))

for ax, t in zip(axes, timesteps):
    x_t = (1 - t) * x0 + t * x1
    x_t_vis = np.clip(x_t, 0, 1)
    ax.imshow(x_t_vis)
    ax.set_title(f"t = {t}")
    ax.axis("off")

plt.tight_layout()
plt.show()
