import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("src/wam/scripts/tests/test.avif").convert("RGB")
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
