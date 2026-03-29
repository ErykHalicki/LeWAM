"""
Loads 4 shuffled samples from the community dataset with delta_timestamps and visualizes them.

Run:
    source .venv/bin/activate
    python src/wam/scripts/tests/test_community_dataloader.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import DataLoader
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

REPO_ID    = "ehalicki/LeWAM_community_dataset"
SUBPATH    = "danaaubakirova/so100_task_1"
FPS        = 30.0
NUM_PAST   = 8
NUM_FUTURE = 8
CHUNK_LEN  = 16

past_ts   = [-(NUM_PAST - 1 - i) / FPS for i in range(NUM_PAST)]
future_ts = [(i + 1) / FPS for i in range(NUM_FUTURE)]

delta_timestamps = {
    "observation.images.image":  past_ts + future_ts,
    "observation.images.image2": past_ts + future_ts,
    "observation.state":         [0.0],
    "action":                    [i / FPS for i in range(CHUNK_LEN)],
}

cache_root = Path(os.environ.get("LE_WAM_ROOT", ".")) / ".cache"
local_root = cache_root / REPO_ID / SUBPATH

dataset = StreamingLeRobotDataset(
    repo_id=REPO_ID,
    root=local_root,
    delta_timestamps=delta_timestamps,
    shuffle=True,
    buffer_size=20,
    seed=42,
)

dataset.hf_dataset = dataset.hf_dataset.skip(int(dataset.meta.total_frames * 0.9))

loader = DataLoader(dataset, batch_size=4, num_workers=0)

print("Fetching batch...")
batch = next(iter(loader))
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)}")

# (B, T, 3, H, W)
imgs1   = batch["observation.images.image"].numpy()
imgs2   = batch["observation.images.image2"].numpy()
actions = batch["action"].numpy()  # (B, CHUNK_LEN, 6)
B       = imgs1.shape[0]
T       = NUM_PAST + NUM_FUTURE
all_ts  = past_ts + future_ts
t_axis  = np.arange(CHUNK_LEN) / FPS * 1000

def to_img(arr):  # (3, H, W) -> (H, W, 3)
    return np.clip(arr.transpose(1, 2, 0), 0, 1)

SHOW_FRAMES = 6  # subset of T to show (every other frame)
frame_indices = np.linspace(0, T - 1, SHOW_FRAMES, dtype=int)
action_labels = ["j0", "j1", "j2", "j3", "j4", "gripper"]

fig = plt.figure(figsize=(10, 2.0 * B))
outer = gridspec.GridSpec(B, 1, figure=fig, hspace=0.55)

for i in range(B):
    ncols = SHOW_FRAMES + 3
    inner = gridspec.GridSpecFromSubplotSpec(2, ncols, subplot_spec=outer[i], wspace=0.06, hspace=0.25)

    for col, t in enumerate(frame_indices):
        ax1 = fig.add_subplot(inner[0, col])
        ax2 = fig.add_subplot(inner[1, col])
        ax1.imshow(to_img(imgs1[i, t]))
        ax2.imshow(to_img(imgs2[i, t]))
        ax1.axis("off")
        ax2.axis("off")
        ms = int(all_ts[t] * 1000)
        color = "#cc4444" if t >= NUM_PAST else "black"
        ax1.set_title(f"{ms:+d}ms", fontsize=7, color=color)
        if col == 0:
            ax1.set_ylabel("cam1", fontsize=7)
            ax2.set_ylabel("cam2", fontsize=7)

    ax_act = fig.add_subplot(inner[:, SHOW_FRAMES:])
    for j, label in enumerate(action_labels):
        ax_act.plot(t_axis, actions[i, :, j], label=label)
    ax_act.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_act.set_xlabel("ms from t=0")
    ax_act.set_ylabel("action value")
    ep = batch["episode_index"][i].item()
    fi = batch["frame_index"][i].item()
    ax_act.set_title(f"sample {i}  ep={ep}  frame={fi}", fontsize=9)
    ax_act.legend(fontsize=7, loc="upper right")
    ax_act.grid(True, alpha=0.3)

plt.suptitle(
    f"{SUBPATH}  |  shuffle=True  |  past={NUM_PAST} future={NUM_FUTURE} @ {FPS}Hz  |  chunk={CHUNK_LEN}",
    fontsize=10,
)
plt.show()
