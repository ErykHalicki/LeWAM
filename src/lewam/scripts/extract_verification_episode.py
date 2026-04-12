"""Extract a single episode from ehalicki/so101_multitask into a small
verification dataset used to sanity-check the training loop.

Source episode 55, task "Place the colorful cube in the clear bin" (375 frames).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SOURCE_REPO_ID = "ehalicki/so101_multitask"
SOURCE_EPISODE_INDEX = 55
DEFAULT_TARGET_REPO_ID = "ehalicki/so101_overfit_test"

IMAGE_KEYS = ("observation.images.camera2", "observation.images.camera3")
STATE_KEYS = ("action", "observation.state")


def _copy_features(src: LeRobotDataset) -> dict:
    keep = set(IMAGE_KEYS) | set(STATE_KEYS)
    return {k: v for k, v in src.features.items() if k in keep}


def extract(target_repo_id: str, target_root: Path | None, overwrite: bool) -> Path:
    src = LeRobotDataset(SOURCE_REPO_ID)

    ep_meta = src.meta.episodes[SOURCE_EPISODE_INDEX]
    from_idx = int(ep_meta["dataset_from_index"])
    to_idx = int(ep_meta["dataset_to_index"])
    task_name = ep_meta["tasks"][0]
    length = int(ep_meta["length"])

    print(f"Source ep {SOURCE_EPISODE_INDEX}: frames [{from_idx}, {to_idx}) "
          f"len={length} task={task_name!r}")

    if target_root is not None and target_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target_root} already exists. Pass --overwrite to replace it."
            )
        print(f"Removing existing {target_root}")
        shutil.rmtree(target_root)

    dst = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=src.fps,
        features=_copy_features(src),
        root=target_root,
        robot_type=src.meta.robot_type,
        use_videos=True,
    )

    for i in range(from_idx, to_idx):
        sample = src[i]
        frame = {k: sample[k] for k in STATE_KEYS}
        for key in IMAGE_KEYS:
            img = sample[key]
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0).contiguous()
            frame[key] = img
        frame["task"] = task_name
        dst.add_frame(frame)
        if (i - from_idx) % 50 == 0:
            print(f"  added frame {i - from_idx + 1}/{length}")

    dst.save_episode()
    print(f"Saved episode to {dst.root}")
    return dst.root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo-id", default=DEFAULT_TARGET_REPO_ID)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Local dataset root. Defaults to the HF cache location.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract(args.target_repo_id, args.target_root, args.overwrite)


if __name__ == "__main__":
    main()
