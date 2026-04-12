"""Extract all 'Place the colorful cube in the clear bin' episodes from
ehalicki/so101_multitask into a single-task lerobot dataset and optionally
push it to the HuggingFace Hub.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SOURCE_REPO_ID = "ehalicki/so101_multitask"
TASK_NAME = "Place the colorful cube in the clear bin"
DEFAULT_TARGET_REPO_ID = "ehalicki/so101_cube"

IMAGE_KEYS = ("observation.images.camera2", "observation.images.camera3")
STATE_KEYS = ("action", "observation.state")


def _copy_features(src: LeRobotDataset) -> dict:
    keep = set(IMAGE_KEYS) | set(STATE_KEYS)
    return {k: v for k, v in src.features.items() if k in keep}


def _matching_episode_indices(src: LeRobotDataset) -> list[int]:
    matches = []
    episodes = src.meta.episodes
    iterator = episodes.items() if isinstance(episodes, dict) else enumerate(episodes)
    for idx, ep in iterator:
        tasks = ep["tasks"] if isinstance(ep, dict) else ep.tasks
        if TASK_NAME in tasks:
            matches.append(int(idx))
    return sorted(matches)


def extract(target_repo_id: str, target_root: Path | None, overwrite: bool,
            push_to_hub: bool) -> Path:
    src = LeRobotDataset(SOURCE_REPO_ID, force_cache_sync=True)

    ep_indices = _matching_episode_indices(src)
    if not ep_indices:
        raise RuntimeError(f"No episodes match task {TASK_NAME!r}")
    total_frames = 0
    for i in ep_indices:
        ep = src.meta.episodes[i]
        total_frames += int(ep["length"])
    print(f"Found {len(ep_indices)} episodes, {total_frames} frames total")

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

    for ep_idx in ep_indices:
        ep_meta = src.meta.episodes[ep_idx]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])
        length = int(ep_meta["length"])
        print(f"Episode {ep_idx}: frames [{from_idx}, {to_idx}) len={length}")

        for i in range(from_idx, to_idx):
            sample = src[i]
            frame = {k: sample[k] for k in STATE_KEYS}
            for key in IMAGE_KEYS:
                img = sample[key]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.permute(1, 2, 0).contiguous()
                frame[key] = img
            frame["task"] = TASK_NAME
            dst.add_frame(frame)
            if (i - from_idx) % 100 == 0:
                print(f"  added frame {i - from_idx + 1}/{length}")

        dst.save_episode()

    print(f"Saved {len(ep_indices)} episodes to {dst.root}")

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub as {target_repo_id}...")
        dst.push_to_hub()
        print("Push complete.")

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
    parser.add_argument("--no-push", action="store_true",
                        help="Skip pushing to HuggingFace Hub.")
    args = parser.parse_args()
    extract(args.target_repo_id, args.target_root, args.overwrite,
            push_to_hub=not args.no_push)


if __name__ == "__main__":
    main()
