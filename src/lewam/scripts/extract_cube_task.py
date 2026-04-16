"""Extract all episodes whose task mentions a colorful cube from
ehalicki/so101_multitask into a lerobot dataset and optionally push it
to the HuggingFace Hub.

Videos are encoded as h264 at 256x256 during extraction.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torchvision.transforms.functional as TF

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

SOURCE_REPO_ID = "ehalicki/so101_multitask"
CUBE_PATTERN = re.compile(r"colorful\s+cube", re.IGNORECASE)
DEFAULT_TARGET_REPO_ID = "ehalicki/so101_cube"

IMAGE_KEYS = ("observation.images.camera2", "observation.images.camera3")
STATE_KEYS = ("action", "observation.state")
SCALE = 256


def _copy_features(src: LeRobotDataset) -> dict:
    keep = set(IMAGE_KEYS) | set(STATE_KEYS)
    features = {}
    for k, v in src.features.items():
        if k not in keep:
            continue
        v = dict(v)
        if k in IMAGE_KEYS:
            shape = v.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) == 3:
                v["shape"] = [SCALE, SCALE, shape[2]]
            info = v.get("info", {})
            if "video.height" in info:
                info["video.height"] = SCALE
            if "video.width" in info:
                info["video.width"] = SCALE
            if "video.codec" in info:
                info["video.codec"] = "h264"
            v["info"] = info
        features[k] = v
    return features


def _matching_episode_indices(src: LeRobotDataset) -> list[int]:
    matches = []
    episodes = src.meta.episodes
    iterator = episodes.items() if isinstance(episodes, dict) else enumerate(episodes)
    for idx, ep in iterator:
        tasks = ep["tasks"] if isinstance(ep, dict) else ep.tasks
        if any(CUBE_PATTERN.search(t) for t in tasks):
            matches.append(int(idx))
    return sorted(matches)


def extract(target_repo_id: str, target_root: Path | None, overwrite: bool,
            push_to_hub: bool) -> Path:
    src = LeRobotDataset(SOURCE_REPO_ID)

    ep_indices = _matching_episode_indices(src)
    if not ep_indices:
        raise RuntimeError("No episodes match 'colorful cube' pattern")
    total_frames = 0
    for i in ep_indices:
        ep = src.meta.episodes[i]
        total_frames += int(ep["length"])
    print(f"Found {len(ep_indices)} episodes, {total_frames} frames total")

    effective_root = target_root if target_root is not None else HF_LEROBOT_HOME / target_repo_id
    if effective_root.exists() and overwrite:
        print(f"Removing existing {effective_root}")
        shutil.rmtree(effective_root)

    if effective_root.exists():
        print(f"Resuming existing dataset at {effective_root}")
        dst = LeRobotDataset.resume(
            repo_id=target_repo_id,
            root=target_root,
            image_writer_threads=4,
            vcodec="h264",
        )
        saved = dst.meta.total_episodes
        print(f"  {saved} episodes already saved, skipping those")
    else:
        dst = LeRobotDataset.create(
            repo_id=target_repo_id,
            fps=src.fps,
            features=_copy_features(src),
            root=target_root,
            robot_type=src.meta.robot_type,
            use_videos=True,
            image_writer_threads=4,
            vcodec="h264",
        )
        saved = 0

    for ep_idx in ep_indices[saved:]:
        ep_meta = src.meta.episodes[ep_idx]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])
        length = int(ep_meta["length"])
        ep_tasks = ep_meta["tasks"] if isinstance(ep_meta, dict) else ep_meta.tasks
        ep_task = next(t for t in ep_tasks if CUBE_PATTERN.search(t))
        print(f"Episode {ep_idx}: frames [{from_idx}, {to_idx}) len={length} task={ep_task!r}")

        try:
            for i in range(from_idx, to_idx):
                sample = src[i]
                frame = {k: sample[k] for k in STATE_KEYS}
                for key in IMAGE_KEYS:
                    img = sample[key]
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = TF.resize(img, [SCALE, SCALE], antialias=True)
                        img = img.permute(1, 2, 0).contiguous()
                    elif img.ndim == 3 and img.shape[2] == 3:
                        img = img.permute(2, 0, 1)
                        img = TF.resize(img, [SCALE, SCALE], antialias=True)
                        img = img.permute(1, 2, 0).contiguous()
                    frame[key] = img
                frame["task"] = ep_task
                dst.add_frame(frame)
                if (i - from_idx) % 100 == 0:
                    print(f"  added frame {i - from_idx + 1}/{length}")
            dst.save_episode()
            saved += 1
        except Exception as e:
            print(f"  SKIPPED episode {ep_idx}: {e}")
            dst.clear_episode_buffer()

    print(f"Saved {saved}/{len(ep_indices)} episodes to {dst.root}")

    info_path = dst.root / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        for val in info.get("features", {}).values():
            if not isinstance(val, dict) or val.get("dtype") != "video":
                continue
            vinfo = val.get("info", {})
            vinfo["video.codec"] = "h264"
            vinfo["video.height"] = SCALE
            vinfo["video.width"] = SCALE
            shape = val.get("shape")
            if isinstance(shape, list) and len(shape) == 3:
                val["shape"] = [SCALE, SCALE, shape[2]]
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

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
