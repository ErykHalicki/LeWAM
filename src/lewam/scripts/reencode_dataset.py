"""
Re-encode all videos in a LeRobot dataset from AV1 to h264 short-GOP in place.

AV1 decode is 3-5x slower than h264 and much worse under random seeks, which
punishes training dataloaders. This script swaps every mp4 in the dataset for
an h264 short-GOP (-g 4) version and updates meta/info.json accordingly.

Usage:
    python -m lewam.scripts.reencode_dataset <repo_id>
    python -m lewam.scripts.reencode_dataset ehalicki/so101_overfit_test
    python -m lewam.scripts.reencode_dataset ehalicki/so101_overfit_test --gop 1 --crf 20
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME


def reencode_one(src: Path, gop: int, crf: int, preset: str) -> None:
    tmp = src.with_suffix(".reencode.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    shutil.move(str(tmp), str(src))


def update_info_json(info_path: Path) -> int:
    if not info_path.exists():
        return 0
    with open(info_path) as f:
        info = json.load(f)
    changed = 0
    features = info.get("features", info)
    for val in features.values():
        if not isinstance(val, dict):
            continue
        if val.get("dtype") != "video":
            continue
        vinfo = val.get("info", {})
        if vinfo.get("video.codec") and vinfo["video.codec"] != "h264":
            vinfo["video.codec"] = "h264"
            changed += 1
    if changed:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
    return changed


def main():
    parser = argparse.ArgumentParser(description="Re-encode LeRobot dataset videos AV1 -> h264 in place")
    parser.add_argument("repo_id", help="LeRobot repo id (e.g. ehalicki/so101_overfit_test)")
    parser.add_argument("--gop", type=int, default=4, help="GOP size (4 = short-GOP, 1 = all-intra)")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF (lower = higher quality)")
    parser.add_argument("--preset", default="fast", help="libx264 preset")
    parser.add_argument("--cache-root", default=None, help="Override cache root (default: HF_LEROBOT_HOME)")
    args = parser.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else HF_LEROBOT_HOME
    repo_dir = cache_root / args.repo_id
    if not repo_dir.exists():
        print(f"ERROR: repo not found at {repo_dir}", file=sys.stderr)
        sys.exit(1)

    mp4s = sorted(repo_dir.rglob("*.mp4"))
    if not mp4s:
        print(f"No .mp4 files found under {repo_dir}")
        sys.exit(0)

    total_before = sum(p.stat().st_size for p in mp4s)
    print(f"Found {len(mp4s)} mp4 files under {repo_dir}")
    print(f"Re-encoding with libx264 -g {args.gop} -crf {args.crf} -preset {args.preset}")
    print(f"Total size before: {total_before / 1e9:.2f} GB")
    print()

    for i, src in enumerate(mp4s, 1):
        rel = src.relative_to(repo_dir)
        before = src.stat().st_size
        try:
            reencode_one(src, args.gop, args.crf, args.preset)
        except subprocess.CalledProcessError as e:
            print(f"  [{i}/{len(mp4s)}] FAILED {rel}: {e}", file=sys.stderr)
            continue
        after = src.stat().st_size
        ratio = after / max(before, 1)
        print(f"  [{i}/{len(mp4s)}] {rel}  {before/1e6:.1f}MB -> {after/1e6:.1f}MB  ({ratio:.2f}x)")

    total_after = sum(p.stat().st_size for p in mp4s)
    print()
    print(f"Total size after:  {total_after / 1e9:.2f} GB  ({total_after / max(total_before, 1):.2f}x)")

    info_updates = 0
    for info_path in repo_dir.rglob("meta/info.json"):
        n = update_info_json(info_path)
        if n:
            info_updates += n
            print(f"Updated {info_path.relative_to(repo_dir)} ({n} video feature(s))")
    if info_updates == 0:
        print("No meta/info.json updates needed.")


if __name__ == "__main__":
    main()
