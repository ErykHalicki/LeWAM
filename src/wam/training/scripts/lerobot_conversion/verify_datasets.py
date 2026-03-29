import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def verify_dataset(root: Path, relative_path: str) -> tuple[bool, str]:
    dataset_dir = root / relative_path
    try:
        ds = LeRobotDataset(repo_id=relative_path, root=dataset_dir, download_videos=False)
        return True, f"OK ({len(ds)} frames, {ds.num_episodes} episodes)"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Top-level dataset directory")
    parser.add_argument("--datasets-file", type=Path, default=None)
    args = parser.parse_args()

    datasets_file = args.datasets_file or args.root / "datasets.txt"
    datasets = datasets_file.read_text().splitlines()

    passed, failed = [], []

    for i, relative_path in enumerate(datasets):
        ok, msg = verify_dataset(args.root, relative_path)
        status = "PASS" if ok else "FAIL"
        print(f"[{i+1}/{len(datasets)}] {status} {relative_path}: {msg}")
        (passed if ok else failed).append(relative_path)

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        print("Failed datasets:")
        for d in failed:
            print(f"  {d}")


if __name__ == "__main__":
    main()
