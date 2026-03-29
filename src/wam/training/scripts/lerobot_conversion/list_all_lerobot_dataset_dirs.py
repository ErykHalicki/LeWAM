import argparse
from pathlib import Path


def is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Top-level directory (e.g. /cluster/scratch/ehalicki/lerobot_community_dataset)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output if args.output is not None else args.root / "datasets.txt"

    datasets = []
    for author_dir in sorted(args.root.iterdir()):
        if not author_dir.is_dir():
            continue
        for dataset_dir in sorted(author_dir.iterdir()):
            if dataset_dir.is_dir() and is_lerobot_dataset(dataset_dir):
                datasets.append(f"{author_dir.name}/{dataset_dir.name}")

    output.write_text("\n".join(datasets) + "\n")
    print(f"Found {len(datasets)} datasets -> {output}")


if __name__ == "__main__":
    main()
