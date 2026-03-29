"""
Interactively review and edit tasks from meta/tasks.parquet for each subdataset.

Run:
    source .venv/bin/activate
    python src/wam/scripts/print_tasks.py --sanity-check
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from wam.datasets.community_dataset import CommunityDataset


def review_tasks(repo_id: str, cache_root: str | Path, print_only=False) -> None:
    cache_root = Path(cache_root)
    root = cache_root / repo_id

    cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)

    for subpath in cd.subpaths:
        tasks_path = root / subpath / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            print(f"{subpath}: no tasks.parquet")
            continue

        df = pd.read_parquet(tasks_path)
        tasks = df.index.tolist() if df.index.name == "task" else df["task"].tolist()

        print(f"\n=== {subpath} ===")
        updated = False
        new_tasks = []
        for task in tasks:
            print(f"  Task: {task}")
            if not print_only:
                answer = input("  Good? [y/n]: ").strip().lower()
                if answer == "y":
                    new_tasks.append(task)
                else:
                    new_desc = input("  New description: ").strip()
                    new_tasks.append(new_desc)
                    updated = True

        if updated:
            new_index = pd.Index(new_tasks, name="task")
            new_df = pd.DataFrame({"task_index": range(len(new_tasks))}, index=new_index)
            new_df.to_parquet(tasks_path)
            print(f"  Saved {tasks_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="ehalicki/LeWAM_community_dataset")
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--print-only", action='store_true')
    args = parser.parse_args()

    le_wam_root = os.environ.get("LE_WAM_ROOT")
    if not le_wam_root:
        raise ValueError("LE_WAM_ROOT environment variable not set")

    repo_id = args.repo_id + ("_small" if args.sanity_check else "")
    review_tasks(
        repo_id=repo_id,
        cache_root=Path(le_wam_root) / ".cache" if args.cache_root is None else args.cache_root,
        print_only=args.print_only
    )
