"""
Precompute Gemma language embeddings for all unique task strings in LeWAM_community_dataset.

Saves to $LE_WAM_ROOT/.cache/<repo_id>/task_embeddings.pt as:
    { task_str: (embeddings: Tensor[S, D], mask: Tensor[S]) }

Run:
    source .venv/bin/activate
    python src/wam/scripts/precompute_task_embeddings.py
"""

import argparse
import os
from pathlib import Path

import torch

from wam.datasets.community_dataset import CommunityDataset
from wam.models.encoders import load_t5gemma_encoder


def precompute_task_embeddings(
    repo_id: str,
    cache_root: str | Path,
    batch_size: int = 64,
    device_map: str = "auto",
) -> Path:
    cache_root = Path(cache_root)
    out_path = cache_root / repo_id / "task_embeddings.pt"

    print(f"Loading metadata for {repo_id}...")
    cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)
    cd.prefetch_metadata()

    all_tasks: set[str] = set()
    for meta in cd.metas.values():
        if meta.tasks is not None:
            all_tasks.update(meta.tasks.index.tolist())

    tasks = sorted(all_tasks)
    print(f"Found {len(tasks)} unique task strings.")

    print(f"Loading Gemma encoder (device_map={device_map})...")
    encoder = load_t5gemma_encoder(device_map=device_map)

    cache = {}
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        with torch.no_grad():
            embs, masks = encoder(batch)
        for task, emb, mask in zip(batch, embs, masks):
            cache[task] = (emb.cpu(), mask.cpu())
        print(f"  encoded {min(i + batch_size, len(tasks))}/{len(tasks)}")

    torch.save(cache, out_path)
    print(f"Saved {len(cache)} embeddings to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="ehalicki/LeWAM_community_dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--sanity-check", action="store_true", help="Use _small dataset")
    args = parser.parse_args()

    le_wam_root = os.environ.get("LE_WAM_ROOT")
    if not le_wam_root:
        raise ValueError("LE_WAM_ROOT environment variable not set")

    repo_id = args.repo_id + ("_small" if args.sanity_check else "")
    precompute_task_embeddings(
        repo_id=repo_id,
        cache_root=Path(le_wam_root) / ".cache",
        batch_size=args.batch_size,
        device_map=args.device_map,
    )
