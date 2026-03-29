"""
Precompute normalization stats for relative actions and proprioceptive state.

Reads action/state columns directly from parquet files — no DataLoader overhead.

Saves to $LE_WAM_ROOT/.cache/<repo_id>/norm_stats.pt as:
    {
        'rel_action': {'p2': Tensor[6], 'p98': Tensor[6], 'mean': Tensor[6], 'std': Tensor[6]},
        'state':      {'p2': Tensor[6], 'p98': Tensor[6], 'mean': Tensor[6], 'std': Tensor[6]},
    }

Run:
    source .venv/bin/activate
    python src/wam/scripts/precompute_norm_stats.py
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wam.datasets.community_dataset import CommunityDataset


def precompute_norm_stats(
    repo_id: str,
    cache_root: str | Path,
    tubelet_size: int = 2,
    max_samples: int | None = None,
) -> Path:
    cache_root = Path(cache_root)
    out_path   = cache_root / repo_id / "norm_stats.pt"
    root       = cache_root / repo_id

    print(f"Loading metadata for {repo_id}...")
    cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)
    cd.prefetch_metadata()

    all_rel_actions: list[np.ndarray] = []
    all_states:      list[np.ndarray] = []
    total = 0

    for i, (subpath, meta) in enumerate(cd.metas.items()):
        parquet_dir = root / subpath / "data"
        parquet_files = sorted(parquet_dir.glob("**/*.parquet"))
        if not parquet_files:
            continue

        dfs = [
            pd.read_parquet(p, columns=["action", "observation.state", "episode_index", "frame_index"])
            for p in parquet_files
        ]
        df = pd.concat(dfs, ignore_index=True).sort_values(["episode_index", "frame_index"])

        actions = np.stack(df["action"].values).astype(np.float32)           # (N, 6)
        states  = np.stack(df["observation.state"].values).astype(np.float32) # (N, 6)
        ep_idx  = df["episode_index"].values

        mask        = ep_idx[tubelet_size:] == ep_idx[:-tubelet_size]
        rel_actions = (actions[tubelet_size:] - actions[:-tubelet_size])[mask]

        all_rel_actions.append(rel_actions)
        all_states.append(states)
        total += len(df)
        print(f"  [{i+1}/{len(cd.metas)}] {subpath}: {len(df)} frames  total={total}", end="\r")

        if max_samples is not None and total >= max_samples:
            break

    print(f"\nCollected {total} frames from {len(cd.metas)} datasets.")

    rel_action_data = torch.from_numpy(np.concatenate(all_rel_actions, axis=0))
    state_data      = torch.from_numpy(np.concatenate(all_states,      axis=0))

    def compute_stats(data: torch.Tensor) -> dict:
        p5      = torch.quantile(data, 0.05, dim=0)
        p95     = torch.quantile(data, 0.95, dim=0)
        clipped = torch.maximum(torch.minimum(data, p95), p5)
        mean    = clipped.mean(dim=0)
        std     = clipped.std(dim=0).clamp(min=1e-6)
        return {"p5": p5, "p95": p95, "mean": mean, "std": std}

    stats = {
        "rel_action": compute_stats(rel_action_data),
        "state":      compute_stats(state_data),
    }

    torch.save(stats, out_path)
    print(f"Saved norm stats to {out_path}")
    for key, s in stats.items():
        print(f"  {key}  p5={[f'{v:.4f}' for v in s['p5'].tolist()]}  p95={[f'{v:.4f}' for v in s['p95'].tolist()]}  mean={[f'{v:.4f}' for v in s['mean'].tolist()]}  std={[f'{v:.4f}' for v in s['std'].tolist()]}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id",       type=str, default="ehalicki/LeWAM_community_dataset")
    parser.add_argument("--max-samples",   type=int, default=None)
    parser.add_argument("--tubelet-size",  type=int, default=2)
    parser.add_argument("--sanity-check",  action="store_true")
    args = parser.parse_args()

    le_wam_root = os.environ.get("LE_WAM_ROOT")
    if not le_wam_root:
        raise ValueError("LE_WAM_ROOT environment variable not set")

    repo_id = args.repo_id + ("_small" if args.sanity_check else "")
    precompute_norm_stats(
        repo_id=repo_id,
        cache_root=Path(le_wam_root) / ".cache",
        tubelet_size=args.tubelet_size,
        max_samples=args.max_samples,
    )
