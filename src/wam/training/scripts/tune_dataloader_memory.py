import sys
import multiprocessing

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)

import os
import time
import argparse
import psutil
from torch.utils.data import DataLoader

from wam.datasets.community_dataset import CommunityDataset
from torchvision.transforms import v2 as transforms

FPS = 30.0
NUM_PAST = 8
NUM_FUTURE = 8
CHUNK_LEN = 4
CROP_SIZE = 224
TARGET_FRACTION = 0.80
SAFETY_FRACTION = 0.90
WARMUP_BATCHES = 100
NUM_WORKERS = 4

past_ts = [-(NUM_PAST - 1 - i) / FPS for i in range(NUM_PAST)]
future_ts = [(i + 1) / FPS for i in range(NUM_FUTURE)]

image_transforms = transforms.Compose([
    transforms.Resize(CROP_SIZE, antialias=True),
    transforms.CenterCrop(CROP_SIZE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_parser = argparse.ArgumentParser()
_parser.add_argument("--max-gb", type=float, default=None, help="Override total RAM (GB). Use on clusters where psutil sees the whole machine.")
_args, _ = _parser.parse_known_args()

total_ram_mb = _args.max_gb * 1024 if _args.max_gb else psutil.virtual_memory().total / (1024 * 1024)
target_mb = total_ram_mb * TARGET_FRACTION
safety_mb = total_ram_mb * SAFETY_FRACTION

_proc = psutil.Process()

def current_mem_mb():
    procs = [_proc] + _proc.children(recursive=True)
    return sum(p.memory_info().rss for p in procs if p.is_running()) / (1024 * 1024)


def run_trial(pool_size, buffer_size, cache_root, metas):
    """Returns (peak_mb, hit_safety) — hit_safety=True means we aborted early."""
    print(f"\n  pool_size={pool_size}  buffer_size={buffer_size}")
    ds = CommunityDataset(
        repo_id="ehalicki/LeWAM_community_dataset",
        cache_root=cache_root,
        pool_size=pool_size,
        batch_size=16,
        buffer_size=buffer_size,
        delta_timestamps={
            "observation.images": past_ts + future_ts,
            "observation.state": [0.0],
            "action": [i / FPS for i in range(CHUNK_LEN)],
        },
        image_transforms=image_transforms,
    )
    ds.subpaths = metas["subpaths"]
    ds.buckets = metas["buckets"]
    ds.metas = metas["metas"]

    loader = DataLoader(ds, batch_size=16, num_workers=NUM_WORKERS, prefetch_factor=2)
    peak = 0.0
    hit_safety = False
    total_batches = 0
    t_start = time.perf_counter()

    for i, _ in enumerate(loader):
        mem = current_mem_mb()
        peak = max(peak, mem)
        pct = 100 * mem / total_ram_mb
        total_batches += 1
        elapsed = time.perf_counter() - t_start
        speed = total_batches / elapsed if elapsed > 0 else 0.0
        print(f"    batch {i+1}/{WARMUP_BATCHES}  mem={mem:.0f}MB ({pct:.1f}%)  {speed:.2f} batches/s", end="\r")

        if mem >= safety_mb:
            print(f"\n  [abort] safety ceiling hit at batch {i+1}: {mem:.0f}MB ({pct:.1f}%)")
            hit_safety = True
            break

        if i + 1 >= WARMUP_BATCHES:
            break

    elapsed = time.perf_counter() - t_start
    avg_speed = total_batches / elapsed if elapsed > 0 else 0.0
    print()
    print(f"  peak: {peak:.0f}MB / {total_ram_mb:.0f}MB ({100*peak/total_ram_mb:.1f}%)  avg speed: {avg_speed:.2f} batches/s")
    return peak, hit_safety


def main():
    cache_root = os.path.join(os.environ.get("LE_WAM_ROOT", "."), ".cache")

    print(f"Total RAM: {total_ram_mb:.0f}MB")
    print(f"Target:    {target_mb:.0f}MB ({TARGET_FRACTION*100:.0f}%)")
    print(f"Safety:    {safety_mb:.0f}MB ({SAFETY_FRACTION*100:.0f}%)")

    print("\nPrefetching metadata (once)...")
    _ds = CommunityDataset(repo_id="ehalicki/LeWAM_community_dataset", cache_root=cache_root)
    _ds.prefetch_metadata()
    metas = {"subpaths": _ds.subpaths, "buckets": _ds.buckets, "metas": _ds.metas}
    print(f"Metadata ready: {len(metas['subpaths'])} subdatasets\n")

    pool_size = 2
    buffer_size = 5
    best_pool, best_buffer = pool_size, buffer_size

    while True:
        mem, hit_safety = run_trial(pool_size, buffer_size, cache_root, metas)

        if hit_safety:
            print(f"\nSafety ceiling hit — rolling back to last safe config:")
            print(f"  pool_size={best_pool}  buffer_size={best_buffer}")
            break

        if mem >= target_mb:
            print(f"\nReached target ({TARGET_FRACTION*100:.0f}%) with:")
            print(f"  pool_size={pool_size}  buffer_size={buffer_size}")
            break

        best_pool, best_buffer = pool_size, buffer_size

        remaining_fraction = (target_mb - mem) / total_ram_mb
        if remaining_fraction > 0.20:
            pool_size += 4
            buffer_size += 10
        else:
            pool_size += 1
            buffer_size += 3

        print(f"  below target, stepping up...")


if __name__ == "__main__":
    main()
