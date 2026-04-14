"""
Benchmark raw LeRobot dataset decode speed without any DataLoader/worker overhead.

Isolates whether slowdowns are in the decode path or in multiprocessing/accelerate
plumbing. Uses the same delta_timestamps shape as training (num_context=8, num_future=8
at scaled_fps=5, action_fps=30).

Usage:
    python -m lewam.scripts.benchmark_dataloader ehalicki/so101_cube
    python -m lewam.scripts.benchmark_dataloader ehalicki/so101_cube --backend torchcodec
    python -m lewam.scripts.benchmark_dataloader ehalicki/so101_cube --backend pyav --n 100
"""
import argparse
import random
import time

from torch.utils.data import DataLoader
from torchvision import transforms

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--backend", default="torchcodec", choices=["torchcodec", "pyav"])
    parser.add_argument("--num-context", type=int, default=8)
    parser.add_argument("--num-future", type=int, default=8)
    parser.add_argument("--scaled-fps", type=int, default=5)
    parser.add_argument("--action-fps", type=int, default=30)
    parser.add_argument("--n", type=int, default=50, help="Number of random samples to time")
    parser.add_argument("--warm", type=int, default=5, help="Number of warm-up samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=40, help="Batch size for DataLoader sweep")
    parser.add_argument("--num-batches", type=int, default=20, help="Batches per DataLoader config")
    parser.add_argument("--workers-sweep", type=int, nargs="+", default=[0, 1, 4, 8, 16],
                        help="num_workers values to sweep")
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw __getitem__ timing")
    parser.add_argument("--skip-loader", action="store_true", help="Skip DataLoader sweep")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoader")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Resize frames to crop_size x crop_size (matches training). 0 = no resize.")
    parser.add_argument("--mp-context", default=None, choices=[None, "fork", "spawn", "forkserver"],
                        help="multiprocessing_context for DataLoader")
    args = parser.parse_args()

    past_ts = [-(args.num_context - 1 - i) / args.scaled_fps for i in range(args.num_context)]
    future_ts = [(i + 1) / args.scaled_fps for i in range(args.num_future)]
    action_horizon = int(args.num_future / args.scaled_fps * args.action_fps)
    action_ts = [i / args.action_fps for i in range(action_horizon + 1)]

    print(f"Loading meta for {args.repo_id}...")
    meta = LeRobotDataset(repo_id=args.repo_id, revision="main")
    cam_keys = sorted(k for k in meta.meta.features if k.startswith("observation.images."))
    print(f"  cameras: {cam_keys}")
    print(f"  episodes: {meta.num_episodes}  frames: {meta.num_frames}")

    delta_timestamps = {k: past_ts + future_ts for k in cam_keys}
    delta_timestamps["observation.state"] = [0.0]
    delta_timestamps["action"] = action_ts

    image_tx = None
    if args.crop_size > 0:
        image_tx = transforms.Resize((args.crop_size, args.crop_size), antialias=True)
        print(f"\nUsing image_transforms: Resize({args.crop_size},{args.crop_size}) (matches training)")
    else:
        print("\nNo image_transforms (raw native resolution)")

    print(f"Building dataset with backend={args.backend}...")
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        revision="main",
        delta_timestamps=delta_timestamps,
        video_backend=args.backend,
        image_transforms=image_tx,
    )
    actual_backend = getattr(ds, "_video_backend", "<unknown>")
    print(f"  requested backend: {args.backend}")
    print(f"  actual backend:    {actual_backend}")
    if actual_backend != args.backend:
        print(f"  !!! WARNING: backend mismatch, lerobot fell back to {actual_backend}")
    try:
        import torchcodec
        print(f"  torchcodec version: {torchcodec.__version__}")
    except ImportError:
        print(f"  torchcodec NOT installed (would fall back to pyav)")
    print(f"  len={len(ds)}")
    print(f"  context frames/sample: {args.num_context}  future frames/sample: {args.num_future}")
    print(f"  total frames decoded per sample per camera: {args.num_context + args.num_future}")
    print(f"  cameras: {len(cam_keys)}")
    print(f"  total frame decodes per sample: {(args.num_context + args.num_future) * len(cam_keys)}")

    print(f"\n--- Warm-up: {args.warm} sequential samples")
    for i in range(args.warm):
        _ = ds[i]

    single_thread_ms = None
    if not args.skip_raw:
        print(f"\n--- Sequential: {args.n} samples in order")
        t0 = time.time()
        for i in range(args.n):
            _ = ds[i % len(ds)]
        dt = time.time() - t0
        print(f"  total: {dt:.3f}s   per-sample: {dt / args.n * 1000:.1f} ms   throughput: {args.n / dt:.1f} samples/s")

        print(f"\n--- Random: {args.n} samples (shuffled)")
        rng = random.Random(args.seed)
        idxs = [rng.randrange(len(ds)) for _ in range(args.n)]
        t0 = time.time()
        for i in idxs:
            _ = ds[i]
        dt = time.time() - t0
        print(f"  total: {dt:.3f}s   per-sample: {dt / args.n * 1000:.1f} ms   throughput: {args.n / dt:.1f} samples/s")
        single_thread_ms = dt / args.n * 1000

        sample = ds[0]
        print("\n--- Sample shape")
        print(f"  keys: {list(sample.keys())}")
        for k in cam_keys:
            print(f"  {k}: shape={tuple(sample[k].shape)} dtype={sample[k].dtype}")

    if args.skip_loader:
        return

    print(f"\n=== DataLoader sweep: batch_size={args.batch_size} num_batches={args.num_batches}")
    print(f"{'workers':>8} {'persistent':>11} {'prefetch':>9} {'total_s':>9} {'s/batch':>9} {'samples/s':>11} {'vs_ideal':>10}")

    def _run(num_workers: int, persistent: bool, prefetch: int):
        kwargs = dict(
            batch_size=args.batch_size,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=persistent if num_workers > 0 else False,
            prefetch_factor=prefetch if num_workers > 0 else None,
            pin_memory=not args.no_pin_memory,
        )
        if num_workers > 0 and args.mp_context is not None:
            kwargs["multiprocessing_context"] = args.mp_context
        loader = DataLoader(ds, **kwargs)
        it = iter(loader)
        _ = next(it)
        t0 = time.time()
        count = 0
        for _ in range(args.num_batches):
            try:
                _ = next(it)
                count += 1
            except StopIteration:
                break
        dt = time.time() - t0
        s_per_batch = dt / max(count, 1)
        samples_per_s = count * args.batch_size / dt
        if single_thread_ms is not None and num_workers > 0:
            ideal = num_workers * (1000.0 / single_thread_ms)
            vs = f"{samples_per_s / ideal * 100:.0f}%"
        else:
            vs = "-"
        print(f"{num_workers:>8} {str(persistent):>11} {prefetch:>9} {dt:>9.2f} {s_per_batch:>9.2f} {samples_per_s:>11.1f} {vs:>10}")
        del loader, it

    for nw in args.workers_sweep:
        _run(nw, persistent=True, prefetch=2)


if __name__ == "__main__":
    main()
