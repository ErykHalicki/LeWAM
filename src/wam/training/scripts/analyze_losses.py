"""
Analyze training loss trends and project convergence.

Usage:
    python src/wam/training/scripts/analyze_losses.py <run_tag>
    python src/wam/training/scripts/analyze_losses.py full-dataset-base-model-apr4
    python src/wam/training/scripts/analyze_losses.py full-dataset-base-model-apr4 --pull
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


def analyze(data):
    steps = [d["step"] for d in data]
    total = np.array([d["total_loss"] for d in data])
    video = np.array([d["video_loss"] for d in data])
    action = np.array([d["action_loss"] for d in data])

    print(f"Steps: {steps[0]} - {steps[-1]} ({len(steps)} entries)\n")

    n = len(data)
    if n >= 200:
        boundaries = [0, n // 4, n // 2, 3 * n // 4, n]
    elif n >= 50:
        boundaries = [0, n // 3, 2 * n // 3, n]
    else:
        boundaries = [0, n]

    windows = list(zip(boundaries[:-1], boundaries[1:]))

    print(f"{'Window':>15} {'Total':>10} {'Video':>10} {'Action':>10} {'Total d/step':>12}")
    prev_total = None
    for s, e in windows:
        t_mean = total[s:e].mean()
        v_mean = video[s:e].mean()
        a_mean = action[s:e].mean()
        rate = ""
        if prev_total is not None:
            rate = f"{(t_mean - prev_total) / (steps[e - 1] - steps[s]):.6f}"
        prev_total = t_mean
        print(f"  step {steps[s]:>5}-{steps[min(e - 1, n - 1)]:>5}  {t_mean:>10.4f} {v_mean:>10.4f} {a_mean:>10.4f} {rate:>12}")

    tail = min(50, n // 2)
    if tail >= 10:
        recent = data[-tail:]
        rs = np.array([d["step"] for d in recent])
        print(f"\nLinear trend (last {tail} steps):")
        for name, key in [("total", "total_loss"), ("video", "video_loss"), ("action", "action_loss")]:
            vals = np.array([d[key] for d in recent])
            slope = np.polyfit(rs, vals, 1)[0]
            direction = "decreasing" if slope < 0 else "increasing/flat"
            print(f"  {name:>8} slope: {slope:+.6f}/step ({direction})")

    if n >= 40:
        half = n // 2
        early_mean = total[:half].mean()
        late_mean = total[half:].mean()
        early_std = total[:half].std()
        late_std = total[half:].std()
        improvement = (early_mean - late_mean) / early_mean * 100
        print(f"\nFirst half mean: {early_mean:.4f} (std {early_std:.4f})")
        print(f"Second half mean: {late_mean:.4f} (std {late_std:.4f})")
        print(f"Improvement: {improvement:.1f}%")

        if late_std < early_std * 0.5 and improvement < 5:
            print("WARNING: Loss variance dropping with little improvement - possible plateau")
        elif improvement < 2:
            print("WARNING: Minimal improvement between halves - may be plateauing")
        else:
            print("Loss still actively decreasing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tag")
    parser.add_argument("--pull", action="store_true", help="Pull latest losses.json from S3 first")
    parser.add_argument("--s3-path", default="s3://zima-data/lewam/checkpoints")
    args = parser.parse_args()

    le_wam_root = os.environ.get("LE_WAM_ROOT", ".")
    monitor_dir = os.path.join(le_wam_root, ".cache", "monitor", args.run_tag)
    losses_path = os.path.join(monitor_dir, "losses.json")

    if args.pull:
        os.makedirs(monitor_dir, exist_ok=True)
        s3_prefix = f"{args.s3_path}/{args.run_tag}"
        print("Pulling losses.json from S3...")
        subprocess.run(["aws", "s3", "cp", f"{s3_prefix}/losses.json", losses_path], check=True)
        print()

    if not os.path.exists(losses_path):
        print(f"No losses.json found at {losses_path}. Use --pull to download from S3.")
        sys.exit(1)

    with open(losses_path) as f:
        data = json.load(f)

    if not data:
        print("No loss data yet.")
        sys.exit(0)

    analyze(data)


if __name__ == "__main__":
    main()
