"""
Pull the latest losses.json and ODE visualization from S3 and display both.

Usage:
    python src/lewam/training/scripts/monitor.py <run_tag>
    python src/lewam/training/scripts/monitor.py full-dataset-base-model-apr4
    python src/lewam/training/scripts/monitor.py full-dataset-base-model-apr4 --smooth 100
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.optimize import curve_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tag")
    parser.add_argument("--s3-path", default="s3://zima-data/lewam/checkpoints")
    parser.add_argument("--smooth", type=int, default=50)
    args = parser.parse_args()

    le_wam_root = os.environ.get("LE_WAM_ROOT", ".")
    tmp_dir = os.path.join(le_wam_root, ".cache", "monitor", args.run_tag)
    os.makedirs(tmp_dir, exist_ok=True)

    s3_prefix = f"{args.s3_path}/{args.run_tag}"
    losses_local = os.path.join(tmp_dir, "losses.json")

    print("Pulling losses.json...")
    subprocess.run(["aws", "s3", "cp", f"{s3_prefix}/losses.json", losses_local], check=True)

    print("Pulling latest ODE viz...")
    ls = subprocess.run(
        ["aws", "s3", "ls", f"{s3_prefix}/ode_viz/"],
        capture_output=True, text=True,
    )
    ode_files = [line.split()[-1] for line in ls.stdout.strip().splitlines() if line.strip()]
    if ode_files:
        latest_ode = sorted(ode_files, key=lambda f: int(f.split("step")[1].split(".")[0]))[-1]
        ode_local = os.path.join(tmp_dir, latest_ode)
        subprocess.run(["aws", "s3", "cp", f"{s3_prefix}/ode_viz/{latest_ode}", ode_local], check=True)
    else:
        ode_local = None
        print("No ODE visualizations found yet.")

    with open(losses_local) as f:
        data = json.load(f)

    if not data:
        print("No loss data yet.")
        sys.exit(0)

    steps = [d["step"] for d in data]
    total_losses = [d["total_loss"] for d in data]
    video_losses = [d["video_loss"] for d in data]
    action_losses = [d["action_loss"] for d in data]

    print(f"Steps: {steps[0]} - {steps[-1]} ({len(steps)} entries)")
    print(f"Latest: total={total_losses[-1]:.4f}  video={video_losses[-1]:.4f}  action={action_losses[-1]:.4f}")

    _, ax = plt.subplots(figsize=(8, 8))
    ax.plot(steps, total_losses, alpha=0.25, color="steelblue", linewidth=0.8)
    ax.plot(steps, video_losses, alpha=0.25, color="green", linewidth=0.8)
    ax.plot(steps, action_losses, alpha=0.25, color="orange", linewidth=0.8)
    if len(total_losses) >= args.smooth:
        kernel = np.ones(args.smooth) / args.smooth
        smooth_steps = steps[args.smooth // 2 : args.smooth // 2 + len(np.convolve(total_losses, kernel, mode="valid"))]
        for raw, color, name in [
            (total_losses, "steelblue", "total"),
            (video_losses, "green", "video"),
            (action_losses, "orange", "action"),
        ]:
            smoothed = np.convolve(raw, kernel, mode="valid")
            ax.plot(smooth_steps, smoothed, color=color, linewidth=2, label=f"{name} (k={args.smooth})")

    def power_law_with_floor(s, a, b, c):
        return a * np.power(s, -b) + c

    steps_arr = np.array(steps, dtype=float)
    for losses, color, name in [
        (total_losses, "steelblue", "total"),
        (video_losses, "green", "video"),
        (action_losses, "orange", "action"),
    ]:
        arr = np.array(losses)
        proj_steps = np.logspace(np.log10(steps[0]), np.log10(2.5e4), 200)
        try:
            p0 = [arr[0], 0.5, arr[-1] * 0.5]
            bounds = ([0, 0, 0], [np.inf, 5, np.inf])
            popt, _ = curve_fit(power_law_with_floor, steps_arr, arr, p0=p0, bounds=bounds, maxfev=10000)
            a, b, c = popt
            proj_loss = power_law_with_floor(proj_steps, a, b, c)
            ax.plot(proj_steps, proj_loss, color=color, linewidth=1.5, linestyle=":",
                    label=f"{name} proj (floor={c:.3f})")
        except RuntimeError:
            ax.plot([], [], color=color, linewidth=1.5, linestyle=":",
                    label=f"{name} proj (fit failed)")

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlim(left=10, right=2.5e4)
    ax.set_xlabel("step (log)")
    ax.set_ylabel("loss")
    ax.set_title(f"{args.run_tag} -- {len(steps)} steps")
    ax.grid(True, which="major", alpha=0.3)
    plt.tight_layout()
    plt.show()

    if ode_local:
        _, ax2 = plt.subplots(figsize=(12, 7.5))
        img = mpimg.imread(ode_local)
        ax2.imshow(img)
        ax2.axis("off")
        ax2.set_title(os.path.basename(ode_local))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
