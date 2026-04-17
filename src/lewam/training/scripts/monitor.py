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
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.optimize import curve_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tag")
    parser.add_argument("--s3-path", default="s3://lewam/checkpoints")
    parser.add_argument("--smooth", type=int, default=50)
    parser.add_argument("--ode-step", type=int, default=None,
                        help="Specific training step to visualize (e.g. 1500 → ode-step1500.png). "
                             "Defaults to the latest available.")
    args = parser.parse_args()

    le_wam_root = os.environ.get("LE_WAM_ROOT", ".")
    tmp_dir = os.path.join(le_wam_root, ".cache", "monitor", args.run_tag)
    os.makedirs(tmp_dir, exist_ok=True)

    s3_prefix = f"{args.s3_path}/{args.run_tag}"
    losses_local = os.path.join(tmp_dir, "losses.json")
    config_local = os.path.join(tmp_dir, "config.json")

    print("Pulling losses.json...")
    subprocess.run(["aws", "s3", "cp", f"{s3_prefix}/losses.json", losses_local], check=True)

    print("Pulling config.json...")
    cfg_result = subprocess.run(
        ["aws", "s3", "cp", f"{s3_prefix}/config.json", config_local],
        capture_output=True, text=True,
    )
    warmup_steps = 0
    action_weight = 1.0
    if cfg_result.returncode == 0:
        with open(config_local) as f:
            run_cfg = json.load(f)
        warmup_steps = int(run_cfg.get("train", {}).get("warmup_steps", 0))
        action_weight = float(run_cfg.get("train", {}).get("action_weight", 1.0))
        print(f"  warmup_steps={warmup_steps}  action_weight={action_weight}")
    else:
        print(f"  config.json not found ({cfg_result.stderr.strip()})")

    if args.ode_step is not None:
        print(f"Pulling ODE viz for step {args.ode_step}...")
    else:
        print("Pulling latest ODE viz...")
    ls = subprocess.run(
        ["aws", "s3", "ls", f"{s3_prefix}/ode_viz/"],
        capture_output=True, text=True,
    )
    ode_entries = []
    for line in ls.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        date, time_, name = parts[0], parts[1], parts[-1]
        step_num = None
        m = re.match(r"ode-step(\d+)\.png$", name)
        if m:
            step_num = int(m.group(1))
        ode_entries.append((f"{date} {time_}", name, step_num))

    ode_local = None
    if args.ode_step is not None:
        match = next((e for e in ode_entries if e[2] == args.ode_step), None)
        if match is None:
            available = sorted(e[2] for e in ode_entries if e[2] is not None)
            print(f"No ODE viz found for step {args.ode_step}. Available: {available}")
        else:
            chosen = match[1]
            ode_local = os.path.join(tmp_dir, chosen)
            subprocess.run(["aws", "s3", "cp", f"{s3_prefix}/ode_viz/{chosen}", ode_local], check=True)
    elif ode_entries:
        latest_ode = sorted(ode_entries)[-1][1]
        ode_local = os.path.join(tmp_dir, latest_ode)
        subprocess.run(["aws", "s3", "cp", f"{s3_prefix}/ode_viz/{latest_ode}", ode_local], check=True)
    else:
        print("No ODE visualizations found yet.")

    with open(losses_local) as f:
        data = json.load(f)

    data = [d for d in data if d.get("total_loss") is not None and d["total_loss"] != 0.0]

    if not data:
        print("No loss data yet.")
        sys.exit(0)

    step_offset = data[0]["step"]
    steps = np.array([d["step"] - step_offset for d in data], dtype=float)
    total_losses = np.array([d["total_loss"] for d in data], dtype=float)
    video_losses = np.array([d["video_loss"] for d in data], dtype=float)
    action_losses = np.array([d["action_loss"] for d in data], dtype=float) / action_weight

    val_steps, val_total, val_video, val_action = [], [], [], []
    for d in data:
        if "val_total_loss" in d:
            val_steps.append(d["step"] - step_offset)
            val_total.append(d["val_total_loss"])
            val_video.append(d["val_video_loss"])
            val_action.append(d["val_action_loss"] / action_weight)

    print(f"Steps: {int(step_offset)} - {int(step_offset + steps[-1])} ({len(steps)} entries, offset by {int(step_offset)})")
    print(f"Latest: total={total_losses[-1]:.4f}  video={video_losses[-1]:.4f}  action={action_losses[-1]:.4f}")
    if val_steps:
        print(
            f"Latest val (step {val_steps[-1]}): "
            f"total={val_total[-1]:.4f}  video={val_video[-1]:.4f}  action={val_action[-1]:.4f}"
        )

    _, ax = plt.subplots(figsize=(8, 8))

    series = [
        (video_losses, "green", "video"),
        (action_losses, "orange", "action"),
    ]

    for arr, color, _ in series:
        nz = arr != 0.0
        ax.plot(steps[nz], arr[nz], alpha=0.25, color=color, linewidth=0.8)

    smoothed_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if len(steps) >= args.smooth:
        kernel = np.ones(args.smooth) / args.smooth
        for arr, color, name in series:
            nz = arr != 0.0
            if nz.sum() < args.smooth:
                continue
            s_steps = steps[nz]
            s_vals = arr[nz]
            smoothed = np.convolve(s_vals, kernel, mode="valid")
            sm_steps = s_steps[args.smooth // 2 : args.smooth // 2 + len(smoothed)]
            ax.plot(sm_steps, smoothed, color=color, linewidth=2, label=f"{name} (k={args.smooth})")
            smoothed_series[name] = (sm_steps, smoothed)

    if val_steps:
        for vals, color, name in [
            (val_video, "green", "val video"),
            (val_action, "orange", "val action"),
        ]:
            ax.plot(val_steps, vals, color=color, linewidth=1.5, linestyle="--",
                    marker="o", markersize=3, label=name)

    def power_law_with_floor(s, a, b, c):
        return a * np.power(s, -b) + c

    def _fit_and_plot(fit_steps, fit_arr, color, label_prefix, linestyle=":"):
        if fit_steps.size < 5:
            return
        proj_steps = np.logspace(np.log10(max(fit_steps[0], 1.0)), np.log10(2e5), 200)
        try:
            p0 = [fit_arr[0], 0.5, fit_arr[-1] * 0.5]
            bounds = ([0, 0, 0], [np.inf, 5, np.inf])
            popt, _ = curve_fit(power_law_with_floor, fit_steps, fit_arr, p0=p0, bounds=bounds, maxfev=10000)
            a, b, c = popt
            proj_loss = power_law_with_floor(proj_steps, a, b, c)
            ax.plot(proj_steps, proj_loss, color=color, linewidth=1.5, linestyle=linestyle,
                    label=f"{label_prefix} proj (floor={c:.3f})")
        except RuntimeError:
            ax.plot([], [], color=color, linewidth=1.5, linestyle=linestyle,
                    label=f"{label_prefix} proj (fit failed)")

    for arr, color, name in series:
        if name in smoothed_series:
            fs, fv = smoothed_series[name]
        else:
            nz = arr != 0.0
            fs = steps[nz]
            fv = arr[nz]
        if fs.size == 0:
            continue
        half_step = fs[len(fs) // 2]
        fit_mask = fs >= half_step
        _fit_and_plot(fs[fit_mask], fv[fit_mask], color, name)

    if val_steps:
        for vals, color, name in [
            (val_video, "green", "val video"),
            (val_action, "orange", "val action"),
        ]:
            v_steps = np.asarray(val_steps, dtype=float)
            v_vals = np.asarray(vals, dtype=float)
            nz = v_vals != 0.0
            v_steps = v_steps[nz]
            v_vals = v_vals[nz]
            if v_steps.size == 0:
                continue
            half_step = v_steps[len(v_steps) // 2]
            fit_mask = v_steps >= half_step
            if fit_mask.sum() < 5:
                print(f"  {name} fit skipped: only {int(fit_mask.sum())} val point(s) in second half, need >=5")
                continue
            _fit_and_plot(v_steps[fit_mask], v_vals[fit_mask], color, name, linestyle=(0, (1, 1)))

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{args.run_tag} -- {len(steps)} steps")
    ax.grid(True, which="major", alpha=0.3)
    plt.tight_layout()
    plt.show()

    grad_steps, grad_cos, grad_a_norm, grad_v_norm, grad_ratio = [], [], [], [], []
    for d in data:
        if "grad_cos" in d:
            grad_steps.append(d["step"])
            grad_cos.append(d["grad_cos"])
            grad_a_norm.append(d["grad_action_norm"])
            grad_v_norm.append(d["grad_video_norm"])
            grad_ratio.append(d["grad_ratio_v_a"])

    if grad_steps:
        print(
            f"Latest grad (step {grad_steps[-1]}): "
            f"cos={grad_cos[-1]:+.3f}  ||a||={grad_a_norm[-1]:.3e}  "
            f"||v||={grad_v_norm[-1]:.3e}  v/a={grad_ratio[-1]:.2f}"
        )
        _, (axc, axn) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axc.plot(grad_steps, grad_cos, color="purple", marker="o", markersize=3, linewidth=1.2)
        axc.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        axc.axhline(1.0, color="black", linewidth=0.4, alpha=0.2, linestyle=":")
        axc.axhline(-1.0, color="black", linewidth=0.4, alpha=0.2, linestyle=":")
        axc.set_ylim(-1.05, 1.05)
        axc.set_ylabel("cos(g_action, g_video)")
        axc.set_title(f"{args.run_tag} -- gradient conflict")
        axc.grid(True, alpha=0.3)

        axn.plot(grad_steps, grad_a_norm, color="orange", marker="o", markersize=3,
                 linewidth=1.2, label="||g_action||")
        axn.plot(grad_steps, grad_v_norm, color="green", marker="o", markersize=3,
                 linewidth=1.2, label="||g_video||")
        axn.set_yscale("log")
        axn.set_ylabel("grad norm")
        axn.set_xlabel("step")
        axn.legend()
        axn.grid(True, which="both", alpha=0.3)

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
