"""
Generate attention mask figure for the paper appendix.

Usage:
    python paper/scripts/gen_attn_mask.py
    python paper/scripts/gen_attn_mask.py --output paper/figures/generated/attn_mask.pdf
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from lewam.models.lewam import LeWAM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/figures/generated/attn_mask.png")
    parser.add_argument("--n-ctx", type=int, default=6)
    parser.add_argument("--n-fut", type=int, default=8)
    parser.add_argument("--n-act", type=int, default=8)
    parser.add_argument("--spatial", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=2)
    args = parser.parse_args()

    C, F, A = args.n_ctx, args.n_fut, args.n_act
    N = C + F + A

    mask = LeWAM._build_attn_mask(C, F, A, args.spatial, args.block_size).squeeze().numpy()

    ctx_color = np.array(mcolors.to_rgb("#5BAD72"))
    fut_color = np.array(mcolors.to_rgb("#E8863A"))
    act_color = np.array(mcolors.to_rgb("#4A90D9"))
    empty_color = np.array([0.92, 0.92, 0.92])

    img = np.full((N, N, 3), 1.0)
    for r in range(N):
        if r < C:
            color = ctx_color
        elif r < C + F:
            color = fut_color
        else:
            color = act_color
        for c in range(N):
            img[r, c] = color if mask[r, c] else empty_color

    vbs = args.block_size * args.spatial
    abs_ = args.block_size

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, origin="upper", extent=(-0.5, N - 0.5, N - 0.5, -0.5))

    for i in range(N + 1):
        lw = 0.3
        ax.axhline(i - 0.5, color="white", linewidth=lw)
        ax.axvline(i - 0.5, color="white", linewidth=lw)

    for boundary in [C - 0.5, C + F - 0.5]:
        ax.axhline(boundary, color="black", linewidth=1.2)
        ax.axvline(boundary, color="black", linewidth=1.2)

    for i in range(0, C, vbs):
        if i > 0:
            ax.axhline(i - 0.5, color="black", linewidth=0.5, alpha=0.4)
            ax.axvline(i - 0.5, color="black", linewidth=0.5, alpha=0.4)

    for i in range(0, F, vbs):
        if i > 0:
            ax.axhline(C + i - 0.5, color="black", linewidth=0.5, alpha=0.4)
            ax.axvline(C + i - 0.5, color="black", linewidth=0.5, alpha=0.4)

    for i in range(0, A, abs_):
        if i > 0:
            ax.axhline(C + F + i - 0.5, color="black", linewidth=0.5, alpha=0.4)
            ax.axvline(C + F + i - 0.5, color="black", linewidth=0.5, alpha=0.4)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)

    num_ctx_blocks = C // vbs
    num_fut_blocks = F // vbs
    num_act_blocks = A // abs_

    ctx_ticks = [i for i in range(C)]
    fut_ticks = [C + i for i in range(F)]
    act_ticks = [C + F + i for i in range(A)]

    ctx_labels = [f"$C_{{{i}}}$" for i in range(C)]
    fut_labels = [f"$F_{{{i}}}$" for i in range(F)]
    act_labels = [f"$A_{{{i}}}$" for i in range(A)]

    all_ticks = ctx_ticks + fut_ticks + act_ticks
    all_labels = ctx_labels + fut_labels + act_labels

    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=8)
    ax.set_yticks(all_ticks)
    ax.set_yticklabels(all_labels, fontsize=8)

    for tick, label in zip(ax.get_xticklabels(), all_labels):
        if label.startswith("$C"):
            tick.set_color("#3d7a4d")
        elif label.startswith("$F"):
            tick.set_color("#c06828")
        else:
            tick.set_color("#3570a8")

    for tick, label in zip(ax.get_yticklabels(), all_labels):
        if label.startswith("$C"):
            tick.set_color("#3d7a4d")
        elif label.startswith("$F"):
            tick.set_color("#c06828")
        else:
            tick.set_color("#3570a8")

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_xlabel("Key", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Query", fontsize=11, fontweight="bold", labelpad=8)

    legend_patches = [
        mpatches.Patch(color=ctx_color, label="Context"),
        mpatches.Patch(color=fut_color, label="Future"),
        mpatches.Patch(color=act_color, label="Action"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

    ax.tick_params(length=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
