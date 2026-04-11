"""Generate a grid of frames sampled from various episodes in the LeWAM community dataset."""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.transforms import v2 as transforms
from lewam.datasets.community_dataset import CommunityDataset


def main():
    parser = argparse.ArgumentParser(description="Dataset episode grid visualization")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--repo-id", type=str, default="ehalicki/LeWAM_community_dataset")
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--n-cams", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    le_wam_root = Path(__file__).resolve().parents[2]
    cache_root = args.cache_root or str(le_wam_root / ".cache")

    cd = CommunityDataset(repo_id=args.repo_id, cache_root=cache_root)
    cd.prefetch_metadata()

    ds = cd.datasets[args.n_cams]
    cam_keys = ["observation.images.image"] + [f"observation.images.image{i}" for i in range(2, args.n_cams + 1)]
    rng = random.Random(args.seed)
    crop = transforms.CenterCrop(args.crop_size)

    n_cells = args.rows * args.cols
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_cells]

    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols, args.rows))
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    for idx, ds_idx in enumerate(indices):
        r, c = divmod(idx, args.cols)
        ax = axes[r][c] if args.rows > 1 else axes[c]
        ax.axis("off")
        try:
            sample = ds[ds_idx]
            cam = rng.choice(cam_keys)
            img = crop(sample[cam]).permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(img)
        except Exception:
            pass

    for idx in range(len(indices), n_cells):
        r, c = divmod(idx, args.cols)
        axes[r][c].axis("off")

    output = args.output or str(le_wam_root / "paper" / "figures" / "generated" / "dataset_grid.png")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0)
    print(f"Saved to {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
