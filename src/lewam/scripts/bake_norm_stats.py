"""
Bake norm_stats into an existing LeWAM checkpoint.

Usage:
    python -m lewam.scripts.bake_norm_stats <checkpoint.pt> <norm_stats.pt> [--output <out.pt>]

If --output is omitted, the checkpoint is overwritten in place.
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Bake norm_stats into a LeWAM checkpoint")
    parser.add_argument("checkpoint", help="Path to LeWAM checkpoint (.pt)")
    parser.add_argument("norm_stats", help="Path to norm_stats.pt")
    parser.add_argument("--output", "-o", default=None, help="Output path (default: overwrite in place)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = torch.load(args.norm_stats, map_location="cpu", weights_only=True)

    ckpt["norm_stats"] = stats
    out_path = args.output or args.checkpoint
    torch.save(ckpt, out_path)
    print(f"Saved checkpoint with baked norm_stats to {out_path}")


if __name__ == "__main__":
    main()
