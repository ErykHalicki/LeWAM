"""
Print a table of model sizes from config YAMLs.

Usage:
    source .venv/bin/activate
    python src/lewam/training/scripts/model_sizes.py
    python src/lewam/training/scripts/model_sizes.py --config-dir path/to/configs/model
"""
import os
import argparse
from pathlib import Path

import yaml
import torch

from lewam.models.lewam import LeWAM

_default = os.path.join(os.environ.get("LE_WAM_ROOT", "."), "configs", "model")

parser = argparse.ArgumentParser()
parser.add_argument("--config-dir", default=_default)
args = parser.parse_args()

configs = sorted(Path(args.config_dir).glob("*.yaml"))
if not configs:
    print(f"No .yaml files found in {args.config_dir}")
    raise SystemExit(1)

rows = []
for path in configs:
    cfg = yaml.safe_load(path.read_text())
    with torch.no_grad():
        m = LeWAM(**cfg)
    trainable = m.count_params(millions=True, trainable_only=True)
    total = m.count_params(millions=True, trainable_only=False)
    rows.append((path.stem, trainable, total, cfg))

rows.sort(key=lambda r: r[2])

header = ["name", "trainable", "total", "dim", "depth", "heads", "ctx_f", "fut_f", "vlm"]
widths = [10, 12, 12, 6, 6, 6, 6, 6, 50]

def fmt_row(vals):
    return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

print(fmt_row(header))
print(fmt_row(["-" * w for w in widths]))
for name, trainable, total, cfg in rows:
    print(fmt_row([
        name,
        f"{trainable:.0f}M",
        f"{total:.0f}M",
        cfg.get("model_dim", ""),
        cfg.get("depth", ""),
        cfg.get("num_heads", ""),
        cfg.get("num_context_frames", ""),
        cfg.get("num_future_frames", ""),
        cfg.get("vlm_model_id", "none"),
    ]))
