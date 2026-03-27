#!/bin/bash
|| true
set -e
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --recurse-submodules https://github.com/ErykHalicki/LeWAM.git
cd LeWAM
uv venv .venv
source .venv/bin/activate

mkdir -p weights
curl -L -o weights/vjepa2_1_vitb_dist_vitG_384.pt \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt

uv tool install hf
hf download google/t5gemma-s-s-prefixlm --local-dir ./weights/t5gemma-s-s-prefixlm

uv pip install .
python src/wam/scripts/training/train.py
exit 0
