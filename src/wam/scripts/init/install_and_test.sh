#!/bin/bash
set -e

# Install uv
curl -LsSf https://astral.sh | sh
source $HOME/.cargo/env

# Clone and setup
uv venv .venv
source .venv/bin/activate

# Downloads
mkdir -p weights
curl -L -o weights/vjepa2_1_vitb_dist_vitG_384.pt \
    https://dl.fbaipublicfiles.com

uv tool install hf
hf download google/t5gemma-s-s-prefixlm --local-dir ./weights/t5gemma-s-s-prefixlm

# Install and run
uv pip install .
python src/wam/scripts/init/strip_gemma_decoder.py
python src/wam/scripts/tests/test_model_loads.py

