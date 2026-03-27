#!/bin/bash
set -e

# 1. Install uv (Skip if the binary is in the path)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed, skipping."
fi

# 2. Setup venv (Skip if .venv directory exists)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
else
    echo ".venv already exists, skipping."
fi

source .venv/bin/activate

# 3. Downloads (mkdir -p already handles existing folders silently)
mkdir -p weights

# Skip large file download if it exists
if [ ! -f "weights/vjepa2_1_vitb_dist_vitG_384.pt" ]; then
    echo "Downloading V-JEPA weights..."
    curl -L -o weights/vjepa2_1_vitb_dist_vitG_384.pt \
        https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
else
    echo "V-JEPA weights already exist, skipping."
fi

# Skip HF download if the specific model directory exists
if [ ! -d "weights/t5gemma-s-s-prefixlm" ]; then
    echo "Downloading Gemma weights..."
    uv tool install hf
    hf download google/t5gemma-s-s-prefixlm --local-dir ./weights/t5gemma-s-s-prefixlm
else
    echo "Gemma weights already exist, skipping."
fi

# 4. Install and run
uv pip install -e .
python src/wam/scripts/init/strip_gemma_decoder.py
python src/wam/scripts/tests/test_model_loads.py


