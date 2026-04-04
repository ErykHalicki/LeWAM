#!/bin/bash
set -e

EXTERNAL_VENV=0
for arg in "$@"; do
    [ "$arg" = "--external-venv" ] && EXTERNAL_VENV=1
done

export LE_WAM_ROOT=$(pwd)
if ! grep -q 'LE_WAM_ROOT' ~/.bashrc; then
    export LE_WAM_ROOT=$(pwd)
    echo "export LE_WAM_ROOT=$(pwd)" >> ~/.bashrc
    echo "Added LE_WAM_ROOT to ~/.bashrc"
fi

# 1. Install uv (Skip if the binary is in the path)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed, skipping."
fi

# 2. Setup venv (Skip if .venv directory exists or --external-venv)
if [ "$EXTERNAL_VENV" = "0" ]; then
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv .venv
    else
        echo ".venv already exists, skipping."
    fi
    source .venv/bin/activate
fi

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

# 4. Install and run
uv pip install -e .
./src/wam/scripts/init/install_aws.sh


