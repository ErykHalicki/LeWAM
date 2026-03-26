#!/bin/bash
set -e

git clone --recurse-submodules https://github.com/ErykHalicki/LeWAM.git
cd LeWAM

mkdir -p weights
curl -L -o weights/vjepa2_1_vitb_dist_vitG_384.pt \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt

pip install -e .
