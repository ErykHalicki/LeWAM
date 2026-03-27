#!/bin/bash
git clone --recurse-submodules https://github.com/ErykHalicki/LeWAM
cd LeWAM
./src/wam/scripts/init/install_and_test.sh

# This script can be run using slurm or similar
