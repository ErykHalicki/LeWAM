#!/bin/bash
set -e

if [ -z "$LE_WAM_ROOT" ]; then
    echo "Error: LE_WAM_ROOT environment variable is not set"
    echo "Please add 'export LE_WAM_ROOT=/path/to/LeWAM' to your ~/.bashrc"
    exit 1
fi

DATASET_DIR="$LE_WAM_ROOT/data/somethingsomethingv2"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading Something-Something V2 dataset..."

echo "Downloading part 00 (resuming if partial)..."
curl -L -C - --retry 5 --retry-delay 2 -o 20bn-something-something-v2-00 \
    "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00"

echo "Downloading part 01 (resuming if partial)..."
curl -L -C - --retry 5 --retry-delay 2 -o 20bn-something-something-v2-01 \
    "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01"

if [ ! -f "labels.zip" ]; then
    echo "Downloading labels..."
    curl -L -C - --retry 5 --retry-delay 2 -o labels.zip \
        "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip"
else
    echo "Labels already exist, skipping."
fi

if [ ! -f "20bn-something-something-v2.tar" ]; then
    if [ ! -f "20bn-something-something-v2.tar.gz" ]; then
        echo "Concatenating archive parts into single tar.gz..."
        cat 20bn-something-something-v2-?? > 20bn-something-something-v2.tar.gz

        echo "Cleaning up archive parts..."
        rm -f 20bn-something-something-v2-00 20bn-something-something-v2-01
    fi

    echo "Decompressing tar.gz to uncompressed tar for random access..."
    gunzip -k 20bn-something-something-v2.tar.gz

    echo "Cleaning up tar.gz..."
    rm -f 20bn-something-something-v2.tar.gz
else
    echo "Uncompressed tar already exists, skipping."
fi

echo ""
echo "Something-Something V2 dataset downloaded and prepared!"
echo "Dataset location: $DATASET_DIR"
