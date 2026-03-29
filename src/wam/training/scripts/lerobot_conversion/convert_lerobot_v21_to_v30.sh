#!/bin/bash
#SBATCH --job-name=lerobot_convert
#SBATCH --output=/cluster/scratch/ehalicki/logs/convert_%A_%a.out
#SBATCH --error=/cluster/scratch/ehalicki/logs/convert_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1048MB
#SBATCH --time=1:00:00

SCRATCH=/cluster/scratch/ehalicki
DATASETS_FILE=$SCRATCH/lerobot_community_dataset/datasets.txt
DATASET_ROOT=$SCRATCH/lerobot_community_dataset
VENV=$LE_WAM_ROOT/.venv

source $VENV/bin/activate

mkdir -p $SCRATCH/logs

RELATIVE_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $DATASETS_FILE)

if [ -z "$RELATIVE_PATH" ]; then
    echo "No dataset for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

DATASET_DIR=$DATASET_ROOT/$RELATIVE_PATH

echo "Converting $RELATIVE_PATH (task $SLURM_ARRAY_TASK_ID)"

python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id="$RELATIVE_PATH" \
    --root="$DATASET_DIR" \
    --push-to-hub=false

echo "Done: $RELATIVE_PATH"

# Submit with dynamic array size:
# NUM=$(wc -l < /cluster/scratch/ehalicki/lerobot_community_dataset/datasets.txt) && sbatch --array=0-$((NUM - 1))%40 src/wam/training/scripts/lerobot_conversion/convert_lerobot_v21_to_v30.sh
