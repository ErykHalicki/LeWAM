#!/bin/bash
#SBATCH --job-name=lerobot_reencode
#SBATCH --output=/cluster/scratch/ehalicki/logs/reencode_%A_%a.out
#SBATCH --error=/cluster/scratch/ehalicki/logs/reencode_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048MB
#SBATCH --time=1:00:00

SCRATCH=/cluster/scratch/ehalicki
DATASETS_FILE=$SCRATCH/ehalicki/LeWAM_community_dataset/datasets.txt
DATASET_ROOT=$SCRATCH/ehalicki/LeWAM_community_dataset
VENV=$LE_WAM_ROOT/.venv

GOP=${GOP:-1}
CRF=${CRF:-18}
PRESET=${PRESET:-fast}
SCALE=${SCALE:-256}

source $VENV/bin/activate

mkdir -p $SCRATCH/logs

RELATIVE_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $DATASETS_FILE)

if [ -z "$RELATIVE_PATH" ]; then
    echo "No dataset for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

DATASET_DIR=$DATASET_ROOT/$RELATIVE_PATH

echo "Re-encoding $RELATIVE_PATH (task $SLURM_ARRAY_TASK_ID) gop=$GOP crf=$CRF preset=$PRESET scale=$SCALE"

python -m lewam.scripts.reencode_dataset "$RELATIVE_PATH" \
    --root "$DATASET_DIR" \
    --gop "$GOP" \
    --crf "$CRF" \
    --preset "$PRESET" \
    --scale "$SCALE"

echo "Done: $RELATIVE_PATH"

# Submit with dynamic array size:
# NUM=$(wc -l < /cluster/scratch/ehalicki/ehalicki/LeWAM_community_dataset/datasets.txt) && sbatch --array=0-$((NUM - 1))%12 src/lewam/training/scripts/video_reencode/reencode_datasets.sh
