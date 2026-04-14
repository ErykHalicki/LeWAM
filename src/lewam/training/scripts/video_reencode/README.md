# LeRobot Dataset Video Re-encoding Pipeline

Slurm array pipeline that re-encodes every video in each LeRobot dataset to h264 short-GOP (default all-intra) at 256x256. Trades disk size for dramatically faster random-seek decode in training dataloaders.

Mirrors `lerobot_conversion/` in structure. Each array task runs `lewam.scripts.reencode_dataset` against one dataset directory, passed via `--root` (not `HF_LEROBOT_HOME`).

## Steps

### 1. Generate dataset list

Reuses the list script from `lerobot_conversion/`:
```bash
python src/lewam/training/scripts/lerobot_conversion/list_all_lerobot_dataset_dirs.py \
    /cluster/scratch/ehalicki/lerobot_community_dataset
```

### 2. Run re-encode as Slurm array job

```bash
NUM=$(wc -l < /cluster/scratch/ehalicki/lerobot_community_dataset/datasets.txt)
sbatch --array=0-$((NUM - 1))%40 src/lewam/training/scripts/video_reencode/reencode_datasets.sh
```

Re-encoding is in-place. `meta/info.json` is updated to reflect the new codec and resolution.

Override ffmpeg knobs via env vars:
```bash
GOP=4 CRF=20 SCALE=0 sbatch --array=0-$((NUM - 1))%40 src/lewam/training/scripts/video_reencode/reencode_datasets.sh
```

Defaults: `GOP=1 CRF=18 PRESET=fast SCALE=256`.

### 3. Monitor progress

```bash
squeue -u $USER
tail -f /cluster/scratch/ehalicki/logs/reencode_*.out
```

## Notes

- `reencode_dataset.py` now has a `--root` flag that points directly at a dataset directory, bypassing the default `HF_LEROBOT_HOME/repo_id` join. That is what makes this pipeline work against datasets in arbitrary scratch paths.
- `--cpus-per-task=4` gives libx264 enough cores to saturate; bump higher if you hit a slow dataset.
- All-intra (`GOP=1`) roughly doubles file size vs long-GOP but makes random-seek decode essentially free, which is the whole point for training.
