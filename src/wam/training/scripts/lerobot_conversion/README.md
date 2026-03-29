# LeRobot Community Dataset v2.1 -> v3.0 Conversion

Source: `HuggingFaceVLA/community_dataset_v3` (~791 datasets, 235 contributors)
Cluster: Euler ETH (`ehalicki@euler.ethz.ch`)
Dataset location: `/cluster/scratch/ehalicki/lerobot_community_dataset`
LeRobot version: `0.5.0`

## Steps

### 1. Generate dataset list

```bash
python src/wam/training/scripts/lerobot_conversion/list_all_lerobot_dataset_dirs.py \
    /cluster/scratch/ehalicki/lerobot_community_dataset
```

Outputs `datasets.txt` to the dataset root. Each line is `author/dataset_name`.

### 2. Run conversion as Slurm array job

```bash
NUM=$(wc -l < /cluster/scratch/ehalicki/lerobot_community_dataset/datasets.txt)
sbatch --array=0-$((NUM - 1))%40 src/wam/training/scripts/lerobot_conversion/convert_lerobot_v21_to_v30.sh
```

Conversion is in-place. The script backs up each original dataset as `<name>_old`.

### 3. Monitor progress

```bash
watch -n 10 "find /cluster/scratch/ehalicki/lerobot_community_dataset -name 'info.json' -path '*/meta/info.json' | xargs grep -l '\"codebase_version\": \"v3.0\"' | wc -l"
```

### 4. Cleanup

Remove incomplete conversions (have `data/` but no `meta/info.json`):
```bash
find /cluster/scratch/ehalicki/lerobot_community_dataset -mindepth 2 -maxdepth 2 -type d ! -exec test -f {}/meta/info.json \; -exec rm -rf {} \;
```

Verify `_old` backups have no `tasks.parquet` (v3-only file), then delete them:
```bash
find /cluster/scratch/ehalicki/lerobot_community_dataset -mindepth 2 -maxdepth 2 -type d -name "*_old" -exec find {} -name "tasks.parquet" \;
find /cluster/scratch/ehalicki/lerobot_community_dataset -mindepth 2 -maxdepth 2 -type d -name "*_old" -exec rm -rf {} \;
```

Check all converted datasets have `tasks.parquet`:
```bash
diff \
  <(find /cluster/scratch/ehalicki/lerobot_community_dataset -mindepth 2 -maxdepth 2 -type d ! -name "*_old" -exec test -f {}/meta/info.json \; -print | sort) \
  <(find /cluster/scratch/ehalicki/lerobot_community_dataset -mindepth 2 -maxdepth 2 -type d ! -name "*_old" -exec test -f {}/meta/tasks.parquet \; -print | sort)
```

Any datasets appearing only in the left side are missing `tasks.parquet` and should be deleted manually.

### 5. Regenerate dataset list and verify

After cleanup, regenerate `datasets.txt` to exclude deleted entries:
```bash
python src/wam/training/scripts/lerobot_conversion/list_all_lerobot_dataset_dirs.py /cluster/scratch/ehalicki/lerobot_community_dataset
```

Then verify all remaining datasets load correctly with the official LeRobot loader:
```bash
python src/wam/training/scripts/lerobot_conversion/verify_datasets.py /cluster/scratch/ehalicki/lerobot_community_dataset
```

This attempts to instantiate each dataset via `LeRobotDataset` and reports pass/fail per dataset with frame and episode counts. Failed datasets should be deleted and `datasets.txt` regenerated.

## Final state

321 datasets successfully verified as v3.0.
3 datasets deleted (`LegrandFrederic/Orange-brick-lower-resolution` due to schema mismatch, `Odog16/so100_tea_towel_folding_v1` and `Yotofu/so100_sweeper_shoes` due to missing `tasks.parquet`).
