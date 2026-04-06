# LeWAM Project Instructions

- Always use `uv` instead of `pip` for package management
- Always source the venv before running any Python commands: `source .venv/bin/activate`
- When listing directory contents, exclude `.venv`: `ls | grep -v .venv` or use glob patterns that exclude it

## Key files

- `src/lewam/models/lewam.py` — `LeWAM` class (constructor, `from_checkpoint`, `ode_solve`)
- `src/lewam/models/action_encoders.py` — `ActionPreprocessor` (quantile-based norm), `StateEncoder`, `ActionEncoder`
- `src/lewam/models/video_encoder.py` — `VJEPA2VideoEncoder`, `VJEPA2VideoPreprocessor` (frozen VJEPA2-B, multi-camera stitching)
- `src/lewam/models/vlm_encoder.py` — `VLMEncoder` (truncated SmolVLM2, frozen)
- `src/lewam/models/common.py` — `Block`, `RoPE3D`, `SelfAttention`, `CrossAttention`, `PatchPositionIds`
- `src/lewam/training/common.py` — `save_ode_viz`, `resolve_checkpoint`, `find_max_batch_size`, `compute_norm_stats_*`
- `src/lewam/training/scripts/train.py` — main training loop (flow matching, gradient accumulation, S3 sync)

## Dataset

- HuggingFace repo: `ehalicki/LeWAM_community_dataset` (full), `ehalicki/LeWAM_community_dataset_small` (overfitting subset)
- Local cache root: `$LE_WAM_ROOT/.cache/`
- norm_stats:       `$LE_WAM_ROOT/.cache/ehalicki/LeWAM_community_dataset[_small]/norm_stats.pt`
- Checkpoints:      `$LE_WAM_ROOT/runs/<MODEL_TAG>/`
- VJEPA2 weights:   `$LE_WAM_ROOT/weights/vjepa2_1_vitb_dist_vitG_384.pt`

