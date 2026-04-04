# LeWAM Project Instructions

- Always use `uv` instead of `pip` for package management
- Always source the venv before running any Python commands: `source .venv/bin/activate`
- When listing directory contents, exclude `.venv`: `ls | grep -v .venv` or use glob patterns that exclude it

## Key training files

- `src/wam/models/lewam.py` — `build_lewam`, `build_lewam_with_encoders`, `build_lewam_for_resume`
- `src/wam/models/action_encoders.py` — `ActionPreprocessor` (p5/p95 norm), `StateEncoder`, `ActionDecoder`
- `src/wam/training/common.py` — `save_ode_viz`, `lookup_language_embeddings`, `resolve_checkpoint`
- `src/wam/training/scripts/precompute_norm_stats.py` — precomputes p5/p95/mean/std for actions and state

## Dataset

- HuggingFace repo: `ehalicki/LeWAM_community_dataset` (full), `ehalicki/LeWAM_community_dataset_small` (overfitting subset)
- Local cache root: `$LE_WAM_ROOT/.cache/`
- norm_stats:       `$LE_WAM_ROOT/.cache/ehalicki/LeWAM_community_dataset[_small]/norm_stats.pt`
- Checkpoints:      `$LE_WAM_ROOT/runs/<MODEL_TAG>/`
- VJEPA2 weights:   `$LE_WAM_ROOT/weights/vjepa2_1_vitb_dist_vitG_384.pt`

