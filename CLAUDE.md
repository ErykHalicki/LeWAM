# LeWAM Project Instructions

- Always use `uv` instead of `pip` for package management
- Always source the venv before running any Python commands: `source .venv/bin/activate`
- When listing directory contents, exclude `.venv`: `ls | grep -v .venv` or use glob patterns that exclude it

## Key training files

- `src/wam/training/scripts/LeWAM_community_train.py` — main training script
- `src/wam/models/lewam.py` — `build_lewam`, `build_lewam_with_encoders`, `build_lewam_for_resume`
- `src/wam/models/IDM.py` — IDM transformer (cross-attends past+future → action latents)
- `src/wam/models/DiT.py` — flow-matching DiT (predicts velocity field v = x1 - x0)
- `src/wam/models/encoders.py` — `ActionPreprocessor` (p5/p95 norm), `StateEncoder`, `ActionDecoder`
- `src/wam/training/common.py` — `save_ode_viz`, `lookup_language_embeddings`, `resolve_checkpoint`
- `src/wam/scripts/precompute_norm_stats.py` — precomputes p5/p95/mean/std for actions and state

## Dataset

- HuggingFace repo: `ehalicki/LeWAM_community_dataset` (full), `ehalicki/LeWAM_community_dataset_small` (overfitting subset)
- Local cache root: `$LE_WAM_ROOT/.cache/`
- norm_stats:       `$LE_WAM_ROOT/.cache/ehalicki/LeWAM_community_dataset[_small]/norm_stats.pt`
- task_embeddings:  `$LE_WAM_ROOT/.cache/ehalicki/LeWAM_community_dataset[_small]/task_embeddings.pt`
- Checkpoints:      `$LE_WAM_ROOT/runs/<MODEL_TAG>/`
- VJEPA2 weights:   `$LE_WAM_ROOT/weights/vjepa2_1_vitb_dist_vitG_384.pt`

## Architecture overview

- **DiT**: language-conditioned long-horizon planner; language conditioning lives here only
- **IDM**: 2-frame stateless executor (idm_num_past_frames=1, idm_num_future_frames=1); no language; runs iteratively at inference
- **Staged training curriculum**:
  1. IDM + VJEPA2 fine-tuning with teacher forcing (external action supervision prevents encoder collapse)
  2. Frozen encoder + DiT training
  3. End-to-end alignment
- Teacher forcing: IDM receives GT future VJEPA2 embeddings from frozen encoder during Stage 1
- Multi-camera: tokens concatenated along token dimension; RoPE ids repeated per camera

## Current IDM refactor (in progress)

The IDM is being changed from `num_future_frames=5` to `idm_num_future_frames=1` (2-frame formulation):
- `lewam.py`: add `idm_num_past_frames=1, idm_num_future_frames=1` params to `build_lewam`, pass to IDM constructor
- Training script: replace the monolithic IDM loss block with batched consecutive tubelet pairs (n_pairs=6), no language conditioning in IDM calls

## Repo structure

```
LeWAM/
├── src/
│   ├── wam/                        # Main project package (importable as `wam`)
│   │   ├── models/
│   │   │   ├── DiT.py              # Flow-matching Diffusion Transformer: predicts v(x_t, t, past, lang, state)
│   │   │   ├── IDM.py              # Inverse Dynamics Model: predicts actions from past+future frames
│   │   │   ├── common.py           # Shared building blocks (Block, RoPE, SwiGLU, etc.)
│   │   │   └── losses.py           # SIGReg (Sketch Isotropic Gaussian Regularizer)
│   │   └── scripts/
│   │       └── tests/              # Manual/exploratory scripts (not pytest)
│   │           ├── test_vjepa2_1.py         # Loads VJEPA2.1 encoder from weights/
│   │           ├── test_language_encoder.py
│   │           └── visualize_noise_blending.py
│   └── vjepa2/                     # Git submodule — Meta's VJEPA2 encoder (read-only upstream)
│       ├── src/                    # Core VJEPA2 model code (vision_transformer, predictor, etc.)
│       ├── app/                    # Training apps (vjepa, vjepa_2_1, vjepa_droid)
│       ├── evals/                  # Evaluation harnesses
│       └── tests/                  # VJEPA2's own pytest suite
├── tests/                          # LeWAM pytest suite (run with `pytest tests/`)
│   ├── test_dit.py                 # DiT shape + training smoke tests
│   ├── test_idm.py                 # IDM shape + training smoke tests
│   ├── test_sigreg.py              # SIGReg loss tests
│   └── test_vjepa2_load.py         # Checks VJEPA2 encoder loads from weights/
├── weights/                        # Model checkpoints (not committed to git)
│   └── vjepa2_1_vitb_dist_vitG_384.pt
├── docs/
│   └── architecture.png
├── paper/                          # LaTeX paper source
├── pyproject.toml                  # Package config + pytest settings
└── setup.sh                       # Environment bootstrap script
```
