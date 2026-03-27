# LeWAM Project Instructions

- Always use `uv` instead of `pip` for package management
- Always source the venv before running any Python commands: `source .venv/bin/activate`
- When listing directory contents, exclude `.venv`: `ls | grep -v .venv` or use glob patterns that exclude it

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
