# LeWorldActionModel (LeWAM)

A joint video-action world model built on [VJEPA2](https://github.com/facebookresearch/vjepa2) latents. A single flow-matching DiT jointly predicts future video latents and action trajectories, conditioned on context frames, proprioceptive state, and language instructions from a frozen VLM. Block-causal attention lets video and action tokens attend to aligned temporal chunks while staying autoregressive across time.

Inspired by [LeWorldModel](https://arxiv.org/pdf/2603.19312v1) and [DreamZero](https://dreamzero.github.io).

![Model Architecture](docs/architecture.png)

---

## Components

| Component | Model | Frozen |
|-----------|-------|--------|
| Visual encoder | [VJEPA2](https://github.com/facebookresearch/vjepa2) ViT-B | Yes |
| Language encoder | [SmolVLM2-256M](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | Yes |
| Flow-matching DiT | LeWAM (joint video + action) | No |

### Model sizes

| Size | Dim | Depth | Heads | Context frames | Future frames |
|------|-----|-------|-------|----------------|---------------|
| Baby | 256 | 6 | 4 | 4 | 2 |
| Small | 512 | 8 | 8 | 4 | 2 |
| Base | 512 | 12 | 8 | 32 | 8 |
| Large | 1024 | 16 | 16 | 32 | 8 |

---

## Repo structure

```
LeWAM/
├── configs/
│   ├── model/                  # model size configs (baby, small, base, large)
│   └── train/                  # training configs
├── src/
│   ├── wam/
│   │   ├── models/
│   │   │   ├── lewam.py        # joint video-action flow-matching DiT
│   │   │   ├── common.py       # shared primitives (3D RoPE, attention blocks)
│   │   │   ├── action_encoders.py  # state/action encoding and normalization
│   │   │   ├── video_encoder.py    # VJEPA2 encoder wrapper
│   │   │   └── vlm_encoder.py     # SmolVLM2 language encoder
│   │   ├── training/
│   │   │   ├── scripts/
│   │   │   │   ├── train.py                # main training script (Accelerate)
│   │   │   │   ├── precompute_norm_stats.py # action/state normalization stats
│   │   │   │   ├── plot_losses.py          # loss curve visualization
│   │   │   │   └── model_sizes.py          # print model param counts
│   │   │   ├── common.py       # training utilities, ODE visualization, S3 helpers
│   │   │   └── losses.py       # loss functions
│   │   ├── datasets/           # dataset loaders
│   │   └── scripts/            # dev scripts and tests
│   └── vjepa2/                 # VJEPA2 encoder (git submodule)
├── tests/                      # pytest suite
├── paper/                      # LaTeX source
└── docs/
```

---

## Datasets

- [LeWAM community dataset](https://huggingface.co/datasets/ehalicki/LeWAM_community_dataset) -- full training set
- [LeWAM community dataset (small)](https://huggingface.co/datasets/ehalicki/LeWAM_community_dataset_small) -- small subset for overfitting tests

---

## Setup

```bash
git clone --recurse-submodules https://github.com/ErykHalicki/LeWAM
cd LeWAM
export LE_WAM_ROOT=$(pwd)

./src/wam/scripts/init/install_and_test.sh
```

## Training

```bash
accelerate launch src/wam/training/scripts/train.py --config configs/train/default.yaml
```

Training configs control dataset, fps, action fps, batch size, and other hyperparameters. Model configs control architecture size. Norm stats are precomputed automatically at the start of each run.

Checkpoints and loss logs are saved locally to `$LE_WAM_ROOT/runs/` and optionally uploaded to S3.

---

## Environment Variables

Add the following to your shell profile:

```bash
export LE_WAM_ROOT=/path/to/LeWAM
export HF_TOKEN=your_huggingface_token
```

- `LE_WAM_ROOT`: Root directory for configs, caches, weights, and run outputs
- `HF_TOKEN`: Required for downloading models from HuggingFace
