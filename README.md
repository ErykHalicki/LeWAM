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
│   ├── lewam/
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
source src/lewam/scripts/init/install.sh
```

On vast.ai (or any environment with an existing venv), skip venv creation:

```bash
source src/lewam/scripts/init/install.sh --external-venv
```

If resuming from a checkpoint (V-JEPA2 weights already baked in), skip the ~1GB download:

```bash
source src/lewam/scripts/init/install.sh --no-vjepa
```

## Training

### Pretraining from scratch (community dataset)

```bash
accelerate launch src/lewam/training/scripts/train.py \
    --config configs/train/pretrain.yaml
```

### Finetuning on a single LeRobot dataset

```bash
accelerate launch src/lewam/training/scripts/train.py \
    --config configs/train/finetune.yaml
```

### Resuming a training run

```bash
accelerate launch src/lewam/training/scripts/train.py \
    --config configs/train/finetune_resume.yaml
```

Training configs control dataset, fps, action fps, batch size, and other hyperparameters. Model configs control architecture size. Norm stats are computed automatically at the start of each run.

Checkpoints and loss logs are saved to `$LE_WAM_ROOT/.cache/<run_tag>/` and optionally uploaded to S3. If a checkpoint is not found locally, it is automatically downloaded from S3.

---

## Environment Variables

The install script sets `LE_WAM_ROOT` automatically. If needed, add the following to your shell profile:

```bash
export LE_WAM_ROOT=/path/to/LeWAM
export HF_TOKEN=your_huggingface_token
```

- `LE_WAM_ROOT`: Root directory for configs, caches, and run outputs
- `HF_TOKEN`: Required for downloading models from HuggingFace
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: Required for S3 checkpoint sync
