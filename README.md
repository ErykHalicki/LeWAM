# LeWorldActionModel (LeWAM)

A video world action model built on VJEPA2 patch embeddings. A flow-matching DiT predicts future video latents conditioned on past frames and language, and a transformer IDM decodes action chunks from the predicted transitions. The separation lets the DiT pretrain on unlabeled internet video and the IDM finetune independently on robot data.

Inspired by [LeWorldModel](https://arxiv.org/pdf/2603.19312v1) and [DreamZero](https://dreamzero.github.io).

![Model Architecture](docs/architecture.png)

---

## Components

| Component | Model | Params | Frozen |
|-----------|-------|--------|--------|
| Visual encoder | [VJEPA2](https://github.com/facebookresearch/vjepa2) ViT-L | 80M | No |
| Language encoder | [T5Gemma-S](https://huggingface.co/google/t5gemma-s-s-prefixlm) | 270M | Yes |
| Latent predictor | Flow-matching DiT | ~[TBD]M | No |
| Action decoder | Transformer IDM | ~[TBD]M | No |

---

## Repo structure

```
LeWAM/
├── src/
│   ├── wam/
│   │   ├── models/
│   │   │   ├── DiT.py        # flow-matching latent predictor
│   │   │   ├── IDM.py        # inverse dynamics model
│   │   │   ├── common.py     # shared primitives (RoPE3D, attention blocks)
│   │   │   └── losses.py     # SIGReg anti-collapse regularizer
│   │   └── scripts/tests/    # dev / loading scripts
│   └── vjepa2/               # VJEPA2 encoder (git submodule)
├── tests/                    # pytest suite
├── weights/                  # model checkpoints (not committed)
├── paper/                    # LaTeX source
└── docs/
```

---

## Datasets

- [SmolVLA Training Set](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2) — robot finetuning
- [SomethingSomethingV2](https://huggingface.co/datasets/HuggingFaceM4/something_something_v2) — video pretraining

---

## Setup

```bash
export HF_TOKEN={YOUR_TOKEN} # this is needed to be able to install the Gemmma encoder

git clone --recurse-submodules https://github.com/ErykHalicki/LeWAM
cd LeWAM
./src/wam/scripts/init/install_and_test.sh
```

---

## Environment Variables

Add the following to your `~/.bashrc`:

```bash
export LE_WAM_ROOT=/path/to/LeWAM
export HF_TOKEN=your_huggingface_token
```

- `LE_WAM_ROOT`: Used by dataset download scripts to determine where to place data files
- `HF_TOKEN`: Required for downloading the Gemma encoder from HuggingFace
