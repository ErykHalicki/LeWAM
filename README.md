# LeWorldActionModel (LeWAM)

A video world action model built on VJEPA2 patch embeddings. A flow-matching DiT predicts future video latents, and a transformer IDM decodes actions from the predicted transitions. The separation lets the DiT be pretrained on internet video without action labels, and the IDM finetuned on robot data independently.

Inspired by [LeWorldModel](https://arxiv.org/pdf/2603.19312v1) and [DreamZero](https://dreamzero.github.io).

![Model Architecture](docs/architecture.png)

---

## Stack

### Visual encoder, VJEPA2 (80M params)
VJEPA2 is a video ViT trained with the Joint Embedding Predictive Architecture (JEPA) objective: predict masked patch embeddings from visible context, entirely in latent space (no pixel reconstruction). Additionally, LeWAM provides the training objective of predicting future frame embeddings given past embeddings as conditioning. To prevent trivial collapse (the encoder predicting the same embedding for all inputs), LeWAM uses the SIGReg regularization introduced in LeJEPA and LeWorldModel. I predict that by staying in latent space, it will be possible to train a policy with many less parameters than other contemporary WAM's like DreamZero.

All frames are encoded to per-patch embeddings `z ∈ R^(T×H×W×D)`. The encoder is finetuned end-to-end, with gradients flowing through it from both the DiT prediction loss and the IDM loss.

### Language encoder, T5Gemma (frozen, 270M params)
T5Gemma is an encoder-decoder language model. I use only the encoder to embed task instructions into a sequence of token embeddings `l ∈ R^(S×D)`. It is frozen, with classifier-free guidance dropout during training (randomly zero out `l` with probability `p_drop` to enable unconditional generation at inference).

### Latent predictor, Flow Matching DiT
A Diffusion Transformer trained with flow matching to predict future patch embeddings conditioned on past frames and language.

Flow matching interpolates between noise `x0 ~ N(0, I)` and the target `x1 = z_{t+1}^{gt}`:
```
x_t = (1 - t) * x0 + t * x1,   t ~ U[0, 1]
```
The DiT learns the velocity field `v(x_t, t)` such that integrating it recovers `x1` from `x0`. Training loss is:
```
L_pred = MSE(v_pred, x1 - x0)
```

**Architecture:** each block runs SA, CA, MLP with adaLN-Zero conditioning from the timestep embedding.
- Self-attention on the noisy future patches with 3D RoPE over (T, H, W)
- Cross-attention to past frame patches (3D RoPE), language tokens (no RoPE), and proprioceptive state token (no RoPE)

At inference, an ODE solver integrates the learned velocity field from `t=0` to `t=1` to produce predicted future patch embeddings.

### Inverse Dynamics Model, Transformer IDM
A transformer that predicts an action chunk `a_{0:K}` given the current and predicted future patch embeddings. One action is predicted per predicted future frame, so chunk length equals number of future frames.

**Architecture:** learned action query tokens (one per action step) cross-attend to the frame embeddings, then self-attend to each other. Each block runs CA, SA, MLP. CA runs first because action queries have no input-dependent content before the first cross-attention, so running SA first would be a no-op in the first block.
- Cross-attention to current frame patches (3D RoPE), future frame patches (3D RoPE), and state token (no RoPE)
- Bidirectional self-attention across action steps, with position encoded via learned embeddings

Training loss is MSE against ground truth actions:
```
L_IDM = MSE(a_pred, a_gt)
```

The multimodality of the task lives in the DiT (which future to imagine), not the IDM. I hypothesize that given fixed current and future endpoints, the action distribution is near-unimodal, so regression will be appropriate and the model will not experience mode collapse.

---

## Training objectives

**With action labels:**
```math
L = \underbrace{L_{\text{pred}}}_{\text{flow matching}} + \lambda_{\text{IDM}} \cdot \underbrace{L_{\text{IDM}}}_{\text{MSE}} + \lambda_{\text{SIGReg}} \cdot \underbrace{\text{SIGReg}(Z)}_{\text{anti-collapse}}
```

**Video only (internet pretraining):**
```math
L = \underbrace{L_{\text{pred}}}_{\text{flow matching}} + \lambda_{\text{SIGReg}} \cdot \underbrace{\text{SIGReg}(Z)}_{\text{anti-collapse}}
```

Teacher forcing: during robot data training, the IDM is conditioned on ground truth future patches with probability `p_tf`, and on DiT-predicted patches otherwise. This bridges the train/test distribution gap from imperfect predictions.

---

## Pseudocode

### Inference
```python
z_current = vjepa2_encoder(video_context)          # (T*H*W, D) per-patch embeddings

l = t5gemma_encoder(language_instruction)          # (S, D) token embeddings, or zeros for unconditional

x0 = randn_like(z_current)
z_future = ode_solve(dit, x0, z_current, l, state) # integrate velocity field t: 0 to 1

if video_only:
    return z_future                                # visualise via separate decoder

a_chunk = idm(z_current, z_future, state)          # (K, action_dim)
return z_future, a_chunk
```

### Training step
```python
z_current = vjepa2_encoder(video_context)
z_future_gt = vjepa2_encoder(video_future)

# language dropout for classifier-free guidance
l = t5gemma_encoder(instruction) if rand() > p_drop else zeros(S, D)

# flow matching: sample timestep and interpolate
t = rand()
x0 = randn_like(z_future_gt)
x_t = (1 - t) * x0 + t * z_future_gt
v_pred = dit(x_t, t, z_current, l, state)
L_pred = mse(v_pred, z_future_gt - x0)

if video_only:
    L = L_pred + lambda_sigreg * sigreg(z_current)
else:
    # teacher forcing: condition IDM on GT future or DiT prediction
    z_future_dit = ode_solve(dit, randn_like(z_future_gt), z_current, l, state)
    z_idm_input  = z_future_gt if rand() < p_tf else z_future_dit
    a_pred = idm(z_current, z_idm_input, state)
    L_IDM  = mse(a_pred, a_gt)
    L = L_pred + lambda_idm * L_IDM + lambda_sigreg * sigreg(z_current)
```

---

## Repo structure

```
src/
  wam/
    models/
      common.py     # shared primitives: RoPE3D, SelfAttention, CrossAttention, Block
      DiT.py        # flow matching latent predictor
      IDM.py        # inverse dynamics model
  vjepa2/           # VJEPA2 encoder (submodule)
  scripts/          # dev / loading scripts

tests/
  test_dit.py
  test_idm.py
```

---

## Pretrained models

| Component | Model | Params | Frozen |
|-----------|-------|--------|--------|
| Visual encoder | [VJEPA2](https://github.com/facebookresearch/vjepa2) | 80M | No |
| Language encoder | [T5Gemma-S](https://huggingface.co/google/t5gemma-s-s-prefixlm) | 270M | Yes |

## Datasets

[SmolVLA Training Set](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2)

---

## Open questions

- Can play data / motor babbling train the IDM without teleoperation (cross-embodiment transfer with zero demos)?
- How does teacher forcing affect final IDM performance?
- Does training the models seperately first help with convergance? 
- Can RL improve the policy post-IL?
