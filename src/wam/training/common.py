import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def lookup_language_embeddings(
    cache: dict,
    labels: list[str],
    device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    embs  = [cache[l][0] for l in labels]
    masks = [cache[l][1] for l in labels]
    max_len = max(e.shape[0] for e in embs)
    padded_embs  = torch.zeros(len(embs), max_len, embs[0].shape[-1], dtype=dtype)
    padded_masks = torch.ones(len(masks), max_len, dtype=torch.bool)
    for i, (e, m) in enumerate(zip(embs, masks)):
        padded_embs[i, :e.shape[0]]  = e.to(dtype)
        padded_masks[i, :m.shape[0]] = m
    return padded_embs.to(device), padded_masks.to(device)


def save_pca_viz(embeddings, step, run_dir, patch_h, patch_w):
    tokens = embeddings[0].float().cpu().numpy()
    T = tokens.shape[0] // (patch_h * patch_w)
    pca = PCA(n_components=3)
    components = pca.fit_transform(tokens)
    components -= components.min(axis=0)
    components /= components.max(axis=0)
    frames_pca = components.reshape(T, patch_h, patch_w, 3)

    fig, axes = plt.subplots(1, T, figsize=(2 * T, 2), squeeze=False)
    axes = axes[0]
    for t in range(T):
        axes[t].imshow(frames_pca[t])
        axes[t].axis('off')
        axes[t].set_title(f't={t}', fontsize=8)
    plt.suptitle(f'V-JEPA2 PCA — step {step}')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'plots', f'pca-step{step}.png'))
    plt.close(fig)


@torch.no_grad()
def save_ode_viz(model, x1, past_frames, lang, l_mask, label, step, run_dir, patch_h, patch_w, num_ode_steps=10, raw_future_frames=None):
    """
    Run Euler ODE integration x0 → x1_pred and visualize alongside actual x1.

    x1:                (1, T*H*W, D)  actual future embeddings
    past_frames:       (1, T*H*W, D)
    lang:              (1, S, D) | None
    l_mask:            (1, S)    | None
    raw_future_frames: (1, C, T, H, W) | None  ImageNet-normalized, one frame per token
    """
    device = x1.device
    past_frames = past_frames.float()
    x1          = x1.float()
    if lang is not None:
        lang = lang.float()

    x  = torch.randn_like(x1)
    dt = 1.0 / num_ode_steps

    for i in range(num_ode_steps):
        t_val = i * dt
        t = torch.full((1,), t_val, device=device)
        v = model.predict_future(x, t, past_frames, lang=lang, l_mask=l_mask)
        x = x + v * dt

    x1_pred = x.float()
    x1_actual = x1.float()

    T = x1_actual.shape[1] // (patch_h * patch_w)

    tokens_actual = x1_actual[0].cpu().numpy()
    tokens_pred   = x1_pred[0].cpu().numpy()

    pca = PCA(n_components=3)
    pca.fit(tokens_actual)
    def to_rgb(tokens):
        c = pca.transform(tokens)
        c -= c.min(axis=0)
        c /= c.max(axis=0) + 1e-8
        return c.reshape(T, patch_h, patch_w, 3)

    actual_rgb = to_rgb(tokens_actual)
    pred_rgb   = to_rgb(tokens_pred)

    n_rows = 3 if raw_future_frames is not None else 2
    fig, axes = plt.subplots(n_rows, T, figsize=(2 * T, 2 * n_rows), squeeze=False)

    row = 0
    if raw_future_frames is not None:
        frames = raw_future_frames[0].permute(1, 0, 2, 3).float().cpu()  # (T, C, H, W)
        frames = model.video_encoder.preprocessor.unnormalize(frames)
        frames = F.interpolate(frames, size=(patch_h * 8, patch_w * 8), mode='bilinear', align_corners=False)
        frames_np = frames.permute(0, 2, 3, 1).numpy()
        for t in range(T):
            axes[row][t].imshow(frames_np[t])
            axes[row][t].axis('off')
            axes[row][t].set_title(f't={t}', fontsize=8)
        axes[row][0].set_ylabel('gt frame', fontsize=8)
        row += 1

    for t in range(T):
        axes[row][t].imshow(actual_rgb[t])
        axes[row][t].axis('off')
        if raw_future_frames is None:
            axes[row][t].set_title(f't={t}', fontsize=8)
    axes[row][0].set_ylabel('gt embed', fontsize=8)
    row += 1

    for t in range(T):
        axes[row][t].imshow(pred_rgb[t])
        axes[row][t].axis('off')
    axes[row][0].set_ylabel('pred embed', fontsize=8)

    plt.suptitle(f'ODE integration ({num_ode_steps} steps) — step {step}\n"{label}"', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'plots', f'ode-step{step}.png'))
    plt.close(fig)
