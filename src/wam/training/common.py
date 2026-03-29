import os
import subprocess
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def aws_available():
    return subprocess.run(['which', 'aws'], capture_output=True).returncode == 0


def download_checkpoint_from_s3(s3_path: str, local_path: str) -> str:
    """Download a checkpoint from S3 to local_path only if it doesn't already exist. Returns local_path."""
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading checkpoint from {s3_path}...")
        subprocess.run(['aws', 's3', 'cp', s3_path, local_path], check=True)
    else:
        print(f"Checkpoint already exists at {local_path}, skipping download.")
    return local_path


def upload_to_s3_async(local_path: str, s3_path: str):
    """Fire-and-forget S3 upload (non-blocking)."""
    subprocess.Popen(['aws', 's3', 'cp', local_path, s3_path])


def resolve_checkpoint(resume: str, le_wam_root: str, s3_path: str, from_aws: bool) -> str:
    """
    Resolve a checkpoint path, optionally downloading from S3.

    resume:      filename or relative key (e.g. 'MyRun/MyRun_latest.pt')
    le_wam_root: root directory for local runs
    s3_path:     S3 base path (e.g. 's3://bucket/checkpoints')
    from_aws:    if True, download from S3 if not already local
    Returns the local absolute path.
    """
    local = os.path.join(le_wam_root, 'runs', resume)
    if from_aws:
        return download_checkpoint_from_s3(f'{s3_path}/{resume}', local)
    return local


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


def embed_pca_rgb(tokens_list: list, T: int, patch_h: int, patch_w: int) -> list:
    """
    Fit PCA on tokens_list[0], then project all tensors to RGB spatial grids.

    tokens_list: list of (N, D) float tensors (cpu or gpu)
    Returns list of (T, patch_h, patch_w, 3) numpy arrays in [0, 1].
    """
    anchor = tokens_list[0].float().cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(anchor)
    results = []
    for tokens in tokens_list:
        c = pca.transform(tokens.float().cpu().numpy())
        c -= c.min(axis=0)
        c /= c.max(axis=0) + 1e-8
        results.append(c.reshape(T, patch_h, patch_w, 3))
    return results


def save_pca_viz(embeddings, step, run_dir, patch_h, patch_w):
    tokens = embeddings[0].float().cpu()
    T = tokens.shape[0] // (patch_h * patch_w)
    frames_pca = embed_pca_rgb([tokens], T, patch_h, patch_w)[0]

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
def ode_solve(model, past_frames, x1, lang, l_mask, num_steps: int = 10):
    """
    Euler ODE integration from x0 ~ N(0,1) toward x1_pred.

    past_frames: (B, T*H*W, D)
    x1:          (B, T*H*W, D)  used only for shape / device
    Returns predicted x1 of the same shape.
    """
    device = x1.device
    x  = torch.randn_like(x1)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((x.shape[0],), i * dt, device=device)
        v = model.predict_future(x.float(), t, past_frames.float(), lang=lang, l_mask=l_mask)
        x = x + v * dt
    return x


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
    x1 = x1.float()
    if lang is not None:
        lang = lang.float()

    x1_pred   = ode_solve(model, past_frames.float(), x1, lang, l_mask, num_steps=num_ode_steps)
    x1_actual = x1

    T = x1_actual.shape[1] // (patch_h * patch_w)

    actual_rgb, pred_rgb = embed_pca_rgb([x1_actual[0], x1_pred[0]], T, patch_h, patch_w)

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
