import os
import subprocess
from contextlib import contextmanager

import torch
import matplotlib

@contextmanager
def peak_vram_fraction(device_idx: int = 0):
    """
    Context manager that records the peak VRAM fraction used within the block.

        with peak_vram_fraction() as tracker:
            <forward + backward + optimizer.step>
        print(tracker.fraction)
    """
    class _Tracker:
        fraction: float = 0.0

    tracker = _Tracker()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_idx)
    try:
        yield tracker
    finally:
        peak  = torch.cuda.max_memory_allocated(device_idx)
        total = torch.cuda.get_device_properties(device_idx).total_memory
        tracker.fraction = peak / total


def find_max_batch_size(full_step_fn, target_fraction=0.85, device_idx: int = 0):
    """
    Binary-search for the largest batch size whose peak VRAM fraction stays
    below target_fraction.

    full_step_fn(batch_size) must execute a complete training iteration:
    forward + backward + optimizer.step() + optimizer.zero_grad().
    Should raise torch.cuda.OutOfMemoryError on OOM.
    """
    def _run(batch_size):
        with peak_vram_fraction(device_idx) as tracker:
            full_step_fn(batch_size)
        return tracker.fraction

    lo, hi = 1, 1
    while True:
        try:
            used = _run(hi)
            print(f"  batch_size={hi}: {used*100:.1f}% VRAM peak")
            if used >= target_fraction:
                lo = max(1, hi // 2)
                break
            lo = hi
            hi *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if hi == 1:
                raise RuntimeError("OOM at batch_size=1, reduce crop_size or model size")
            lo = hi // 2
            break

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        try:
            used = _run(mid)
            print(f"  batch_size={mid}: {used*100:.1f}% VRAM peak")
            if used >= target_fraction:
                hi = mid
            else:
                lo = mid
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hi = mid

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_idx)
    print(f"Selected batch_size={lo}")
    return lo
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


_s3_procs: list = []


def upload_to_s3_async(local_path: str, s3_path: str):
    """Non-blocking S3 upload. Call wait_for_s3_uploads() before exit."""
    _s3_procs[:] = [p for p in _s3_procs if p.poll() is None]
    _s3_procs.append(subprocess.Popen(['aws', 's3', 'cp', local_path, s3_path]))


def wait_for_s3_uploads(timeout=120):
    """Block until all pending S3 uploads finish."""
    for p in _s3_procs:
        p.wait(timeout=timeout)
    _s3_procs.clear()


def copy_s3(src: str, dst: str, blocking=False):
    """Server-side S3 copy. Non-blocking by default. Silently ignores missing files."""
    if blocking:
        subprocess.run(['aws', 's3', 'cp', src, dst], capture_output=True)
    else:
        _s3_procs[:] = [p for p in _s3_procs if p.poll() is None]
        _s3_procs.append(subprocess.Popen(['aws', 's3', 'cp', src, dst]))


def resolve_checkpoint(resume: str, cache_root: str, s3_path: str | None = None) -> str:
    """
    Resolve a checkpoint path, downloading from S3 if not found locally.

    resume:     relative key (e.g. 'MyRun/MyRun_latest.pt')
    cache_root: local cache directory (e.g. $LE_WAM_ROOT/.cache)
    s3_path:    S3 base path (e.g. 's3://bucket/checkpoints'), or None
    Returns the local absolute path.
    """
    local = os.path.join(cache_root, resume)
    if not os.path.exists(local):
        if not s3_path or not aws_available():
            raise FileNotFoundError(
                f"Checkpoint not found at {local} and AWS CLI is not available to download it. "
                f"Install awscli or manually place the checkpoint at {local}."
            )
        download_checkpoint_from_s3(f'{s3_path}/{resume}', local)
    return local

def embed_pca_rgb(tokens_list: list, T: int, patch_h: int, patch_w: int) -> list:
    """
    Fit PCA on tokens_list[0], then project all tensors to RGB spatial grids.

    tokens_list: list of (N, D) float tensors (cpu or gpu)
    Returns list of (T, patch_h, patch_w, 3) numpy arrays in [0, 1].
    """
    import numpy as np
    anchor = tokens_list[0].float().cpu().numpy()
    if not np.isfinite(anchor).all():
        return [np.zeros((T, patch_h, patch_w, 3)) for _ in tokens_list]
    pca = PCA(n_components=3)
    pca.fit(anchor)
    results = []
    for tokens in tokens_list:
        c = pca.transform(tokens.float().cpu().numpy())
        c -= c.min(axis=0)
        c /= c.max(axis=0) + 1e-8
        results.append(c.reshape(T, patch_h, patch_w, 3))
    return results


def save_pca_viz(embeddings, step, run_dir, patch_h, patch_w, cam: int = 0, raw_frames=None):
    """
    raw_frames: (T_total, C, H, W) uint8 or float tensor, optional.
                When provided, all raw frames are shown in a top row; PCA patches in a bottom row.
                The two rows have independent column counts via GridSpec.
    """
    from matplotlib.gridspec import GridSpec

    tokens = embeddings[0].float().cpu()
    T_pca = tokens.shape[0] // (patch_h * patch_w)
    frames_pca = embed_pca_rgb([tokens], T_pca, patch_h, patch_w)[0]

    if raw_frames is not None:
        rf = raw_frames.cpu()
        if rf.dtype == torch.uint8:
            rf = rf.float() / 255.0
        else:
            rf = rf.float().clamp(0.0, 1.0)
        T_raw = rf.shape[0]
        fig = plt.figure(figsize=(max(2 * T_raw, 2 * T_pca), 4))
        gs_top = GridSpec(1, T_raw, figure=fig, top=0.88, bottom=0.52)
        gs_bot = GridSpec(1, T_pca, figure=fig, top=0.45, bottom=0.05)
        for i in range(T_raw):
            ax = fig.add_subplot(gs_top[0, i])
            ax.imshow(rf[i].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis('off')
            ax.set_title(f'f={i}', fontsize=6)
        for t in range(T_pca):
            ax = fig.add_subplot(gs_bot[0, t])
            ax.imshow(frames_pca[t])
            ax.axis('off')
            ax.set_title(f't={t}', fontsize=8)
    else:
        fig, axes = plt.subplots(1, T_pca, figsize=(2 * T_pca, 2), squeeze=False)
        for t in range(T_pca):
            axes[0][t].imshow(frames_pca[t])
            axes[0][t].axis('off')
            axes[0][t].set_title(f't={t}', fontsize=8)

    plt.suptitle(f'V-JEPA2 PCA — step {step} cam {cam}')
    plt.savefig(os.path.join(run_dir, 'plots', f'pca-step{step}-cam{cam}.png'), bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def save_ode_viz(
    model,
    raw_batch: dict,
    run_dir: str,
    step: int,
    num_ode_steps: int = 10,
):
    """
    Run Euler ODE integration on a single sample and visualize:
    - GT RGB frames at tubelet boundaries
    - GT / predicted future embeddings (shared PCA)
    - GT / predicted relative actions per dimension

    model must be the unwrapped LeWAM instance (not accelerator-wrapped).
    """
    import math as _math
    import numpy as np
    from matplotlib.gridspec import GridSpec as _GS

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    s: dict = {}
    for k, v in raw_batch.items():
        if hasattr(v, "shape"):
            s[k] = v[:1].to(device)
        elif isinstance(v, list):
            s[k] = v[:1]
        else:
            s[k] = v

    cam_keys = sorted(
        k for k in s
        if k.startswith("observation.images.") and not k.endswith("_is_pad")
    )
    num_cams = len(cam_keys)
    total_frames = model.num_context_frames + model.num_future_frames

    all_frames = torch.stack([s[k][:, :total_frames] for k in cam_keys], dim=1)  # (1, N, T, C, H, W)
    ctx_frames = all_frames[:, :, :model.num_context_frames]
    fut_frames = all_frames[:, :, model.num_context_frames:]

    context_tokens = model.encode_video(ctx_frames)
    future_tokens = model.encode_video(fut_frames)

    state = s["observation.state"].to(dtype=dtype).squeeze(1)
    state = model.normalize_state(state)

    actions = s["action"].to(dtype=dtype)
    dt = 1.0 / model.action_fps
    gt_rel = (actions[:, 1:] - actions[:, :-1]) / dt
    gt_rel = model.normalize_actions(gt_rel)

    lang_tokens, lang_mask = None, None
    if model.vlm_encoder is not None:
        texts = s["task"] if isinstance(s["task"], list) else [s["task"]]
        last_ctx = torch.cat([ctx_frames[:, i, -1] for i in range(num_cams)], dim=-1)
        lang_tokens, lang_mask = model.encode_language(texts, images=last_ctx)

    pred_vid, pred_act = model.ode_solve(
        context_tokens, state, lang_tokens, lang_mask, num_steps=num_ode_steps,
    )
    pred_act_smooth = model.smooth_actions(pred_act)

    patch_h, patch_w = model.frame_latent_h, model.frame_latent_w
    T_emb = model.num_future_tubelets
    tubelet_size = model.VJEPA_TUBELET_SIZE

    stride_idx = [i * tubelet_size for i in range(T_emb)]

    raw_ctx = torch.cat([ctx_frames[0, i] for i in range(num_cams)], dim=-1).float().cpu()
    if raw_ctx.max() > 1.5:
        raw_ctx = raw_ctx / 255.0
    ctx_stride = [i * tubelet_size for i in range(model.num_context_tubelets)]
    ctx_rgb = [raw_ctx[i].permute(1, 2, 0).clamp(0, 1).numpy() for i in ctx_stride if i < raw_ctx.shape[0]]
    T_ctx_show = len(ctx_rgb)

    raw_fut = torch.cat([fut_frames[0, i] for i in range(num_cams)], dim=-1).float().cpu()
    if raw_fut.max() > 1.5:
        raw_fut = raw_fut / 255.0
    rgb_frames = [raw_fut[i].permute(1, 2, 0).clamp(0, 1).numpy() for i in stride_idx if i < raw_fut.shape[0]]
    T_show = len(rgb_frames)

    token_list = [
        future_tokens[0].float().cpu(),
        pred_vid[0].float().cpu(),
    ]
    pca_grids = embed_pca_rgb(token_list, T_emb, patch_h, patch_w)
    gt_emb_frames = pca_grids[0]
    pred_emb_frames = pca_grids[1]

    action_dim = gt_rel.shape[-1]
    n_act_rows = _math.ceil(action_dim / 3)
    n_act_cols = min(action_dim, 3)
    max_cols = max(T_ctx_show, T_show, T_emb)
    fig_w = max(2.5 * max_cols, 10.0)
    fig_h = 12.0 + 1.5 * n_act_rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    img_top = 0.94
    img_bottom = 0.94 - 0.60 * (12.0 / fig_h)
    act_top = img_bottom - 0.04 * (12.0 / fig_h)
    act_bottom = 0.04

    gs_img = _GS(4, max_cols, figure=fig, top=img_top, bottom=img_bottom, hspace=0.06, wspace=0.04)
    gs_act = _GS(n_act_rows, n_act_cols, figure=fig, top=act_top, bottom=act_bottom, hspace=0.55, wspace=0.4)

    for t in range(T_ctx_show):
        ax = fig.add_subplot(gs_img[0, t])
        ax.imshow(ctx_rgb[t])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"f={ctx_stride[t]}", fontsize=7)
        if t == 0:
            ax.set_ylabel("context frame", fontsize=7)

    for t in range(T_show):
        ax = fig.add_subplot(gs_img[1, t])
        ax.imshow(rgb_frames[t])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"f={stride_idx[t]}", fontsize=7)
        if t == 0:
            ax.set_ylabel("ground-truth frame", fontsize=7)

    for t in range(T_emb):
        ax = fig.add_subplot(gs_img[2, t])
        ax.imshow(gt_emb_frames[t])
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.set_ylabel("ground-truth latent frame", fontsize=7)

    for t in range(T_emb):
        ax = fig.add_subplot(gs_img[3, t])
        ax.imshow(pred_emb_frames[t])
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.set_ylabel("predicted latent frame", fontsize=7)

    gt_np = gt_rel[0].float().cpu().numpy()
    pred_np = pred_act[0].float().cpu().numpy()
    pred_smooth_np = pred_act_smooth[0].float().cpu().numpy()
    xs = range(gt_rel.shape[1])
    for d in range(action_dim):
        r, c = divmod(d, n_act_cols)
        ax = fig.add_subplot(gs_act[r, c])
        ax.plot(xs, gt_np[:, d], label="gt", linewidth=1.2)
        ax.plot(xs, pred_np[:, d], label="pred", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.plot(xs, pred_smooth_np[:, d], label="pred (smooth)", linewidth=1.2, linestyle="-.")
        ax.set_ylim(-3, 3)
        ax.set_title(f"action dim {d}", fontsize=7)
        ax.tick_params(labelsize=6)
        if d == 0:
            ax.legend(fontsize=6)

    label = s["task"][0] if isinstance(s["task"], list) else str(s["task"])
    plt.suptitle(f'ODE ({num_ode_steps} steps) -- step {step}\n"{label}"', fontsize=9)
    out_path = os.path.join(run_dir, "ode_viz.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Norm stats ───────────────────────────────────────────────────────────────

def _compute_stats(data: torch.Tensor) -> dict:
    stats = {}
    for q in list(range(1, 6)) + list(range(95, 100)):
        stats[f"q{q}"] = torch.quantile(data, q / 100.0, dim=0)
    stats["mean"] = data.mean(dim=0)
    stats["std"] = data.std(dim=0).clamp(min=1e-6)
    return stats


def compute_norm_stats_community(repo_id: str, cache_root: str, action_fps: int = 30, native_fps: int = 30) -> dict:
    """Compute rel_action/state norm stats from a CommunityDataset. Returns dict ready for ActionPreprocessor."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from lewam.datasets.community_dataset import CommunityDataset

    root = Path(cache_root) / repo_id
    cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)
    cd.prefetch_metadata()

    all_rel_actions, all_states = [], []
    for subpath in cd.metas:
        parquet_files = sorted((root / subpath / "data").glob("**/*.parquet"))
        if not parquet_files:
            continue
        df = pd.concat([
            pd.read_parquet(p, columns=["action", "observation.state", "episode_index", "frame_index"])
            for p in parquet_files
        ], ignore_index=True).sort_values(["episode_index", "frame_index"])

        actions = np.stack(df["action"].values).astype(np.float32)
        states = np.stack(df["observation.state"].values).astype(np.float32)
        ep_idx = df["episode_index"].values

        stride = native_fps // action_fps
        dt = stride / native_fps
        mask = ep_idx[stride:] == ep_idx[:-stride]
        all_rel_actions.append(((actions[stride:] - actions[:-stride]) / dt)[mask])
        all_states.append(states)

    return {
        "rel_action": _compute_stats(torch.from_numpy(np.concatenate(all_rel_actions))),
        "state": _compute_stats(torch.from_numpy(np.concatenate(all_states))),
    }


def compute_norm_stats_lerobot(dataset, action_fps: int) -> dict:
    """Compute rel_action/state norm stats from a LeRobotDataset. Returns dict ready for ActionPreprocessor."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    root = Path(dataset.root)
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    native_fps = dataset.fps

    all_rel_actions, all_states = [], []
    for p in parquet_files:
        df = pd.read_parquet(p, columns=["action", "observation.state", "episode_index", "frame_index"])
        df = df.sort_values(["episode_index", "frame_index"])

        actions = np.stack(df["action"].values).astype(np.float32)
        states = np.stack(df["observation.state"].values).astype(np.float32)
        ep_idx = df["episode_index"].values

        stride = native_fps // action_fps
        dt = stride / native_fps
        mask = ep_idx[stride:] == ep_idx[:-stride]
        all_rel_actions.append(((actions[stride:] - actions[:-stride]) / dt)[mask])
        all_states.append(states)

    return {
        "rel_action": _compute_stats(torch.from_numpy(np.concatenate(all_rel_actions))),
        "state": _compute_stats(torch.from_numpy(np.concatenate(all_states))),
    }

