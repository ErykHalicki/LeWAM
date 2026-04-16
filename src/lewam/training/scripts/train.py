"""
LeWAM training script (Accelerate).

Usage (community dataset, pretraining):
    accelerate launch src/lewam/training/scripts/train.py --config configs/train/default.yaml

Usage (single lerobot dataset, finetuning):
    accelerate launch src/lewam/training/scripts/train.py --config configs/train/finetune.yaml

CLI flags override values from the YAML config.
"""
import os
import warnings

le_wam_root = os.environ.get("LE_WAM_ROOT")
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(le_wam_root, ".cache", "torch_compile")

import argparse
from contextlib import nullcontext
import datetime
import json
import random
import time
from zoneinfo import ZoneInfo

import yaml
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from lewam.models.lewam import LeWAM
from lewam.training.common import (
    aws_available, copy_s3, find_max_batch_size,
    resolve_checkpoint, save_ode_viz, upload_to_s3_async, wait_for_s3_uploads,
    compute_norm_stats_community, compute_norm_stats_lerobot,
)

PATCH_SIZE = LeWAM.VJEPA_PATCH_SIZE
TUBELET = LeWAM.VJEPA_TUBELET_SIZE

# ── Data helpers ─────────────────────────────────────────────────────────────

def _camera_keys(batch: dict) -> list[str]:
    return sorted(
        k for k in batch
        if k.startswith("observation.images.") and not k.endswith("_is_pad")
    )


def get_camera_frames(batch: dict, num_frames: int) -> torch.Tensor:
    """
    Collect all camera streams into a (B, N, T, C, H, W) tensor.
    Cameras are ordered by key name.
    """
    return torch.stack([batch[k][:, :num_frames] for k in _camera_keys(batch)], dim=1)


def count_cameras(batch: dict) -> int:
    return len(_camera_keys(batch))


class _SafeDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        try:
            return self._ds[idx]
        except Exception:
            return None


def _collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


def _split_episodes(episode_indices, val_fraction, seed):
    indices = list(episode_indices)
    if len(indices) < 2:
        return indices, []
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, round(len(indices) * val_fraction))
    return sorted(indices[:-n_val]), sorted(indices[-n_val:])


def _infinite_interleaved(loaders, accelerator):
    iters = {n: iter(l) for n, l in loaders.items()}
    keys = list(loaders.keys())
    weights = [len(loaders[n].dataset) for n in keys]
    while True:
        choice = [random.choices(keys, weights=weights, k=1)[0]] if accelerator.is_main_process else [None]
        broadcast_object_list(choice, from_process=0)
        n = choice[0]
        try:
            batch = next(iters[n])
        except StopIteration:
            iters[n] = iter(loaders[n])
            batch = next(iters[n])
        if batch is not None:
            yield batch


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(argv=None):
    """
    Load training config from YAML, then apply CLI overrides.
    The YAML 'model_config' key points to a separate model YAML.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, required=True)
    pre_args, remaining = pre.parse_known_args(argv)

    with open(os.path.join(le_wam_root, pre_args.config)) as f:
        train_cfg = yaml.safe_load(f)

    p = argparse.ArgumentParser()
    for key, val in train_cfg.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            p.add_argument(flag, default=val, action="store_true" if not val else "store_false")
        elif val is None:
            p.add_argument(flag, default=None, type=str)
        else:
            p.add_argument(flag, default=val, type=type(val))
    args = p.parse_args(remaining)
    train_args = vars(args)

    model_config_path = train_args.pop("model_config")
    with open(os.path.join(le_wam_root, model_config_path)) as f:
        model_cfg = yaml.safe_load(f)

    return model_cfg, train_args, pre_args.config


# ── Training step ────────────────────────────────────────────────────────────

def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _get_pad_masks(raw_batch, num_future_frames, num_future_tubelets, spatial, action_horizon):
    """
    Extract is_pad flags from a lerobot batch and convert to token-level masks.

    Returns (future_valid, action_valid):
        future_valid: (B, N_fut) bool, True = valid token
        action_valid: (B, N_act) bool, True = valid token
    Both are None when the batch has no is_pad keys.
    """
    cam_pad_keys = sorted(
        k for k in raw_batch
        if k.startswith("observation.images.") and k.endswith("_is_pad")
    )
    action_pad_key = "action_is_pad"
    if not cam_pad_keys and action_pad_key not in raw_batch:
        return None, None

    B = next(v.shape[0] for v in raw_batch.values() if hasattr(v, "shape"))
    device = next(v.device for v in raw_batch.values() if hasattr(v, "device"))

    if cam_pad_keys:
        cam_pads = torch.stack([raw_batch[k] for k in cam_pad_keys], dim=1)  # (B, N_cams, T)
        frame_is_pad = cam_pads.any(dim=1)  # (B, T) -- padded if ANY camera is padded
        fut_pad = frame_is_pad[:, -num_future_frames:]  # (B, num_future_frames)
        tub_pad = fut_pad.unfold(1, TUBELET, TUBELET).any(dim=-1)  # (B, num_future_tubelets)
        future_valid = ~tub_pad.unsqueeze(-1).expand(-1, -1, spatial).reshape(B, -1)  # (B, N_fut)
    else:
        future_valid = torch.ones(B, num_future_tubelets * spatial, dtype=torch.bool, device=device)

    if action_pad_key in raw_batch:
        act_pad = raw_batch[action_pad_key]  # (B, action_horizon + 1)
        rel_pad = act_pad[:, :-1] | act_pad[:, 1:]  # (B, action_horizon)
        action_valid = ~rel_pad
    else:
        action_valid = torch.ones(B, action_horizon, dtype=torch.bool, device=device)

    return future_valid, action_valid


def prepare_batch(model, raw_batch, num_cameras):
    """
    Collect camera frames, encode video, normalize actions/state.
    Returns all tensors needed for the flow matching forward pass,
    plus validity masks for loss masking.
    """
    m = _unwrap(model)
    total_frames = m.num_context_frames + m.num_future_frames
    all_frames = get_camera_frames(raw_batch, total_frames)  # (B, N, T, C, H, W)

    ctx_frames = all_frames[:, :, :m.num_context_frames]
    fut_frames = all_frames[:, :, m.num_context_frames:]

    crop_size = m.video_encoder.preprocessor.crop_size
    frame_latent_h = crop_size // PATCH_SIZE
    frame_latent_w = (crop_size // PATCH_SIZE) * num_cameras
    if frame_latent_h != m.frame_latent_h or frame_latent_w != m.frame_latent_w:
        m.set_patch_grid(frame_latent_h, frame_latent_w, num_cameras)

    spatial = frame_latent_h * frame_latent_w
    future_valid, action_valid = _get_pad_masks(
        raw_batch, m.num_future_frames, m.num_future_tubelets, spatial, m.action_horizon,
    )

    context_tokens = m.encode_video(ctx_frames)
    future_tokens = m.encode_video(fut_frames)

    state = raw_batch["observation.state"].squeeze(1)
    state = m.normalize_state(state)

    actions = raw_batch["action"]
    dt = 1.0 / m.action_fps
    rel_velocity = (actions[:, 1:] - actions[:, :-1]) / dt
    rel_velocity = m.normalize_actions(rel_velocity)

    lang_tokens, lang_mask = None, None
    if m.vlm_encoder is not None:
        texts = raw_batch["task"]
        last_ctx_frame = torch.cat([ctx_frames[:, i, -1] for i in range(num_cameras)], dim=-1)
        lang_tokens, lang_mask = m.encode_language(texts, images=last_ctx_frame)

    return context_tokens, future_tokens, rel_velocity, state, lang_tokens, lang_mask, future_valid, action_valid


def _masked_mse(pred, target, valid_mask, name=""):
    """Per-element MSE, averaged only over valid (non-padded) positions.
    Loudly warns and returns 0 (no gradient contribution) when the whole
    batch is masked out, instead of silently swallowing it."""
    if valid_mask is None or valid_mask.all():
        return F.mse_loss(pred, target)
    if not valid_mask.any():
        per_sample = valid_mask.sum(dim=tuple(range(1, valid_mask.ndim))).tolist()
        warnings.warn(
            f"_masked_mse({name}): every token in the batch is padded — "
            f"per-sample valid counts={per_sample}. Returning 0; this step "
            f"contributes no {name} gradient.",
            stacklevel=2,
        )
        return (pred * 0.0).sum()
    mask = valid_mask.unsqueeze(-1).expand_as(pred)
    return (F.mse_loss(pred, target, reduction="none") * mask).sum() / mask.sum()


def train_step(model, raw_batch, accelerator, num_cameras, action_weight=0.0, lang_drop_rate=0.0):
    context_tokens, future_tokens, rel_velocity, state, lang_tokens, lang_mask, future_valid, action_valid = (
        prepare_batch(model, raw_batch, num_cameras)
    )

    if lang_tokens is not None and lang_drop_rate > 0.0:
        B = context_tokens.shape[0]
        drop = torch.rand(B, device=lang_tokens[0].device) < lang_drop_rate
        lang_tokens = [h.clone() for h in lang_tokens]
        for h in lang_tokens:
            h[drop] = 0.0
        lang_mask = lang_mask.clone()
        lang_mask[drop] = True

    B = context_tokens.shape[0]
    t = torch.rand(B, device=context_tokens.device, dtype=context_tokens.dtype)

    m = _unwrap(model)
    action_only = m.action_only
    video_only = m._video_only

    if action_only:
        x_t_video = None
    else:
        x0_video = torch.randn_like(future_tokens)
        x_t_video = (1 - t[:, None, None]) * x0_video + t[:, None, None] * future_tokens

    if video_only:
        x_t_action = None
    else:
        x0_action = torch.randn_like(rel_velocity)
        x_t_action = (1 - t[:, None, None]) * x0_action + t[:, None, None] * rel_velocity

    with accelerator.autocast():
        video_vel, action_vel = model(
            x_t_video=x_t_video,
            x_t_action=x_t_action,
            context_tokens=context_tokens,
            t=t,
            state=state,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
        )

        if video_only:
            action_loss = torch.tensor(0.0, device=context_tokens.device)
        else:
            target_action = rel_velocity - x0_action
            action_loss = _masked_mse(action_vel, target_action, action_valid, name="action")

        if action_only:
            video_loss = torch.tensor(0.0, device=action_loss.device)
            total_loss = action_loss
        else:
            target_video = future_tokens.detach() - x0_video
            video_loss = _masked_mse(video_vel, target_video, future_valid, name="video")
            if video_only:
                total_loss = video_loss
            else:
                total_loss = video_loss + action_weight * action_loss

    return total_loss, {
        "total_loss": total_loss.detach(),
        "video_loss": video_loss.detach(),
        "action_loss": action_loss.detach() * (0.0 if video_only else action_weight),
    }


def diagnose_gradient_conflict(model, raw_batch, accelerator, num_cameras):
    """Two separate forward passes, each consumed by its own backward, so
    gradient_checkpointing (use_reentrant=False) doesn't collide with
    retain_graph. Measures cos sim and norm ratio of per-task gradients.

    Mutates and clears model.grad. Caller must zero/restore grads for training.
    Returns dict or None (action_only / no overlap).
    """
    m = _unwrap(model)
    if m.action_only or m._video_only:
        return None

    context_tokens, future_tokens, rel_velocity, state, lang_tokens, lang_mask, future_valid, action_valid = (
        prepare_batch(model, raw_batch, num_cameras)
    )
    B = context_tokens.shape[0]
    t = torch.rand(B, device=context_tokens.device, dtype=context_tokens.dtype)
    x0_video = torch.randn_like(future_tokens)
    x_t_video = (1 - t[:, None, None]) * x0_video + t[:, None, None] * future_tokens
    x0_action = torch.randn_like(rel_velocity)
    x_t_action = (1 - t[:, None, None]) * x0_action + t[:, None, None] * rel_velocity
    target_action = rel_velocity - x0_action
    target_video = future_tokens.detach() - x0_video

    sync_ctx = model.no_sync if hasattr(model, "no_sync") else nullcontext

    def _one_task_backward(which):
        for p in model.parameters():
            p.grad = None
        with sync_ctx():
            with accelerator.autocast():
                video_vel, action_vel = model(
                    x_t_video=x_t_video, x_t_action=x_t_action,
                    context_tokens=context_tokens, t=t, state=state,
                    lang_tokens=lang_tokens, lang_mask=lang_mask,
                )
                if which == "action":
                    loss = _masked_mse(action_vel, target_action, action_valid, name="action")
                else:
                    loss = _masked_mse(video_vel, target_video, future_valid, name="video")
            loss.backward()
        grads = {
            n: p.grad.detach().clone()
            for n, p in model.named_parameters() if p.grad is not None
        }
        del video_vel, action_vel, loss
        return grads

    action_grads = _one_task_backward("action")
    for p in model.parameters():
        p.grad = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    video_grads = _one_task_backward("video")
    for p in model.parameters():
        p.grad = None

    shared_names = sorted(action_grads.keys() & video_grads.keys())
    if not shared_names:
        return None

    g_a = torch.cat([action_grads[n].flatten().float() for n in shared_names])
    g_v = torch.cat([video_grads[n].flatten().float() for n in shared_names])
    a_norm = g_a.norm().item()
    v_norm = g_v.norm().item()
    cos = F.cosine_similarity(g_a.unsqueeze(0), g_v.unsqueeze(0)).item()

    return {
        "grad_cos": cos,
        "grad_action_norm": a_norm,
        "grad_video_norm": v_norm,
        "grad_ratio_v_a": v_norm / max(a_norm, 1e-12),
        "n_shared_params": len(shared_names),
    }


@torch.no_grad()
def run_validation(model, val_loader, num_batches, accelerator, action_weight):
    """Forward-only pass over ~num_batches from val_loader. Returns averaged losses or None."""
    was_training = _unwrap(model).training
    model.eval()
    try:
        acc = {"total_loss": 0.0, "video_loss": 0.0, "action_loss": 0.0}
        n = 0
        it = iter(val_loader)
        for _ in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(val_loader)
                try:
                    batch = next(it)
                except StopIteration:
                    break
            if batch is None:
                continue
            num_cams = count_cameras(batch)
            _, losses = train_step(
                model, batch, accelerator, num_cams,
                action_weight=action_weight, lang_drop_rate=0.0,
            )
            for k in acc:
                acc[k] += losses[k].item()
            n += 1
    finally:
        if was_training:
            model.train()
    if n == 0:
        return None
    return {k: v / n for k, v in acc.items()}


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*remat_using_tags_for_fwd_loss_bwd_graph.*", category=UserWarning)
    torch.set_float32_matmul_precision("high")
    model_cfg, train_cfg, config_path = load_config()
    accelerator = Accelerator()

    cache_root = os.path.join(le_wam_root, ".cache")

    num_context = model_cfg["num_context_frames"]
    num_future = model_cfg["num_future_frames"]
    scaled_fps = train_cfg["scaled_fps"]
    action_fps = train_cfg["action_fps"]
    crop_size = train_cfg["crop_size"]

    NATIVE_FPS = 30
    assert NATIVE_FPS % scaled_fps == 0, f"native_fps ({NATIVE_FPS}) must be divisible by scaled_fps ({scaled_fps})"
    assert NATIVE_FPS % action_fps == 0, f"native_fps ({NATIVE_FPS}) must be divisible by action_fps ({action_fps})"
    assert action_fps >= scaled_fps, f"action_fps ({action_fps}) must be >= scaled_fps ({scaled_fps})"

    use_lerobot = train_cfg.get("lerobot_repo_id") is not None

    # ── Resolve resume path (main process downloads, others wait) ──────
    resume_path = None
    if train_cfg["resume"]:
        if accelerator.is_main_process:
            resume_path = resolve_checkpoint(
                train_cfg["resume"], cache_root, train_cfg.get("s3_path"),
            )
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            resume_path = os.path.join(cache_root, train_cfg["resume"])

    past_ts = [-(num_context - 1 - i) / scaled_fps for i in range(num_context)]
    future_ts = [(i + 1) / scaled_fps for i in range(num_future)]

    image_tx = transforms.Resize((crop_size, crop_size), antialias=True)

    # ── Dataset (main process downloads, others wait) ─────────────────
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    val_split_seed = int(train_cfg.get("val_split_seed", 42))
    val_num_cameras = int(train_cfg.get("val_num_cameras", 2))
    val_ds = None

    if use_lerobot:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        lerobot_repo_id = train_cfg["lerobot_repo_id"]
        accelerator.print(f"Loading LeRobot dataset: {lerobot_repo_id}")

        if accelerator.is_main_process:
            LeRobotDataset(repo_id=lerobot_repo_id, revision="main", force_cache_sync=True)
        accelerator.wait_for_everyone()

        _meta_ds = LeRobotDataset(repo_id=lerobot_repo_id, revision="main")
        camera_keys = sorted(k for k in _meta_ds.meta.features if k.startswith("observation.images."))
        num_cameras = len(camera_keys)
        action_horizon = int(num_future / scaled_fps * action_fps)
        action_ts = [i / action_fps for i in range(action_horizon + 1)]

        delta_timestamps = {k: past_ts + future_ts for k in camera_keys}
        delta_timestamps["observation.state"] = [0.0]
        delta_timestamps["action"] = action_ts

        ds_kwargs = dict(
            repo_id=lerobot_repo_id,
            revision="main",
            delta_timestamps=delta_timestamps,
            image_transforms=image_tx,
            video_backend="torchcodec",
        )

        if val_fraction > 0:
            train_eps, val_eps = _split_episodes(range(_meta_ds.num_episodes), val_fraction, val_split_seed)
            accelerator.print(f"  episode split: {len(train_eps)} train, {len(val_eps)} val (seed={val_split_seed})")
            ds = LeRobotDataset(episodes=train_eps, **ds_kwargs)
            val_ds = LeRobotDataset(episodes=val_eps, **ds_kwargs)
        else:
            ds = LeRobotDataset(**ds_kwargs)
        del _meta_ds

        accelerator.print(f"  cameras={camera_keys}  episodes={ds.num_episodes}  frames={ds.num_frames}")
        if val_ds is not None:
            accelerator.print(f"  val: episodes={val_ds.num_episodes}  frames={val_ds.num_frames}")
    else:
        from lewam.datasets.community_dataset import CommunityDataset

        dataset_suffix = "_small" if train_cfg["small_dataset"] else "_preprocessed"
        repo_id = f"ehalicki/LeWAM_community_dataset{dataset_suffix}"

        action_ts = [i / action_fps for i in range(int(num_future / scaled_fps * action_fps) + 1)]
        delta_ts = {
            "observation.images.image": past_ts + future_ts,
            "observation.state": [0.0],
            "action": action_ts,
        }
        cd_kwargs = dict(delta_timestamps=delta_ts, image_transforms=image_tx)

        if accelerator.is_main_process:
            _cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)
            _cd.load_metadata()
            if val_fraction > 0:
                train_eps, val_eps = _cd.split_episodes(val_fraction, val_split_seed)
                _cd.prefetch_metadata(episodes=train_eps, **cd_kwargs)
                _cd.build_val_dataset(val_eps, target_num_cameras=val_num_cameras, **cd_kwargs)
            else:
                _cd.prefetch_metadata(**cd_kwargs)
            del _cd
        accelerator.wait_for_everyone()

        cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)
        cd.load_metadata()
        if val_fraction > 0:
            train_eps, val_eps = cd.split_episodes(val_fraction, val_split_seed)
            accelerator.print(f"  episode split (seed={val_split_seed}): {sum(len(v) for v in train_eps.values())} train, {sum(len(v) for v in val_eps.values())} val")
            cd.prefetch_metadata(episodes=train_eps, **cd_kwargs)
            val_ds = cd.build_val_dataset(val_eps, target_num_cameras=val_num_cameras, **cd_kwargs)
        else:
            cd.prefetch_metadata(**cd_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    latent_frame_side = crop_size // PATCH_SIZE
    start_step = 0
    _optimizer_state = None

    if resume_path:
        accelerator.print(f"Loading model from checkpoint: {resume_path}")
        _ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model = LeWAM.from_checkpoint(
            _ckpt, fps=scaled_fps, action_fps=action_fps,
            frame_latent_h=latent_frame_side, frame_latent_w=latent_frame_side,
            num_context_frames=num_context, num_future_frames=num_future,
        )
        if not train_cfg.get("restart"):
            start_step = _ckpt.get("step", 0)
            _optimizer_state = _ckpt.get("optimizer")
        norm_stats = _ckpt["norm_stats"]
        del _ckpt
    else:
        if accelerator.is_main_process:
            if use_lerobot:
                norm_stats = compute_norm_stats_lerobot(ds, action_fps)
            else:
                norm_stats = compute_norm_stats_community(repo_id, cache_root, action_fps)
        else:
            norm_stats = None
        norm_stats = broadcast_object_list([norm_stats], from_process=0)[0]

        accelerator.print(f"Building LeWAM from {config_path}...")
        model = LeWAM(
            **model_cfg,
            frame_latent_h=latent_frame_side,
            frame_latent_w=latent_frame_side,
            fps=scaled_fps,
            action_fps=action_fps,
            norm_stats=norm_stats,
        )

        vjepa2_path = os.path.join(cache_root, "vjepa2_1_vitb_dist_vitG_384.pt")
        if not os.path.exists(vjepa2_path):
            raise FileNotFoundError(
                f"VJEPA2 weights not found at {vjepa2_path}. "
                "Training from a random VJEPA2 init is not allowed."
            )
        accelerator.print("Loading VJEPA2 weights...")
        model.load_vjepa2_weights(vjepa2_path)

    _video_only_cfg = bool(train_cfg.get("video_only", False))
    if _video_only_cfg and model.action_only:
        raise ValueError(
            "Cannot enable both video_only (training flag) and action_only (model flag). "
            "action_only removes the video head from the architecture, so there is nothing "
            "left to train with video_only."
        )
    if _video_only_cfg:
        model.set_video_only_mode(True)
        accelerator.print(
            "VIDEO-ONLY MODE: action tokens dropped from sequence; "
            "action_encoder/action_final frozen."
        )

    accelerator.print(f"Trainable params: {model.count_params()}M | Total params: {model.count_params(trainable_only=False)}M")

    if train_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing = True
        accelerator.print("Gradient checkpointing enabled")

    if accelerator.is_main_process:
        LeWAM.visualize_attn_mask(
            model.num_context_tubelets,
            0 if model.action_only else model.num_future_tubelets,
            0 if model._video_only else model.action_horizon,
            block_size=model.attn_block_size,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg["lr"], betas=(0.9, 0.95), eps=1e-7)

    # ── Batch size calibration (raw model, no compile, no DDP) ───────────
    _CALIB_TASK = (
        "Pick up the red cube and place it in the blue bin, then pick up the sock "
        "and fold it on the table, then pick up the glass and move it to the shelf."
    )
    model = model.to(accelerator.device)
    _model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if use_lerobot:
        if train_cfg["force_batch_size"] is not None:
            batch_size = int(train_cfg["force_batch_size"])
            accelerator.print(f"Using forced batch size: {batch_size}")
        else:
            accelerator.print(f"Calibrating batch size for {num_cameras}-camera dataset...")
            _calib_loader = accelerator.prepare(DataLoader(ds, batch_size=1, num_workers=0))
            _sample = next(iter(_calib_loader))

            def _try_batch(bs, _s=_sample, _nc=num_cameras):
                test_batch = {
                    k: v[:1].repeat(bs, *([1] * (v.ndim - 1))) if hasattr(v, "ndim") else [v[0]] * bs
                    for k, v in _s.items()
                }
                test_batch["task"] = [_CALIB_TASK] * bs
                optimizer.zero_grad()
                total_loss, _ = train_step(
                    model, test_batch, accelerator, _nc, action_weight=train_cfg["action_weight"],
                )
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            batch_size = find_max_batch_size(
                _try_batch, target_fraction=train_cfg["vram_target_fraction"],
                device_idx=accelerator.local_process_index,
            )
            optimizer.zero_grad()

        accelerator.print(f"  batch_size={batch_size}")
        batch_sizes = {num_cameras: batch_size}
    else:
        batch_sizes: dict[int, int] = {}

        if train_cfg["force_batch_size"] is not None:
            forced = int(train_cfg["force_batch_size"])
            for n_cams in cd.datasets:
                batch_sizes[n_cams] = forced
            accelerator.print(f"Using forced batch size: {forced}")
        else:
            for n_cams in sorted(cd.datasets):
                accelerator.print(f"Calibrating batch size for {n_cams}-camera datasets...")
                _calib_loader = accelerator.prepare(DataLoader(cd.datasets[n_cams], batch_size=1, num_workers=0))
                _sample = next(iter(_calib_loader))

                def _try_batch(bs, _s=_sample, _nc=n_cams):
                    test_batch = {
                        k: v[:1].repeat(bs, *([1] * (v.ndim - 1))) if hasattr(v, "ndim") else [v[0]] * bs
                        for k, v in _s.items()
                    }
                    test_batch["task"] = [_CALIB_TASK] * bs
                    optimizer.zero_grad()
                    total_loss, _ = train_step(
                        model, test_batch, accelerator, _nc, action_weight=train_cfg["action_weight"],
                    )
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                batch_sizes[n_cams] = find_max_batch_size(
                    _try_batch, target_fraction=train_cfg["vram_target_fraction"],
                    device_idx=accelerator.local_process_index,
                )
            optimizer.zero_grad()

    # Reset model + optimizer state after calibration
    model.load_state_dict(_model_state)
    del _model_state
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg["lr"], betas=(0.9, 0.95), eps=1e-7)
    if _optimizer_state is not None:
        saved_sizes = [len(g["params"]) for g in _optimizer_state["param_groups"]]
        current_sizes = [len(g["params"]) for g in optimizer.param_groups]
        if saved_sizes != current_sizes:
            accelerator.print(
                f"Skipping optimizer state load: trainable param count changed "
                f"(saved={saved_sizes}, current={current_sizes}). "
                f"Likely a video_only ↔ joint mode switch. Starting optimizer from scratch."
            )
        else:
            optimizer.load_state_dict(_optimizer_state)
            for g in optimizer.param_groups:
                g.pop("initial_lr", None)
        _optimizer_state = None

    # ── Compile + Accelerate prepare ─────────────────────────────────────
    if not train_cfg.get("no_compile"):
        model.forward = torch.compile(model.forward, dynamic=True)
    model, optimizer = accelerator.prepare(model, optimizer)

    _bs_list = [batch_sizes]
    broadcast_object_list(_bs_list, from_process=0)
    batch_sizes = _bs_list[0]

    NUM_GPUS = accelerator.num_processes
    SAMPLES_PER_STEP = train_cfg["effective_batch_size"]
    LR_SCALE = SAMPLES_PER_STEP / train_cfg["base_batch_size"]
    for g in optimizer.param_groups:
        g["lr"] = train_cfg["lr"] * LR_SCALE
    for n_cams, bs in sorted(batch_sizes.items()):
        accelerator.print(f"  {n_cams}-cam batch_size={bs}")
    accelerator.print(
        f"Target samples per step: {SAMPLES_PER_STEP} "
        f"(lr_scale={LR_SCALE:.2f}x)"
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=train_cfg["warmup_steps"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["steps"] - train_cfg["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[train_cfg["warmup_steps"]])

    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()

    # ── Dataloaders ───────────────────────────────────────────────────────
    def _prep_loader(dl):
        dl = accelerator.prepare(dl)
        # Workaround for accelerate bug: persistent_workers + re-iter triggers
        # synchronize_rng_states with a corrupted mt19937 state. Disable RNG sync.
        if hasattr(dl, "rng_types"):
            dl.rng_types = []
        return dl

    if use_lerobot:
        def _build_data_iter():
            common_kwargs = dict(
                num_workers=train_cfg["num_workers"],
                prefetch_factor=2 if train_cfg["num_workers"] > 0 else None,
                persistent_workers=train_cfg["num_workers"] > 0,
                collate_fn=_collate_skip_none,
                pin_memory=True,
                shuffle=True,
                multiprocessing_context="forkserver" if train_cfg["num_workers"] > 0 else None,
            )
            loader = _prep_loader(
                DataLoader(_SafeDataset(ds), batch_size=batch_sizes[num_cameras], **common_kwargs)
            )
            while True:
                for batch in loader:
                    if batch is not None:
                        yield batch

        data_iter = _build_data_iter()
    else:
        def _build_data_iter():
            common_kwargs = dict(
                num_workers=train_cfg["num_workers"],
                prefetch_factor=2 if train_cfg["num_workers"] > 0 else None,
                persistent_workers=train_cfg["num_workers"] > 0,
                collate_fn=_collate_skip_none,
                pin_memory=True,
                shuffle=True,
                multiprocessing_context="forkserver" if train_cfg["num_workers"] > 0 else None,
            )
            loaders = {
                n: _prep_loader(DataLoader(_SafeDataset(ds), batch_size=batch_sizes[n], **common_kwargs))
                for n, ds in cd.datasets.items()
            }
            return _infinite_interleaved(loaders, accelerator)

        data_iter = _build_data_iter()

    val_loader = None
    if val_ds is not None:
        _val_num_cams = num_cameras if use_lerobot else val_num_cameras
        _val_bs = batch_sizes.get(_val_num_cams, min(batch_sizes.values()))
        val_loader = accelerator.prepare(
            DataLoader(
                _SafeDataset(val_ds),
                batch_size=_val_bs,
                num_workers=0,
                collate_fn=_collate_skip_none,
                pin_memory=True,
                shuffle=True,
            )
        )

    # ── Output dir ────────────────────────────────────────────────────────
    run_tag = train_cfg.get("run_tag") or f"lewam-{accelerator.unwrap_model(model).count_params()}M"
    run_dir = os.path.join(cache_root, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    losses_path = os.path.join(run_dir, "losses.json")
    loss_log = []
    if accelerator.is_main_process and start_step > 0 and os.path.exists(losses_path):
        with open(losses_path) as f:
            loss_log = json.load(f)
        if loss_log:
            accelerator.print(f"Resumed loss log with {len(loss_log)} entries (up to step {loss_log[-1]['step']})")

    if accelerator.is_main_process:
        config_snapshot = {"model": model_cfg, "train": train_cfg}
        config_path_local = os.path.join(run_dir, "config.json")
        with open(config_path_local, "w") as f:
            json.dump(config_snapshot, f, indent=2)
        s3_path = train_cfg["s3_path"]
        if s3_path and aws_available():
            s3_prefix = f"{s3_path}/{run_tag}"
            upload_to_s3_async(config_path_local, f"{s3_prefix}/config.json")

    def save_checkpoint(step):
        if step == start_step:
            return
        accelerator.wait_for_everyone()
        path = os.path.join(run_dir, f"{run_tag}_latest.pt")
        accelerator.save({
            "step": step,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": accelerator.unwrap_model(model).config,
            "train_config": train_cfg,
            "norm_stats": norm_stats,
        }, path)
        if accelerator.is_main_process:
            with open(losses_path, "w") as f:
                json.dump(loss_log, f)
            accelerator.print(f"Checkpoint: {path}")
            s3_path = train_cfg["s3_path"]
            if s3_path and aws_available():
                s3_prefix = f"{s3_path}/{run_tag}"
                s3_ckpt = f"{s3_prefix}/{run_tag}_latest.pt"
                copy_s3(s3_ckpt, f"{s3_ckpt}.bak", blocking=True)
                upload_to_s3_async(path, s3_ckpt)
                upload_to_s3_async(losses_path, f"{s3_prefix}/losses.json")
                backup_every = train_cfg.get("backup_every", 500)
                if backup_every and step % backup_every == 0:
                    copy_s3(s3_ckpt, f"{s3_prefix}/backups/{run_tag}_step{step}.pt")

    accelerator.wait_for_everyone()

    # ── Training loop ─────────────────────────────────────────────────────
    steps = train_cfg["steps"]
    save_every = train_cfg["save_every"]
    action_weight = train_cfg["action_weight"]
    lang_drop_rate = train_cfg["lang_drop_rate"]

    step = start_step
    t0 = time.time()
    _t0_step = start_step
    _loss_acc = {k: 0.0 for k in ("total_loss", "video_loss", "action_loss")}
    _micro_count = 0
    _samples_acc = 0
    _data_secs_acc = 0.0
    samples_per_gpu = SAMPLES_PER_STEP // NUM_GPUS

    accelerator.print(f"Training {run_tag} for {steps} steps...")
    optimizer.zero_grad()

    if train_cfg["overfit_test"]:
        raw = next(data_iter)

    last_saved_step = None
    while step < steps:
        if not train_cfg["overfit_test"]:
            _t_data = time.time()
            raw = next(data_iter)
            _data_secs_acc += time.time() - _t_data

        num_cams = count_cameras(raw)
        _samples_acc += batch_sizes[num_cams]
        _micro_count += 1
        do_update = _samples_acc >= samples_per_gpu
        sync_context = model.no_sync if not do_update and hasattr(model, "no_sync") else nullcontext
        with sync_context():
            total_loss, losses = train_step(
                model, raw, accelerator, num_cams,
                action_weight=action_weight, lang_drop_rate=lang_drop_rate,
            )
            accelerator.backward(total_loss)

        for k in _loss_acc:
            _loss_acc[k] += losses[k].item()

        if not do_update:
            if accelerator.is_main_process:
                print(f"\r  micro {_samples_acc}/{samples_per_gpu} samples", end="", flush=True)
            continue

        for p in trainable:
            if p.grad is not None:
                p.grad /= _micro_count
        _norm = accelerator.clip_grad_norm_(trainable, 1.0)
        global_grad_norm = _norm.item() if _norm is not None else float("nan")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        entry = {k: _loss_acc[k] / _micro_count for k in _loss_acc} | {"step": step}
        loss_log.append(entry)
        _data_secs_step = _data_secs_acc
        _loss_acc = {k: 0.0 for k in _loss_acc}
        _micro_count = 0
        _samples_acc = 0
        _data_secs_acc = 0.0

        steps_done = step - _t0_step
        secs_per_step = (time.time() - t0) / max(steps_done, 1)
        eta_secs = secs_per_step * (steps - step)
        eta_delta = str(datetime.timedelta(seconds=int(eta_secs)))
        tz = ZoneInfo(train_cfg["timezone"])
        finish_wall = datetime.datetime.now(tz) + datetime.timedelta(seconds=eta_secs)
        finish_str = finish_wall.strftime("%Y-%m-%d %H:%M %Z")
        current_lr = optimizer.param_groups[0]["lr"]
        accelerator.print(
            f"step {step}/{steps}  "
            f"total={entry['total_loss']:.4f}  "
            f"video={entry['video_loss']:.4f}  "
            f"action={entry['action_loss']:.4f}  "
            f"lr={current_lr:.2e}  "
            f"grad_norm={global_grad_norm:.2f}  "
            f"sec_per_step={secs_per_step:.2f}s  "
            f"data_secs={_data_secs_step:.2f}s  "
            f"eta={eta_delta} ({finish_str})"
        )

        if step % save_every == 0 or step == steps or step == 5:
            if val_loader is not None:
                val_metrics = run_validation(
                    model, val_loader,
                    num_batches=int(train_cfg.get("val_batches", 5)),
                    accelerator=accelerator,
                    action_weight=action_weight,
                )
                if val_metrics is not None:
                    entry["val_total_loss"] = val_metrics["total_loss"]
                    entry["val_video_loss"] = val_metrics["video_loss"]
                    entry["val_action_loss"] = val_metrics["action_loss"]
                    accelerator.print(
                        f"  val: total={val_metrics['total_loss']:.4f}  "
                        f"video={val_metrics['video_loss']:.4f}  "
                        f"action={val_metrics['action_loss']:.4f}"
                    )

            try:
                grad_metrics = diagnose_gradient_conflict(model, raw, accelerator, num_cams)
            except torch.cuda.OutOfMemoryError as e:
                accelerator.print(f"  grad diagnostic OOM, skipping: {e}")
                grad_metrics = None
                for p in model.parameters():
                    p.grad = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if grad_metrics is not None:
                entry["grad_cos"] = grad_metrics["grad_cos"]
                entry["grad_action_norm"] = grad_metrics["grad_action_norm"]
                entry["grad_video_norm"] = grad_metrics["grad_video_norm"]
                entry["grad_ratio_v_a"] = grad_metrics["grad_ratio_v_a"]
                accelerator.print(
                    f"  grad: cos={grad_metrics['grad_cos']:+.3f}  "
                    f"||a||={grad_metrics['grad_action_norm']:.3e}  "
                    f"||v||={grad_metrics['grad_video_norm']:.3e}  "
                    f"v/a={grad_metrics['grad_ratio_v_a']:.2f}"
                )
            optimizer.zero_grad()

            save_checkpoint(step)
            last_saved_step = step
            t0 = time.time()
            _t0_step = step
            if accelerator.is_main_process:
                s3_path = train_cfg["s3_path"]
                s3_prefix = f"{s3_path}/{run_tag}" if s3_path and aws_available() else None
                ode_path = save_ode_viz(
                    model=accelerator.unwrap_model(model),
                    raw_batch=raw,
                    run_dir=run_dir,
                    step=step,
                )
                if s3_prefix:
                    upload_to_s3_async(ode_path, f"{s3_prefix}/ode_viz/ode-step{step}.png")
            accelerator.wait_for_everyone()

    if last_saved_step != step:
        save_checkpoint(step)
    if accelerator.is_main_process:
        accelerator.print("Waiting for S3 uploads...")
        wait_for_s3_uploads()
    accelerator.print("Done.")


if __name__ == "__main__":
    main()
