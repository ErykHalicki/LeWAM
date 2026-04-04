"""
LeWAM training script (Accelerate).

Usage:
    accelerate launch src/wam/training/scripts/train.py --config configs/train/default.yaml
    accelerate launch src/wam/training/scripts/train.py --config configs/train/overfit.yaml --steps 1000

CLI flags override values from the YAML config.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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

from wam.datasets.community_dataset import CommunityDataset
from wam.models.lewam import LeWAM
from wam.training.scripts.precompute_norm_stats import precompute_norm_stats
from wam.training.common import (
    aws_available, backup_s3, find_max_batch_size, save_ode_viz, upload_to_s3_async,
)

PATCH_SIZE = LeWAM.VJEPA_PATCH_SIZE
TUBELET = LeWAM.VJEPA_TUBELET_SIZE

le_wam_root = os.environ.get("LE_WAM_ROOT")
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")

# ── Data helpers ─────────────────────────────────────────────────────────────

def get_camera_frames(batch: dict, num_frames: int) -> torch.Tensor:
    """
    Collect all camera streams into a (B, N, T, C, H, W) tensor.
    Cameras are ordered by key name.
    """
    cam_keys = sorted(
        k for k in batch
        if k.startswith("observation.images.image") and not k.endswith("_is_pad")
    )
    return torch.stack([batch[k][:, :num_frames] for k in cam_keys], dim=1)


def count_cameras(batch: dict) -> int:
    return sum(
        1 for k in batch
        if k.startswith("observation.images.image") and not k.endswith("_is_pad")
    )


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


def prepare_batch(model, raw_batch, num_cameras):
    """
    Collect camera frames, encode video, normalize actions/state.
    Returns all tensors needed for the flow matching forward pass.
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
        m.set_patch_grid(frame_latent_h, frame_latent_w)

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

    return context_tokens, future_tokens, rel_velocity, state, lang_tokens, lang_mask


def train_step(model, raw_batch, accelerator, num_cameras, action_weight=0.0, lang_drop_rate=0.0):
    context_tokens, future_tokens, rel_velocity, state, lang_tokens, lang_mask = (
        prepare_batch(model, raw_batch, num_cameras)
    )

    if lang_tokens is not None and lang_drop_rate > 0.0:
        B = context_tokens.shape[0]
        drop = torch.rand(B, device=lang_tokens.device) < lang_drop_rate
        lang_tokens = lang_tokens.clone()
        lang_tokens[drop] = 0.0
        lang_mask = lang_mask.clone()
        lang_mask[drop] = True

    B = context_tokens.shape[0]
    t = torch.rand(B, device=context_tokens.device, dtype=context_tokens.dtype)
    #diffusion time step sampled form uniform dist

    x0_video = torch.randn_like(future_tokens)
    x_t_video = (1 - t[:, None, None]) * x0_video + t[:, None, None] * future_tokens

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

        target_video = future_tokens.detach() - x0_video
        target_action = rel_velocity - x0_action

        video_loss = F.mse_loss(video_vel, target_video)
        action_loss = F.mse_loss(action_vel, target_action)
        total_loss = video_loss + action_weight * action_loss

    return total_loss, {
        "total_loss": total_loss.detach(),
        "video_loss": video_loss.detach(),
        "action_loss": action_loss.detach(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    model_cfg, train_cfg, config_path = load_config()
    accelerator = Accelerator()

    cache_root = os.path.join(le_wam_root, ".cache")
    weights_dir = os.path.join(le_wam_root, "weights")

    dataset_suffix = "_small" if train_cfg["small_dataset"] else ""
    repo_id = f"ehalicki/LeWAM_community_dataset{dataset_suffix}"

    num_context = model_cfg["num_context_frames"]
    num_future = model_cfg["num_future_frames"]
    scaled_fps = train_cfg["scaled_fps"]
    action_fps = train_cfg["action_fps"]
    crop_size = train_cfg["crop_size"]

    NATIVE_FPS = 30
    assert NATIVE_FPS % scaled_fps == 0, f"native_fps ({NATIVE_FPS}) must be divisible by scaled_fps ({scaled_fps})"
    assert NATIVE_FPS % action_fps == 0, f"native_fps ({NATIVE_FPS}) must be divisible by action_fps ({action_fps})"
    assert action_fps >= scaled_fps, f"action_fps ({action_fps}) must be >= scaled_fps ({scaled_fps})"

    norm_stats_path = os.path.join(cache_root, repo_id, "norm_stats.pt")
    if accelerator.is_main_process:
        accelerator.print("Precomputing norm stats...")
        precompute_norm_stats(repo_id=repo_id, cache_root=cache_root, action_fps=action_fps)
    accelerator.wait_for_everyone()

    patch_side = crop_size // PATCH_SIZE

    accelerator.print(f"Building LeWAM from {config_path}...")
    model = LeWAM(
        **model_cfg,
        frame_latent_h=patch_side,
        frame_latent_w=patch_side,
        fps=scaled_fps,
        action_fps=action_fps,
        stats_path=norm_stats_path if os.path.exists(norm_stats_path) else None,
    )

    accelerator.print(f"Trainable params: {model.count_params()}M | Total params: {model.count_params(trainable_only=False)}M")

    if accelerator.is_main_process:
        LeWAM.visualize_attn_mask(model.num_context_tubelets, model.num_future_tubelets, model.action_horizon)

    vjepa2_path = os.path.join(weights_dir, "vjepa2_1_vitb_dist_vitG_384.pt")
    if os.path.exists(vjepa2_path):
        accelerator.print("Loading VJEPA2 weights...")
        model.load_vjepa2_weights(vjepa2_path)
    else:
        accelerator.print("VJEPA2 weights not found, using random init")

    past_ts = [-(num_context - 1 - i) / scaled_fps for i in range(num_context)]
    future_ts = [(i + 1) / scaled_fps for i in range(num_future)]
    action_ts = [i / action_fps for i in range(model.action_horizon + 1)]

    cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)
    cd.prefetch_metadata(
        delta_timestamps={
            "observation.images.image": past_ts + future_ts,
            "observation.state": [0.0],
            "action": action_ts,
        },
        image_transforms=transforms.Compose([
            transforms.Resize(crop_size, antialias=True),
            transforms.CenterCrop(crop_size),
        ]),
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg["lr"], eps=1e-7)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    _ckpt = None
    if train_cfg["resume"]:
        accelerator.print(f"Resuming from {train_cfg['resume']}")
        _ckpt = torch.load(train_cfg["resume"], map_location="cpu", weights_only=False)
        start_step = _ckpt["step"]

    # ── Accelerate prepare ────────────────────────────────────────────────
    model, optimizer = accelerator.prepare(model, optimizer)

    # ── Batch size (per camera count) ────────────────────────────────────
    _CALIB_TASK = (
        "Pick up the red cube and place it in the blue bin, then pick up the sock "
        "and fold it on the table, then pick up the glass and move it to the shelf."
    )
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

            def _try_batch(batch_size, _s=_sample, _nc=n_cams):
                test_batch = {
                    k: v[:1].repeat(batch_size, *([1] * (v.ndim - 1))) if hasattr(v, "ndim") else [v[0]] * batch_size
                    for k, v in _s.items()
                }
                test_batch["task"] = [_CALIB_TASK] * batch_size
                optimizer.zero_grad()
                total_loss, _ = train_step(
                    model, test_batch, accelerator, _nc, action_weight=train_cfg["action_weight"],
                )
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

            batch_sizes[n_cams] = find_max_batch_size(
                _try_batch, target_fraction=0.925, device_idx=accelerator.local_process_index,
            )
        optimizer.zero_grad()

    if _ckpt is not None:
        accelerator.unwrap_model(model).load_state_dict(_ckpt["model"])
        optimizer.load_state_dict(_ckpt["optimizer"])
        for g in optimizer.param_groups:
            g.pop("initial_lr", None)
        _ckpt = None

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
    scheduler = accelerator.prepare(scheduler)

    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()

    # ── Dataloaders ───────────────────────────────────────────────────────
    def _build_data_iter():
        common_kwargs = dict(
            num_workers=train_cfg["num_workers"],
            prefetch_factor=2 if train_cfg["num_workers"] > 0 else None,
            persistent_workers=train_cfg["num_workers"] > 0,
            collate_fn=_collate_skip_none,
            pin_memory=True,
            shuffle=True,
        )
        loaders = {
            n: accelerator.prepare(DataLoader(_SafeDataset(ds), batch_size=batch_sizes[n], **common_kwargs))
            for n, ds in cd.datasets.items()
        }
        return _infinite_interleaved(loaders, accelerator)

    data_iter = _build_data_iter()

    # ── Output dir ────────────────────────────────────────────────────────
    run_tag = train_cfg.get("run_tag") or f"lewam-{accelerator.unwrap_model(model).count_params()}M"
    run_dir = os.path.join(le_wam_root, "runs", run_tag)
    os.makedirs(run_dir, exist_ok=True)
    losses_path = os.path.join(run_dir, "losses.json")
    loss_log = []
    if start_step > 0 and os.path.exists(losses_path):
        with open(losses_path) as f:
            loss_log = json.load(f)
        accelerator.print(f"Resumed loss log with {len(loss_log)} entries (up to step {loss_log[-1]['step']})")

    if accelerator.is_main_process:
        config_snapshot = {"model": model_cfg, "train": train_cfg}
        config_path_local = os.path.join(run_dir, "config.json")
        with open(config_path_local, "w") as f:
            json.dump(config_snapshot, f, indent=2)
        s3_path = train_cfg["s3_path"]
        if s3_path and aws_available():
            s3_prefix = f"{s3_path}/{run_tag}"
            upload_to_s3_async(norm_stats_path, f"{s3_prefix}/norm_stats.pt")
            upload_to_s3_async(config_path_local, f"{s3_prefix}/config.json")

    def save_checkpoint(step):
        accelerator.wait_for_everyone()
        path = os.path.join(run_dir, f"{run_tag}_latest.pt")
        accelerator.save({
            "step": step,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": accelerator.unwrap_model(model).config,
            "train_config": train_cfg,
        }, path)
        if accelerator.is_main_process:
            with open(losses_path, "w") as f:
                json.dump(loss_log, f)
            accelerator.print(f"Checkpoint: {path}")
            s3_path = train_cfg["s3_path"]
            if s3_path and aws_available():
                s3_prefix = f"{s3_path}/{run_tag}"
                s3_ckpt = f"{s3_prefix}/{run_tag}_latest.pt"
                backup_s3(s3_ckpt)
                upload_to_s3_async(path, s3_ckpt)
                upload_to_s3_async(losses_path, f"{s3_prefix}/losses.json")

    accelerator.wait_for_everyone()

    # ── Training loop ─────────────────────────────────────────────────────
    steps = train_cfg["steps"]
    save_every = train_cfg["save_every"]
    action_weight = train_cfg["action_weight"]
    lang_drop_rate = train_cfg["lang_drop_rate"]

    step = start_step
    t0 = time.time()
    _loss_acc = {k: 0.0 for k in ("total_loss", "video_loss", "action_loss")}
    _micro_count = 0
    _samples_acc = 0
    samples_per_gpu = SAMPLES_PER_STEP // NUM_GPUS

    accelerator.print(f"Training {run_tag} for {steps} steps...")
    optimizer.zero_grad()

    if train_cfg["overfit_test"]:
        raw = next(data_iter)

    while step < steps:
        if not train_cfg["overfit_test"]:
            raw = next(data_iter)

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
        _loss_acc = {k: 0.0 for k in _loss_acc}
        _micro_count = 0
        _samples_acc = 0

        steps_done = step - start_step
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
            f"eta={eta_delta} ({finish_str})"
        )

        if step % save_every == 0 or step == steps or step == 5:
            save_checkpoint(step)
            if accelerator.is_main_process:
                os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
                s3_path = train_cfg["s3_path"]
                s3_prefix = f"{s3_path}/{run_tag}" if s3_path and aws_available() else None
                ode_path = save_ode_viz(
                    model=accelerator.unwrap_model(model),
                    raw_batch=raw,
                    run_dir=run_dir,
                    step=step,
                )
                if s3_prefix:
                    upload_to_s3_async(ode_path, f"{s3_prefix}/ode-step{step}.png")
            accelerator.wait_for_everyone()

    save_checkpoint(step)
    accelerator.print("Done.")


if __name__ == "__main__":
    main()
