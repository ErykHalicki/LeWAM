import sys
import multiprocessing

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)

import random
import os
import json
import time
import datetime
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*sdp_kernel.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*timm.models.layers.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*lr_scheduler.step().*')

import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision.transforms import v2 as transforms
from zoneinfo import ZoneInfo

from wam.datasets.community_dataset import CommunityDataset
from wam.scripts.precompute_task_embeddings import precompute_task_embeddings
from wam.scripts.precompute_norm_stats import precompute_norm_stats
from wam.models.encoders import ActionPreprocessor
from wam.models.lewam import build_lewam_with_encoders, VERSION as LEWAM_VERSION
from wam.training.common import (
    save_ode_viz, lookup_language_embeddings, resolve_checkpoint,
    aws_available, upload_to_s3_async, find_max_batch_size,
)
from wam.training.losses import teacher_forcing_loss, end_to_end_loss  # noqa: F401 — end_to_end_loss reserved for future non-teacher-forcing mode

# NOTE: a separate script is needed to precompute language embeddings for all unique task strings
# in the community dataset and save to task_embeddings.pt (same format as ssv2 label_embeddings.pt)


parser = argparse.ArgumentParser()
parser.add_argument('--identifier', type=str, default='community_pretrain')
parser.add_argument('--effective-batch-size', type=int, default=64)
parser.add_argument('--base-lr', type=float, default=5e-4)
parser.add_argument('--base-batch-size', type=int, default=64)
parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16')
parser.add_argument('--warmup-steps', type=int, default=200)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--resume-from-aws', action='store_true')
parser.add_argument('--save-interval-steps', type=int, default=200)
parser.add_argument('--save-interval-seconds', type=int, default=600)
parser.add_argument('--s3-path', type=str, default='s3://zima-data/lewam/checkpoints')
parser.add_argument('--steps', type=int, default=8000)
parser.add_argument('--timezone', type=str, default='Europe/Zurich')
parser.add_argument('--crop-size', type=int, default=224)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--dit-size', type=str, default='S')
parser.add_argument('--idm-size', type=str, default='S')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--idm-lr-mult', type=float, default=5.0)
parser.add_argument('--small', action='store_true', help='Use the _small dataset variant for overfitting tests')
parser.add_argument('--idm-only', action='store_true', help='Stage 1: IDM + unfrozen VJEPA2 encoder only, no DiT')
parser.add_argument('--sanity-check', action='store_true', help='Will it work? maybe yes maybe no')
args = parser.parse_args()


FPS                 = 30
NUM_PAST            = 6
NUM_FUTURE          = 10
VJEPA2_TUBELET_SIZE = 2
PATCH_SIZE          = 16
CROP_SIZE           = args.crop_size

PATCH_H      = PATCH_W = CROP_SIZE // PATCH_SIZE
PAST_TOKENS   = (NUM_PAST // VJEPA2_TUBELET_SIZE) * PATCH_H * PATCH_W
FUTURE_TOKENS = (NUM_FUTURE // VJEPA2_TUBELET_SIZE) * PATCH_H * PATCH_W
VJEPA2_DIM    = 768  # ViT-B output dim
# CHUNK_LEN = NUM_FUTURE // TUBELET_SIZE + 1: extra frame allows relative delta for the last action latent
CHUNK_LEN    = NUM_FUTURE // VJEPA2_TUBELET_SIZE + 1

LAMBDA_IDM_LOSS = 1.0
STEPS           = args.steps

device    = args.device
_is_cuda  = device == 'cuda'
_bf16_ok  = (torch.cuda.get_device_capability()[0] >= 8) if _is_cuda else False
AMP_DTYPE = torch.bfloat16 if _bf16_ok else torch.float16
scaler    = GradScaler(enabled=_is_cuda and not _bf16_ok)
amp_ctx   = autocast(device_type='cuda', dtype=AMP_DTYPE) if _is_cuda else nullcontext()
if _is_cuda:
    print(f"AMP dtype: {AMP_DTYPE} ({'bfloat16 natively supported' if _bf16_ok else 'float16 + GradScaler'})")
else:
    print(f"Device: {device} — autocast disabled, running float16")

le_wam_root = os.environ.get('LE_WAM_ROOT')
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")
cache_root  = os.path.join(le_wam_root, '.cache')
weights_dir = os.path.join(le_wam_root, 'weights')


IDM_NUM_PAST_FRAMES   = 1
IDM_NUM_FUTURE_FRAMES = 1

model = build_lewam_with_encoders(
    vjepa2_checkpoint=os.path.join(weights_dir, 'vjepa2_1_vitb_dist_vitG_384.pt'),
    load_language_encoder=False,
    crop_size=CROP_SIZE,
    fps=float(NUM_PAST // VJEPA2_TUBELET_SIZE),
    num_past_frames=NUM_PAST // VJEPA2_TUBELET_SIZE,
    num_future_frames=NUM_FUTURE // VJEPA2_TUBELET_SIZE,
    idm_num_past_frames=IDM_NUM_PAST_FRAMES,
    idm_num_future_frames=IDM_NUM_FUTURE_FRAMES,
    raw_state_dim=6,
    action_dim=6,
    dit_size=None if args.idm_only else args.dit_size,
    idm_size=args.idm_size,
)
if args.idm_only:
    model.video_encoder.set_frozen(False)
model.train()
model = model.to(device)
if not _is_cuda:
    model = model.to(dtype=AMP_DTYPE)


_dit_tag = f"DiT-{args.dit_size}" if model.dit is not None else "no-DiT"
_idm_tag = f"IDM-{args.idm_size}" if model.idm is not None else "no-IDM"
MODEL_TAG        = f"LeWAM-{LEWAM_VERSION}_{args.identifier}_{_dit_tag}_{_idm_tag}"
TRAINING_RUN_DIR = os.path.join(le_wam_root, 'runs', MODEL_TAG)
os.makedirs(os.path.join(TRAINING_RUN_DIR, 'plots'), exist_ok=True)
print(f"{MODEL_TAG}\nTrainable params: {model.count_params()}M")
print(f"IDM: {IDM_NUM_PAST_FRAMES} past frame(s) → {IDM_NUM_FUTURE_FRAMES} future frame(s), {NUM_FUTURE // VJEPA2_TUBELET_SIZE} pairs per step")


_resume_ckpt = None
if args.resume:
    args.resume = resolve_checkpoint(args.resume, le_wam_root, args.s3_path, args.resume_from_aws)
if args.resume:
    print(f"Loading checkpoint from {args.resume}...")
    _resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(_resume_ckpt['model'])
    print(f"Resumed model from step {_resume_ckpt['num_steps']}")


dataset_suffix = "_small" if (args.sanity_check or args.small) else ""

norm_stats_path = os.path.join(cache_root, f"ehalicki/LeWAM_community_dataset{dataset_suffix}", 'norm_stats.pt')
if not os.path.exists(norm_stats_path):
    print(f"norm_stats.pt not found at {norm_stats_path}, precomputing now...")
    precompute_norm_stats(
        repo_id=f"ehalicki/LeWAM_community_dataset{dataset_suffix}",
        cache_root=cache_root,
        tubelet_size=VJEPA2_TUBELET_SIZE,
    )
action_preprocessor = ActionPreprocessor(norm_stats_path).to(device)

if not args.sanity_check:
    task_emb_path = os.path.join(cache_root, f"ehalicki/LeWAM_community_dataset{dataset_suffix}", 'task_embeddings.pt')
    if not os.path.exists(task_emb_path):
        print(f"task_embeddings.pt not found at {task_emb_path}, precomputing now...")
        precompute_task_embeddings(
            repo_id=f"ehalicki/LeWAM_community_dataset{dataset_suffix}",
            cache_root=cache_root,
        )
    task_emb_cache = torch.load(task_emb_path, map_location='cpu', weights_only=True)
    
else:
    task_emb_cache       = None


past_ts   = [-(NUM_PAST - 1 - i) / FPS for i in range(NUM_PAST)]
future_ts = [(i + 1) / FPS for i in range(NUM_FUTURE)]

image_transforms = transforms.Compose([
    transforms.Resize(CROP_SIZE, antialias=True),
    transforms.CenterCrop(CROP_SIZE),
])
cd = CommunityDataset(
    repo_id=f"ehalicki/LeWAM_community_dataset{dataset_suffix}",
    cache_root=cache_root,
)
cd.prefetch_metadata(
    delta_timestamps={
        "observation.images.image": past_ts + future_ts,
        "observation.state":        [0.0],
        "action":                   [(i-1) * VJEPA2_TUBELET_SIZE / FPS for i in range(CHUNK_LEN)],
    },
    image_transforms=image_transforms,
)


_idm_params = (
    list(model.idm.parameters()) +
    list(model.action_decoder.parameters()) +
    list(model.state_encoder.parameters())
)
if args.idm_only:
    _encoder_params = list(model.video_encoder.parameters())
    trainable_params = _idm_params + _encoder_params
    optimizer = torch.optim.AdamW([
        {'params': _idm_params,      'lr': args.base_lr * args.idm_lr_mult, 'base_lr': args.base_lr * args.idm_lr_mult},
        {'params': _encoder_params,  'lr': args.base_lr,                    'base_lr': args.base_lr},
    ])
else:
    _dit_params = list(model.dit.parameters())
    trainable_params = _dit_params + _idm_params
    optimizer = torch.optim.AdamW([
        {'params': _dit_params, 'lr': args.base_lr,                         'base_lr': args.base_lr},
        {'params': _idm_params, 'lr': args.base_lr * args.idm_lr_mult, 'base_lr': args.base_lr * args.idm_lr_mult},
    ])

ACCUM_STEPS = 1


def train_step(batch, do_update):
    with amp_ctx:
        state   = batch["observation.state"].to(device=device, dtype=AMP_DTYPE, non_blocking=True).squeeze(1)
        actions = batch["action"].to(device=device, dtype=AMP_DTYPE, non_blocking=True)
        rel_actions = actions[:, 1:] - actions[:, :-1]
        if action_preprocessor is not None:
            state       = action_preprocessor.normalize_state(state).to(AMP_DTYPE)
            rel_actions = action_preprocessor.normalize_rel_action(rel_actions).to(AMP_DTYPE)

        #print mean and std dev of rel actions and state

        cam_keys  = sorted(k for k in batch if k.startswith("observation.images.image") and not k.endswith("_is_pad"))
        n_cameras = len(cam_keys)
        B         = state.shape[0]

        with torch.no_grad():
            cam_past_list   = []
            cam_future_list = []
            if args.sanity_check:
                for _ in cam_keys:
                    cam_past_list.append(torch.randn(B, PAST_TOKENS, VJEPA2_DIM, device=device, dtype=AMP_DTYPE))
                    cam_future_list.append(torch.randn(B, FUTURE_TOKENS, VJEPA2_DIM, device=device, dtype=AMP_DTYPE))
                text_emb, text_mask = None, None
            else:
                for cam_key in cam_keys:
                    imgs  = batch[cam_key].to(device=device, non_blocking=True)
                    video = model.video_encoder.preprocessor(imgs).to(dtype=AMP_DTYPE)
                    emb   = model.encode_video(video)
                    cam_past_list.append(emb[:, :PAST_TOKENS])
                    cam_future_list.append(emb[:, PAST_TOKENS:])
                text_emb, text_mask = lookup_language_embeddings(task_emb_cache, batch["task"], device, dtype=AMP_DTYPE)

        # DiT flow matching loss: one forward per camera, other cameras' past as aux_frames
        dit_loss = torch.zeros(1, device=device, dtype=AMP_DTYPE)
        if model.dit is not None:
            for i, (pf, ff) in enumerate(zip(cam_past_list, cam_future_list)):
                aux = torch.cat([p for j, p in enumerate(cam_past_list) if j != i], dim=1) if n_cameras > 1 else None
                x0     = torch.randn_like(ff)
                t      = torch.rand(B, device=device, dtype=AMP_DTYPE)
                x_t    = (1 - t[:, None, None]) * x0 + t[:, None, None] * ff
                v_pred = model.predict_future(x_t, t, pf, state=state, lang=text_emb, aux_frames=aux, l_mask=text_mask)
                dit_loss += F.mse_loss(v_pred, ff - x0)
            dit_loss = dit_loss / n_cameras / ACCUM_STEPS

        # IDM loss: teacher forcing over consecutive tubelet pairs, all cameras concatenated
        # tubelet_seq: [last_past, fut_0, fut_1, ..., fut_{n-1}] each (B, C*tpt, D)
        tpt = PATCH_H * PATCH_W
        n_pairs = NUM_FUTURE // VJEPA2_TUBELET_SIZE
        tubelet_seq = [torch.cat([p[:, -tpt:] for p in cam_past_list], dim=1)]
        for _i in range(n_pairs):
            tubelet_seq.append(torch.cat([f[:, _i*tpt:(_i+1)*tpt] for f in cam_future_list], dim=1))

        pair_a = torch.cat([tubelet_seq[_i]   for _i in range(n_pairs)], dim=0)  # (B*n_pairs, C*tpt, D)
        pair_b = torch.cat([tubelet_seq[_i+1] for _i in range(n_pairs)], dim=0)
        pred   = model.infer_actions(pair_a, pair_b, state.repeat(n_pairs, 1))   # (B*n_pairs, 1, action_dim)
        pred   = pred.squeeze(1).view(n_pairs, B, -1).permute(1, 0, 2)           # (B, n_pairs, action_dim)
        idm_loss = F.mse_loss(pred, rel_actions) / ACCUM_STEPS

        loss = (dit_loss + LAMBDA_IDM_LOSS * idm_loss)

    scaler.scale(loss).backward()

    if do_update:
        if _is_cuda and not _bf16_ok:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    raw_future = batch[cam_keys[0]][:, NUM_PAST:].to(device=device, non_blocking=True).permute(0, 2, 1, 3, 4)
    return dit_loss * ACCUM_STEPS, idm_loss * ACCUM_STEPS, cam_past_list[0], cam_future_list[0], text_emb, text_mask, raw_future


print("Calibrating batch size...")
_max_cam_ds = cd.datasets[max(cd.datasets)]
_calib_loader = DataLoader(_max_cam_ds, batch_size=1, num_workers=0)
_sample_batch = next(iter(_calib_loader))

if _is_cuda and not args.sanity_check:
    def _try_batch(batch_size):
        test_batch = {
            k: v[:1].repeat(batch_size, *([1] * (v.ndim - 1))) if hasattr(v, 'ndim') else [v[0]] * batch_size
            for k, v in _sample_batch.items()
        }
        optimizer.zero_grad()
        train_step(test_batch, do_update=True)
        optimizer.zero_grad()

    BATCH_SIZE = find_max_batch_size(_try_batch, target_fraction=0.8)
    optimizer.state.clear()
    optimizer.zero_grad()
else:
    BATCH_SIZE = 1
    if args.sanity_check:
        print("Sanity check — skipping calibration, using BATCH_SIZE=1")
    else:
        print(f"Non-CUDA device ({device}) — skipping calibration, using BATCH_SIZE=4")

ACCUM_STEPS = max(1, args.effective_batch_size // BATCH_SIZE)
LR_SCALE    = (BATCH_SIZE * ACCUM_STEPS) / args.base_batch_size
for g in optimizer.param_groups:
    g['lr'] = g['base_lr'] * LR_SCALE
print(f"Effective batch size: {BATCH_SIZE * ACCUM_STEPS} (batch={BATCH_SIZE}, accum={ACCUM_STEPS}, lr_scale={LR_SCALE:.2f}x)")


warmup    = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=args.warmup_steps)
cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS - args.warmup_steps)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

'''
if _is_cuda:
    if model.dit is not None:
        model.dit = torch.compile(model.dit)
        print("torch.compile: DiT compiled")
    if model.idm is not None:
        model.idm = torch.compile(model.idm)
        print("torch.compile: IDM compiled")
    model.video_encoder.backbone = torch.compile(model.video_encoder.backbone)
    print("torch.compile: VJEPA2 backbone compiled")
'''

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
    return torch.utils.data.dataloader.default_collate(batch)


print("Building final dataloaders...")
_loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=args.num_workers,
    prefetch_factor=4 if args.num_workers > 0 else None,
    persistent_workers=args.num_workers > 0,
    collate_fn=_collate_skip_none,
    pin_memory=True,
    shuffle=True,
)
train_loaders = {n: DataLoader(_SafeDataset(ds), **_loader_kwargs) for n, ds in cd.datasets.items()}

def _infinite_interleaved(loaders):
    iters   = {n: iter(l) for n, l in loaders.items()}
    keys    = list(loaders.keys())
    weights = [len(loaders[n].dataset) for n in keys]
    while True:
        n = random.choices(keys, weights=weights, k=1)[0]
        #print(f"picked #{n}")
        try:
            batch = next(iters[n])
        except StopIteration:
            iters[n] = iter(loaders[n])
            batch = next(iters[n])
        if batch is not None:
            yield batch


num_steps  = 0
micro_step = 0

if _resume_ckpt is not None:
    ckpt_opt = _resume_ckpt['optimizer']
    if len(ckpt_opt['param_groups']) == len(optimizer.param_groups):
        optimizer.load_state_dict(ckpt_opt)
    else:
        print(f"Warning: optimizer param groups mismatch, skipping optimizer state.")
    num_steps  = _resume_ckpt['num_steps']
    micro_step = _resume_ckpt['micro_step']
    del _resume_ckpt
    warmup    = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=args.warmup_steps)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS - args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])
    print(f"Fast-forwarding scheduler to step {num_steps}...")
    for _ in range(num_steps):
        scheduler.step()
    print(f"Resuming training from step {num_steps}.")


loss_log = []



def save_checkpoint(past_frames=None, future_frames=None, text_emb=None, text_mask=None, label=None, raw_future=None):
    ckpt_path   = os.path.join(TRAINING_RUN_DIR, f'{MODEL_TAG}_latest.pt')
    losses_path = os.path.join(TRAINING_RUN_DIR, 'losses.json')
    s3_prefix   = f'{args.s3_path}/{MODEL_TAG}'

    torch.save({
        'num_steps':  num_steps,
        'micro_step': micro_step,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'config': {
            'crop_size':             CROP_SIZE,
            'fps':                   float(NUM_PAST // VJEPA2_TUBELET_SIZE),
            'num_past_frames':       NUM_PAST // VJEPA2_TUBELET_SIZE,
            'num_future_frames':     NUM_FUTURE // VJEPA2_TUBELET_SIZE,
            'patch_h':               PATCH_H,
            'patch_w':               PATCH_W,
            'dit_size':              args.dit_size if model.dit is not None else None,
            'idm_size':              args.idm_size if model.idm is not None else None,
            'idm_num_past_frames':   IDM_NUM_PAST_FRAMES,
            'idm_num_future_frames': IDM_NUM_FUTURE_FRAMES,
            'amp_dtype':             str(AMP_DTYPE),
            'batch_size':            BATCH_SIZE,
            'accum_steps':           ACCUM_STEPS,
            'effective_batch_size':  BATCH_SIZE * ACCUM_STEPS,
        },
    }, ckpt_path)

    with open(losses_path, 'w') as f:
        json.dump(loss_log, f)

    ode_path = None
    if model.dit is not None and past_frames is not None and future_frames is not None and text_emb is not None and text_mask is not None and raw_future is not None:
        save_ode_viz(
            model, future_frames[:1].detach(), past_frames[:1].detach(),
            text_emb[:1].detach(), text_mask[:1].detach(),
            label, num_steps, TRAINING_RUN_DIR, PATCH_H, PATCH_W,
            raw_future_frames=raw_future[:1].detach(),
        )
        ode_path = os.path.join(TRAINING_RUN_DIR, 'plots', f'ode-step{num_steps}.png')

    print(f"Checkpoint saved: {ckpt_path}")
    if args.s3_path and aws_available():
        upload_to_s3_async(ckpt_path, f'{s3_prefix}/{MODEL_TAG}_latest.pt')
        upload_to_s3_async(losses_path, f'{s3_prefix}/losses.json')
        if ode_path and os.path.exists(ode_path):
            upload_to_s3_async(ode_path, f'{s3_prefix}/ode-step{num_steps}.png')


start      = time.time()
start_step = num_steps
last_save_time = time.time()

print("Starting training loop...")
_dit_loss_acc = 0.0
_idm_loss_acc = 0.0
for batch in _infinite_interleaved(train_loaders):
    micro_step += 1
    do_update = (micro_step % ACCUM_STEPS == 0)
    dit_loss, idm_loss, past_frames, future_frames, text_emb, text_mask, raw_future = train_step(batch, do_update)
    _dit_loss_acc += dit_loss.item()
    _idm_loss_acc += idm_loss.item()

    if not do_update:
        continue

    scheduler.step()
    dit_loss_avg = _dit_loss_acc / ACCUM_STEPS
    idm_loss_avg = _idm_loss_acc / ACCUM_STEPS
    _dit_loss_acc = 0.0
    _idm_loss_acc = 0.0
    loss_log.append({'step': num_steps, 'dit_loss': dit_loss_avg, 'idm_loss': idm_loss_avg})

    steps_done    = num_steps - start_step + 1
    secs_per_step = (time.time() - start) / steps_done
    eta_secs      = secs_per_step * (STEPS - num_steps)
    eta_delta     = str(datetime.timedelta(seconds=int(eta_secs)))
    tz            = ZoneInfo(args.timezone)
    finish_wall   = datetime.datetime.now(tz) + datetime.timedelta(seconds=eta_secs)
    finish_str    = finish_wall.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"Step: {num_steps}/{STEPS}\tDiT: {dit_loss_avg:.4f}\tIDM: {idm_loss_avg:.4f}\tSPS: {secs_per_step:.2f}s\tETA: {eta_delta} ({finish_str})")

    if (num_steps + 1) % args.save_interval_steps == 0 or time.time() - last_save_time >= args.save_interval_seconds or num_steps == 10:
        save_checkpoint(past_frames, future_frames, text_emb, text_mask, batch['task'][0], raw_future)
        last_save_time = time.time()

    num_steps += 1
    if num_steps >= STEPS:
        break

save_checkpoint()
