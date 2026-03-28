import os
import sys
import json
import subprocess
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*sdp_kernel.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*timm.models.layers.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*lr_scheduler.step().*')
import torch
from torch.utils.data import DataLoader
from wam.datasets.somethingsomethingv2 import SomethingSomethingV2Dataset
from wam.models.lewam import build_lewam_with_encoders, VERSION as LEWAM_VERSION
from wam.training.losses import SIGReg
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import time
import datetime
from zoneinfo import ZoneInfo
from torch.profiler import profile, record_function, ProfilerActivity
from wam.training.common import save_ode_viz, lookup_language_embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true')
parser.add_argument('--identifier', type=str, default='ssv2_pretrain')
parser.add_argument('--effective-batch-size', type=int, default=256)
parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16')
parser.add_argument('--warmup-steps', type=int, default=100)
parser.add_argument('--frozen-encoder-steps', type=int, default=100)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save-interval-steps', type=int, default=100)
parser.add_argument('--save-interval-seconds', type=int, default=600)
parser.add_argument('--s3-path', type=str, default='s3://zima-data/lewam/checkpoints')
parser.add_argument('--steps', type=int, default=2000)
parser.add_argument('--timezone', type=str, default='Europe/Zurich')
args = parser.parse_args()

IDENTIFIER = args.identifier

SSV2_FPS            = 12
VJEPA2_TUBELET_SIZE = 2
PATCH_SIZE          = 16
CROP_SIZE           = 224

TOKENS_PER_SECOND = SSV2_FPS // VJEPA2_TUBELET_SIZE
PATCH_H = PATCH_W = CROP_SIZE // PATCH_SIZE
RAW_FRAMES        = (TOKENS_PER_SECOND * 2) * VJEPA2_TUBELET_SIZE

STEPS         = args.steps
LAMBDA_SIGREG = 0.1  # following LeWorldModel, https://arxiv.org/pdf/2603.19312

AMP_DTYPE = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
scaler = GradScaler(enabled=args.dtype == 'float16')
device = "cuda"

NUM_WORKERS = max(1, (os.cpu_count() or 8) // 4)

le_wam_root = os.environ.get('LE_WAM_ROOT')
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")

data_dir    = os.path.join(le_wam_root, 'data', 'somethingsomethingv2')
weights_dir = os.path.join(le_wam_root, 'weights')

TRAINING_RUN_DIR = os.path.join(le_wam_root, 'runs', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(os.path.join(TRAINING_RUN_DIR, 'plots'), exist_ok=True)

model = build_lewam_with_encoders(
    vjepa2_checkpoint=os.path.join(weights_dir, 'vjepa2_1_vitb_dist_vitG_384.pt'),
    load_language_encoder=False,
    crop_size=CROP_SIZE,
    fps=float(TOKENS_PER_SECOND),
    num_past_frames=TOKENS_PER_SECOND,
    num_future_frames=TOKENS_PER_SECOND,
    patch_h=PATCH_H,
    patch_w=PATCH_W,
    dit_size='B',
    idm_size=None,
)
model.train()
model = model.to(device)
_dit_tag = f"DiT-B" if model.dit is not None else "no-DiT"
_idm_tag = f"IDM-B" if model.idm is not None else "no-IDM"
MODEL_TAG = f"LeWAM-{LEWAM_VERSION}_{IDENTIFIER}_{_dit_tag}_{_idm_tag}"
print(f"{MODEL_TAG} | {model.count_params()}M trainable params (all unfrozen)")
if args.frozen_encoder_steps > 0:
    model.video_encoder.set_frozen(True)
    print(f"Video encoder frozen for first {args.frozen_encoder_steps} steps. Trainable: {model.count_params()}M")
model.video_encoder.preprocessor = model.video_encoder.preprocessor.to('cpu')

_resume_ckpt = None
if args.resume:
    print(f"Loading checkpoint from {args.resume}...")
    _resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(_resume_ckpt['model'])
    print(f"Resumed model from step {_resume_ckpt['num_steps']}")

label_emb_path = os.path.join(data_dir, 'label_embeddings.pt')
if not os.path.exists(label_emb_path):
    print("label_embeddings.pt not found — running precompute_label_embeddings.py...")
    subprocess.run(
        [sys.executable, os.path.join(le_wam_root, 'src/wam/training/scripts/precompute_label_embeddings.py')],
        check=True,
    )

label_emb_cache = torch.load(label_emb_path, map_location='cpu', weights_only=True)


train_dataset = SomethingSomethingV2Dataset(
    data_dir, split='train', load_videos=True, num_frames=RAW_FRAMES,
    transform=model.video_encoder.preprocess,
)
train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True,
    num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True,
    collate_fn=train_dataset.collate_fn,
)

num_steps = 0
optimizer = torch.optim.AdamW([ 
    {'params': model.video_encoder.parameters(), 'lr': 1e-4, 'base_lr': 5e-4},
    {'params': model.dit.parameters(), 'lr': 1e-3, 'base_lr': 1e-3},
])
sigreg_loss = SIGReg(dtype=AMP_DTYPE).to(device)

PROFILE_STEPS   = 5
ACCUM_STEPS     = 1  # updated after batch size calibration


def train_step(batch, do_update, use_sigreg=True):
    with autocast(device_type='cuda', dtype=AMP_DTYPE):
        with record_function("encode_video"):
            raw = batch['video'].to(device, non_blocking=True)
            B   = raw.shape[0]
            video_embeddings = model.encode_video(raw)

        with record_function("encode_language"):
            text_embeddings, text_mask = lookup_language_embeddings(label_emb_cache, batch['label'], device, dtype=AMP_DTYPE)

        with record_function("dit_forward"):
            tokens_per_clip = TOKENS_PER_SECOND * PATCH_H * PATCH_W
            past_frames   = video_embeddings[:, :tokens_per_clip]
            future_frames = video_embeddings[:, tokens_per_clip:]
            x0  = torch.randn_like(future_frames)
            x1  = future_frames
            t_s = torch.rand(B, device=device)
            x_t = (1 - t_s[:, None, None]) * x0 + t_s[:, None, None] * x1
            v_pred = model.predict_future(x_t, t_s, past_frames, lang=text_embeddings, l_mask=text_mask)
            if use_sigreg:
                loss = (F.mse_loss(v_pred, x1 - x0) + LAMBDA_SIGREG * sigreg_loss(video_embeddings)) / ACCUM_STEPS
            else:
                loss = (F.mse_loss(v_pred, x1 - x0)) / ACCUM_STEPS


    with record_function("backward"):
        scaler.scale(loss).backward()

    if do_update:
        with record_function("optimizer_step"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    raw_future = raw[:, :, RAW_FRAMES // 2 :: VJEPA2_TUBELET_SIZE]
    return loss * ACCUM_STEPS, past_frames, future_frames, text_embeddings, text_mask, raw_future


#---------------BATCH SIZE CALIBRATION---------------------
def find_max_batch_size(sample_batch, target_fraction=0.9):
    total_vram = torch.cuda.get_device_properties(0).total_memory

    def try_batch(batch_size):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        test_batch = {
            'video': sample_batch['video'][:1].repeat(batch_size, 1, 1, 1, 1),
            'label': [sample_batch['label'][0]] * batch_size,
        }
        optimizer.zero_grad()
        train_step(test_batch, do_update=True)
        optimizer.zero_grad()
        peak = torch.cuda.max_memory_allocated()
        return peak / total_vram

    lo, hi = 1, 1
    while True:
        try:
            used = try_batch(hi)
            print(f"  batch_size={hi}: {used*100:.1f}% VRAM peak")
            if used >= target_fraction:
                lo = hi // 2
                break
            lo = hi
            hi *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            lo = hi // 2
            break

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        try:
            used = try_batch(mid)
            print(f"  batch_size={mid}: {used*100:.1f}% VRAM peak")
            if used >= target_fraction:
                hi = mid
            else:
                lo = mid
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hi = mid

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Selected batch_size={lo}")
    return lo

print("Calibrating batch size...")
_calib_iter = iter(train_loader)
_sample_batch = next(_calib_iter)
BATCH_SIZE = find_max_batch_size(_sample_batch)
optimizer.state.clear()
optimizer.zero_grad()
ACCUM_STEPS = max(1, args.effective_batch_size // BATCH_SIZE)
LR_SCALE = (BATCH_SIZE * ACCUM_STEPS) / 64
for g in optimizer.param_groups:
    g['lr'] = g['base_lr'] * LR_SCALE
print(f"Effective batch size: {BATCH_SIZE * ACCUM_STEPS} (batch={BATCH_SIZE}, accum={ACCUM_STEPS}, lr_scale={LR_SCALE:.2f}x)")
warmup    = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=args.warmup_steps)
cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS - args.warmup_steps)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True,
    pin_memory=True, collate_fn=train_dataset.collate_fn,
)

if _resume_ckpt is not None:
    optimizer.load_state_dict(_resume_ckpt['optimizer'])
    scheduler.load_state_dict(_resume_ckpt['scheduler'])
    num_steps  = _resume_ckpt['num_steps']
    micro_step = _resume_ckpt['micro_step']
    del _resume_ckpt
    if num_steps >= args.frozen_encoder_steps:
        model.video_encoder.set_frozen(False)
    print(f"Resuming training from step {num_steps}.")
#---------------BATCH SIZE CALIBRATION END-----------------


#---------------PROFILING ONLY START-----------------------
prof = None
profiling_done = False
if args.profile:
    trace_path = os.path.join(TRAINING_RUN_DIR, 'profiler', 'trace.json')
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    def on_trace_ready(p):
        p.export_chrome_trace(trace_path)
        print(f"Trace saved to {trace_path} — open at https://ui.perfetto.dev")
        print(p.key_averages().table(sort_by='cpu_time_total', row_limit=15))

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
        on_trace_ready=on_trace_ready,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=PROFILE_STEPS),
    )
    prof.start()

#---------------PROFILING ONLY END -----------------------


loss_log = []

def _aws_available():
    return args.s3_path and subprocess.run(['which', 'aws'], capture_output=True).returncode == 0

def save_checkpoint(past_frames=None, future_frames=None, text_embeddings=None, text_mask=None, label=None, raw_future=None):
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
            'crop_size':        CROP_SIZE,
            'fps':              float(TOKENS_PER_SECOND),
            'num_past_frames':  TOKENS_PER_SECOND,
            'num_future_frames': TOKENS_PER_SECOND,
            'patch_h':          PATCH_H,
            'patch_w':          PATCH_W,
            'dit_size':         'B' if model.dit is not None else None,
            'idm_size':         'B' if model.idm is not None else None,
            'language_encoder': model.language_encoder is not None,
            'raw_frames':       RAW_FRAMES,
            'tokens_per_second': TOKENS_PER_SECOND,
            'amp_dtype':        str(AMP_DTYPE),
            'batch_size':       BATCH_SIZE,
            'accum_steps':      ACCUM_STEPS,
            'effective_batch_size': BATCH_SIZE * ACCUM_STEPS,
        },
    }, ckpt_path)

    with open(losses_path, 'w') as f:
        json.dump(loss_log, f)

    ode_path = None
    if past_frames is not None and future_frames is not None and text_embeddings is not None and text_mask is not None and raw_future is not None:
        save_ode_viz(
            model, future_frames.detach(), past_frames.detach(),
            text_embeddings.detach(), text_mask.detach(),
            label, num_steps, TRAINING_RUN_DIR, PATCH_H, PATCH_W,
            raw_future_frames=raw_future.detach(),
        )
        ode_path = os.path.join(TRAINING_RUN_DIR, 'plots', f'ode-step{num_steps}.png')

    print(f"Checkpoint saved: {ckpt_path}")
    if _aws_available():
        subprocess.run(['aws', 's3', 'cp', ckpt_path, f'{s3_prefix}/{MODEL_TAG}_latest.pt'], check=False)
        subprocess.run(['aws', 's3', 'cp', losses_path, f'{s3_prefix}/losses.json'], check=False)
        if ode_path and os.path.exists(ode_path):
            subprocess.run(['aws', 's3', 'cp', ode_path, f'{s3_prefix}/ode-step{num_steps}.png'], check=False)


start = time.time()

#---------------MAIN LOOP----------------------
last_save_time = time.time()
micro_step = 0
while num_steps <= STEPS:
    for batch in train_loader:
        if batch is None:
            continue

        micro_step += 1
        do_update = (micro_step % ACCUM_STEPS == 0)
        loss, past_frames, future_frames, text_embeddings, text_mask, raw_future = train_step(batch, do_update, use_sigreg=(num_steps >= args.frozen_encoder_steps))

        if not do_update:
            continue

        if prof is not None and not profiling_done:
            prof.step()
            if num_steps >= 1 + 1 + PROFILE_STEPS:
                prof.stop()
                profiling_done = True

        if num_steps == args.frozen_encoder_steps:
            model.video_encoder.set_frozen(False)
            print(f"Step {num_steps}: video encoder unfrozen.")

        scheduler.step()
        loss_val = loss.item()
        loss_log.append({'step': num_steps, 'loss': loss_val})
        secs_per_step = (time.time() - start) / (num_steps + 1)
        eta_secs = secs_per_step * (STEPS - num_steps)
        eta_delta = str(datetime.timedelta(seconds=int(eta_secs)))
        tz = ZoneInfo(args.timezone)
        finish_wall = datetime.datetime.now(tz) + datetime.timedelta(seconds=eta_secs)
        finish_str = finish_wall.strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"Step: {num_steps}/{STEPS}\tLoss: {loss_val:.4f}\tSPS: {secs_per_step:.2f}s\tETA: {eta_delta} ({finish_str})")
        if (num_steps + 1) % args.save_interval_steps == 0 or time.time() - last_save_time >= args.save_interval_seconds:
            save_checkpoint(past_frames, future_frames, text_embeddings, text_mask, batch['label'][0], raw_future)
            last_save_time = time.time()

        num_steps += 1
        if num_steps > STEPS:
            break
