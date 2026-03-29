import os
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*sdp_kernel.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*timm.models.layers.*')
import torch
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from wam.models.lewam import build_lewam_for_resume
from wam.models.encoders import load_t5gemma_encoder
from wam.training.common import resolve_checkpoint, embed_pca_rgb, ode_solve

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help='Checkpoint path relative to $LE_WAM_ROOT/runs/, e.g. MyRun/MyRun_latest.pt')
parser.add_argument('--video', type=str, required=True,
                    help='Path to input video file (mp4 or other torchvision-readable format)')
parser.add_argument('--label', type=str, required=True,
                    help='Language annotation for the video, e.g. "picking up the cup"')
parser.add_argument('--from-aws', action='store_true')
parser.add_argument('--s3-path', type=str, default='s3://zima-data/lewam/checkpoints')
parser.add_argument('--ode-steps', type=int, nargs='+', default=[5, 10, 20, 50, 100])
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--out-dir', type=str, default=None,
                    help='Output directory for plots. Defaults to $LE_WAM_ROOT/runs/<model_dir>/ode_eval/')
parser.add_argument('--t5gemma-path', type=str, default=None,
                    help='Path to T5Gemma weights. Defaults to $LE_WAM_ROOT/weights/t5gemma-s-s-prefixlm')
args = parser.parse_args()

device = args.device

le_wam_root = os.environ.get('LE_WAM_ROOT')
if not le_wam_root:
    raise ValueError('LE_WAM_ROOT environment variable not set')

t5gemma_path = args.t5gemma_path or os.path.join(le_wam_root, 'weights', 't5gemma-s-s-prefixlm')

ckpt_path = resolve_checkpoint(args.model, le_wam_root, args.s3_path, args.from_aws)
print(f'Loading checkpoint: {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
cfg  = ckpt['config']

VJEPA2_TUBELET_SIZE = 2
PATCH_SIZE          = 16
CROP_SIZE           = cfg.get('crop_size', 224)
PATCH_H = PATCH_W   = CROP_SIZE // PATCH_SIZE
TOKENS_PER_SECOND   = int(cfg.get('tokens_per_second', cfg.get('fps', 6)))
RAW_FRAMES          = cfg.get('raw_frames', TOKENS_PER_SECOND * 2 * VJEPA2_TUBELET_SIZE)
tokens_per_clip     = TOKENS_PER_SECOND * PATCH_H * PATCH_W

model = build_lewam_for_resume(cfg)
model.load_state_dict(ckpt['model'])
model.eval()
model = model.to(device)
model.video_encoder.preprocessor = model.video_encoder.preprocessor.to('cpu')

print(f'Loading language encoder from: {t5gemma_path}')
lang_encoder = load_t5gemma_encoder(path=t5gemma_path, device_map='cpu')

out_dir = args.out_dir
if out_dir is None:
    out_dir = os.path.join(le_wam_root, 'runs', os.path.dirname(args.model), 'ode_eval')
os.makedirs(out_dir, exist_ok=True)
print(f'Saving plots to: {out_dir}')


def load_video_frames(path: str, num_frames: int) -> torch.Tensor:
    """
    Load a video and uniformly sample num_frames frames.

    Returns (1, T, C, H, W) uint8.
    """
    with av.open(path) as container:
        frames = [
            torch.from_numpy(f.to_ndarray(format='rgb24'))
            for f in container.decode(video=0)
        ]  # list of (H, W, C) uint8
    total = len(frames)
    indices = torch.linspace(0, total - 1, num_frames).long()
    sampled = torch.stack([frames[i] for i in indices])          # (T, H, W, C)
    return sampled.permute(0, 3, 1, 2).unsqueeze(0)              # (1, T, C, H, W)


def save_comparison(raw_future, past_frames, future_frames, lang, l_mask, label):
    T = future_frames.shape[1] // (PATCH_H * PATCH_W)

    preds = {}
    for n in args.ode_steps:
        preds[n] = ode_solve(model, past_frames, future_frames, lang, l_mask, num_steps=n)[0]

    tokens_list = [future_frames[0]] + [preds[n] for n in args.ode_steps]
    rgbs        = embed_pca_rgb(tokens_list, T, PATCH_H, PATCH_W)
    gt_rgb      = rgbs[0]
    pred_rgbs   = {n: rgbs[i + 1] for i, n in enumerate(args.ode_steps)}

    frames = raw_future[0].permute(1, 0, 2, 3).float().cpu()
    frames = model.video_encoder.preprocessor.unnormalize(frames)
    frames = F.interpolate(frames, size=(PATCH_H * 8, PATCH_W * 8), mode='bilinear', align_corners=False)
    frames_np = frames.permute(0, 2, 3, 1).numpy().clip(0, 1)

    n_rows = 2 + len(args.ode_steps)
    fig, axes = plt.subplots(n_rows, T, figsize=(2 * T, 2 * n_rows), squeeze=False)

    for t in range(T):
        axes[0][t].imshow(frames_np[t])
        axes[0][t].axis('off')
        axes[0][t].set_title(f't={t}', fontsize=8)
    axes[0][0].set_ylabel('gt frame', fontsize=8)

    for t in range(T):
        axes[1][t].imshow(gt_rgb[t])
        axes[1][t].axis('off')
    axes[1][0].set_ylabel('gt embed', fontsize=8)

    for row, n in enumerate(args.ode_steps, start=2):
        for t in range(T):
            axes[row][t].imshow(pred_rgbs[n][t])
            axes[row][t].axis('off')
        axes[row][0].set_ylabel(f'{n} steps', fontsize=8)

    plt.suptitle(f'ODE steps comparison\n"{label}"', fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'ode_steps.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


print(f'Loading video: {args.video}')
raw = load_video_frames(args.video, RAW_FRAMES)

with torch.no_grad():
    preprocessed = model.video_encoder.preprocess(raw)
    video_embeddings = model.encode_video(preprocessed.to(device))
    lang, l_mask = lang_encoder([args.label])
    lang, l_mask = lang.to(device=device, dtype=torch.float32), l_mask.to(device)

past_frames   = video_embeddings[:, :tokens_per_clip]
future_frames = video_embeddings[:, tokens_per_clip:]
raw_future    = raw[:, RAW_FRAMES // 2 :: VJEPA2_TUBELET_SIZE]

save_comparison(raw_future, past_frames, future_frames, lang, l_mask, args.label)

print('Done.')
