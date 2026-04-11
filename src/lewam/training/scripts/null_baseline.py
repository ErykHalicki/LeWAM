"""
Compute null baseline losses for flow matching training.

Null baseline = a model that learned to:
  - predict velocity toward the last context frame embedding (video)
  - predict velocity toward zero actions (actions)

The flow matching training loss is:
  L = E_{t,x0}[ || v_pred - (x_1 - x_0) ||^2 ]
where x_0 ~ N(0,I) is noise, x_1 is the target, and v_pred is the model output.

A null model that predicts v_null = x_null - x_0 (velocity toward a trivial
target x_null instead of the true x_1) achieves:
  L_null = E[ || (x_null - x_0) - (x_1 - x_0) ||^2 ]
         = E[ || x_null - x_1 ||^2 ]

The x_0 terms cancel, so the null baseline is independent of noise and timestep.
This simplifies to a direct MSE between the trivial target and ground truth:
  video:  E[ || last_ctx_repeated - future_tokens ||^2 ]
  action: E[ || 0 - rel_velocity ||^2 ] = E[ || rel_velocity ||^2 ]

Usage:
    python src/lewam/training/scripts/null_baseline.py --config configs/train/default.yaml
    python src/lewam/training/scripts/null_baseline.py --config configs/train/default.yaml --samples 500
"""
import os
import argparse
import random

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from lewam.datasets.community_dataset import CommunityDataset
from lewam.models.lewam import LeWAM
from lewam.models.common import ActionPreprocessor
from lewam.training.scripts.precompute_norm_stats import precompute_norm_stats

PATCH_SIZE = LeWAM.VJEPA_PATCH_SIZE
TUBELET = LeWAM.VJEPA_TUBELET_SIZE

le_wam_root = os.environ.get("LE_WAM_ROOT")
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    with open(os.path.join(le_wam_root, args.config)) as f:
        train_cfg = yaml.safe_load(f)

    model_config_path = train_cfg["model_config"]
    with open(os.path.join(le_wam_root, model_config_path)) as f:
        model_cfg = yaml.safe_load(f)

    dataset_suffix = "_small" if train_cfg["small_dataset"] else ""
    repo_id = f"ehalicki/LeWAM_community_dataset{dataset_suffix}"
    cache_root = os.path.join(le_wam_root, ".cache")

    num_context = model_cfg["num_context_frames"]
    num_future = model_cfg["num_future_frames"]
    scaled_fps = train_cfg["scaled_fps"]
    action_fps = train_cfg["action_fps"]
    crop_size = train_cfg["crop_size"]

    norm_stats_path = os.path.join(cache_root, repo_id, "norm_stats.pt")
    precompute_norm_stats(repo_id=repo_id, cache_root=cache_root, action_fps=action_fps)
    norm_stats = torch.load(norm_stats_path, map_location="cpu", weights_only=True)
    norm_strategy = model_cfg.get("norm_strategy", "q2_q98")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = ActionPreprocessor(norm_stats, norm_strategy).to(device)

    patch_side = crop_size // PATCH_SIZE
    num_fut_tubelets = num_future // TUBELET

    print("Building VJEPA2 encoder...")
    from lewam.models.video_encoder import build_vjepa2_encoder_arch, load_vjepa2_encoder
    encoder = build_vjepa2_encoder_arch(crop_size=crop_size)
    vjepa2_path = os.path.join(le_wam_root, "weights", "vjepa2_1_vitb_dist_vitG_384.pt")
    if os.path.exists(vjepa2_path):
        loaded = load_vjepa2_encoder(vjepa2_path, crop_size=crop_size)
        encoder.backbone.load_state_dict(loaded.backbone.state_dict())
        print("Loaded VJEPA2 weights")
    encoder.set_frozen(True)
    encoder = encoder.to(device)

    past_ts = [-(num_context - 1 - i) / scaled_fps for i in range(num_context)]
    future_ts = [(i + 1) / scaled_fps for i in range(num_future)]
    action_horizon = int((num_future / scaled_fps) * action_fps)
    action_ts = [i / action_fps for i in range(action_horizon + 1)]

    print(f"Loading dataset ({repo_id})...")
    cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)
    cd.prefetch_metadata(
        delta_timestamps={
            "observation.images.image": past_ts + future_ts,
            "observation.state": [0.0],
            "action": action_ts,
        },
        image_transforms=transforms.Resize((crop_size, crop_size), antialias=True),
    )

    loaders = {}
    for n_cams, ds in sorted(cd.datasets.items()):
        loaders[n_cams] = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    cam_keys_list = list(loaders.keys())
    weights = [len(cd.datasets[n]) for n in cam_keys_list]
    iters = {n: iter(l) for n, l in loaders.items()}

    video_mse_sum = 0.0
    action_mse_sum = 0.0
    count = 0

    print(f"\nSampling from {len(cam_keys_list)} camera subsets (weighted by size)...")
    while count < args.samples:
        n_cams = random.choices(cam_keys_list, weights=weights, k=1)[0]
        try:
            raw = next(iters[n_cams])
        except StopIteration:
            iters[n_cams] = iter(loaders[n_cams])
            raw = next(iters[n_cams])

        cam_spatial = patch_side * patch_side * n_cams
        cam_keys = sorted(
            k for k in raw
            if k.startswith("observation.images.image") and not k.endswith("_is_pad")
        )
        total_frames = num_context + num_future
        all_frames = torch.stack([raw[k][:, :total_frames] for k in cam_keys], dim=1)
        ctx_frames = all_frames[:, :, :num_context].to(device)
        fut_frames = all_frames[:, :, num_context:].to(device)

        with torch.no_grad():
            ctx_tokens = encoder(encoder.preprocessor(ctx_frames))
            fut_tokens = encoder(encoder.preprocessor(fut_frames))

        last_ctx_tubelet = ctx_tokens[:, -cam_spatial:]
        null_video = last_ctx_tubelet.unsqueeze(1).expand(-1, num_fut_tubelets, -1, -1).reshape(ctx_tokens.shape[0], -1, ctx_tokens.shape[-1])

        video_mse = (null_video - fut_tokens).pow(2).mean().item()

        actions = raw["action"].to(device)
        dt = 1.0 / action_fps
        rel_vel = (actions[:, 1:] - actions[:, :-1]) / dt
        rel_vel = preprocessor.normalize_rel_action(rel_vel.to(device))
        action_mse = rel_vel.pow(2).mean().item()

        B = ctx_tokens.shape[0]
        video_mse_sum += video_mse * B
        action_mse_sum += action_mse * B
        count += B
        print(f"  {count}/{args.samples} samples processed")

    video_baseline = video_mse_sum / count
    action_baseline = action_mse_sum / count
    action_weight = train_cfg["action_weight"]
    total_baseline = video_baseline + action_weight * action_baseline

    print(f"\n{'='*60}")
    print(f"Null baseline losses ({count} samples)")
    print(f"{'='*60}")
    print(f"  video_loss:  {video_baseline:.4f}  (predict last context frame)")
    print(f"  action_loss: {action_baseline:.4f}  (predict zero velocity)")
    print(f"  total_loss:  {total_baseline:.4f}  (action_weight={action_weight})")
    print(f"{'='*60}")
    print(f"\nIf your model's loss is below these values, it's learning")
    print(f"something beyond trivially copying context / predicting zero.")


if __name__ == "__main__":
    main()
