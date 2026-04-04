"""
Generate figure assets for the paper: raw frames and PCA visualizations.

Usage:
    python paper/generate_figures.py
    python paper/generate_figures.py --crop-size 384 --num-context 16 --num-future 32
"""
import os
import argparse

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2 as transforms

from wam.datasets.community_dataset import CommunityDataset
from wam.models.video_encoder import build_vjepa2_encoder_arch, load_vjepa2_encoder
from wam.training.common import embed_pca_rgb

PATCH_SIZE = 16
TUBELET_SIZE = 2

le_wam_root = os.environ.get("LE_WAM_ROOT", ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--num-context", type=int, default=16)
    parser.add_argument("--num-future", type=int, default=32)
    parser.add_argument("--scaled-fps", type=float, default=15.0)
    parser.add_argument("--out-dir", type=str, default="paper/figures/generated")
    args = parser.parse_args()

    out_dir = os.path.join(le_wam_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    crop_size = args.crop_size
    num_context = args.num_context
    num_future = args.num_future
    scaled_fps = args.scaled_fps
    total_frames = num_context + num_future
    patch_side = crop_size // PATCH_SIZE

    past_ts = [-(num_context - 1 - i) / scaled_fps for i in range(num_context)]
    future_ts = [(i + 1) / scaled_fps for i in range(num_future)]

    repo_id = "ehalicki/LeWAM_community_dataset_small"
    cache_root = os.path.join(le_wam_root, ".cache")

    print("Loading dataset...")
    cd = CommunityDataset(repo_id=repo_id, cache_root=cache_root)
    cd.prefetch_metadata(
        delta_timestamps={
            "observation.images.image": past_ts + future_ts,
            "observation.state": [0.0],
            "action": [0.0],
        },
        image_transforms=transforms.Compose([
            transforms.Resize(crop_size, antialias=True),
            transforms.CenterCrop(crop_size),
        ]),
    )

    target_cams = 2
    if target_cams not in cd.datasets:
        available = list(cd.datasets.keys())
        print(f"No {target_cams}-camera subset, available: {available}")
        target_cams = available[0]

    ds = cd.datasets[target_cams]
    sample = ds[0]

    cam_keys = sorted(
        k for k in sample
        if k.startswith("observation.images.image") and not k.endswith("_is_pad")
    )
    print(f"Camera keys: {cam_keys}")

    all_frames = torch.stack([sample[k][:total_frames] for k in cam_keys], dim=0)
    ctx_frames = all_frames[:, :num_context]
    fut_frames = all_frames[:, num_context:]

    print("Saving raw frames...")
    for ci, cam_key in enumerate(cam_keys):
        for phase, frames, indices in [
            ("ctx", ctx_frames[ci], [0, num_context // 2, num_context - 1]),
            ("fut", fut_frames[ci], [0, num_future // 2, num_future - 1]),
        ]:
            for fi in indices:
                frame = frames[fi]
                if frame.dtype == torch.uint8:
                    img = frame.permute(1, 2, 0).numpy()
                else:
                    img = (frame.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
                Image.fromarray(img).save(
                    os.path.join(out_dir, f"cam{ci}_{phase}_f{fi}.png")
                )
                print(f"  cam{ci}_{phase}_f{fi}.png")

    print("Building V-JEPA2 encoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_vjepa2_encoder_arch(crop_size=crop_size)
    vjepa2_path = os.path.join(le_wam_root, "weights", "vjepa2_1_vitb_dist_vitG_384.pt")
    if os.path.exists(vjepa2_path):
        loaded = load_vjepa2_encoder(vjepa2_path, crop_size=crop_size)
        encoder.backbone.load_state_dict(loaded.backbone.state_dict())
        print("Loaded V-JEPA2 weights")
    encoder.set_frozen(True)
    encoder = encoder.to(device)

    print("Encoding and generating PCA visualizations...")
    with torch.no_grad():
        ctx_tokens = encoder(encoder.preprocessor(ctx_frames.unsqueeze(0).to(device)))
        fut_tokens = encoder(encoder.preprocessor(fut_frames.unsqueeze(0).to(device)))

    num_ctx_tubelets = num_context // TUBELET_SIZE
    num_fut_tubelets = num_future // TUBELET_SIZE
    spatial = patch_side * patch_side * len(cam_keys)

    ctx_pca = embed_pca_rgb(
        [ctx_tokens[0]], num_ctx_tubelets, patch_side, patch_side * len(cam_keys),
    )[0]
    fut_pca = embed_pca_rgb(
        [fut_tokens[0]], num_fut_tubelets, patch_side, patch_side * len(cam_keys),
    )[0]

    for phase, pca_frames, n_tubelets in [
        ("ctx", ctx_pca, num_ctx_tubelets),
        ("fut", fut_pca, num_fut_tubelets),
    ]:
        indices = [0, n_tubelets // 2, n_tubelets - 1]
        for ti in indices:
            img = (pca_frames[ti] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(
                os.path.join(out_dir, f"pca_{phase}_t{ti}.png")
            )
            print(f"  pca_{phase}_t{ti}.png")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
