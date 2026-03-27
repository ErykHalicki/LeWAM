import os
import torch
from torch.utils.data import DataLoader
from wam.datasets.somethingsomethingv2 import SomethingSomethingV2Dataset
from wam.models.lewam import build_lewam_with_encoders
from torch.amp import autocast, GradScaler

scaler = GradScaler()
device="cuda"

le_wam_root = os.environ.get('LE_WAM_ROOT')
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")

data_dir    = os.path.join(le_wam_root, 'data', 'somethingsomethingv2')
weights_dir = os.path.join(le_wam_root, 'weights')

SSV2_FPS            = 12
VJEPA2_TUBELET_SIZE = 2
PATCH_SIZE          = 16
CROP_SIZE           = 224

TOKENS_PER_SECOND = SSV2_FPS // VJEPA2_TUBELET_SIZE            # 6
PATCH_H = PATCH_W = CROP_SIZE // PATCH_SIZE                    # 14
RAW_FRAMES        = (TOKENS_PER_SECOND * 2) * VJEPA2_TUBELET_SIZE  # 24: tubelet*(past+future)

model = build_lewam_with_encoders(
    vjepa2_checkpoint=os.path.join(weights_dir, 'vjepa2_1_vitb_dist_vitG_384.pt'),
    t5gemma_checkpoint=os.path.join(weights_dir, 't5gemma-s-s-prefixlm'),
    crop_size=CROP_SIZE,
    fps=float(TOKENS_PER_SECOND),
    num_past_frames=TOKENS_PER_SECOND,
    num_future_frames=TOKENS_PER_SECOND,
    patch_h=PATCH_H,
    patch_w=PATCH_W,
    dit_size='Baby',
    idm_size='Baby',
)
model.eval()
model = model.to(device)
model.video_encoder.preprocessor = model.video_encoder.preprocessor.to('cpu')

train_dataset = SomethingSomethingV2Dataset(
    data_dir, split='train', load_videos=True, num_frames=RAW_FRAMES,
    transform=model.video_encoder.preprocess,
)
train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn)

with torch.no_grad():
    for batch in train_loader:
        if batch is None:
            continue
        
        raw = batch['video'].to(device)  # (B, C, RAW_FRAMES//TUBELET, H, W) — preprocessed
        B   = raw.shape[0]
        print(f"preprocessed:  {tuple(raw.shape)}")

        video_embeddings = model.encode_video(raw)  # (B, TOKENS_PER_SECOND*2 * PATCH_H*PATCH_W, 768)
        print(f"video_embeddings:    {tuple(video_embeddings.shape)}")

        text_embeddings, text_mask = model.encode_language(batch['label'])
        print(f"text_embeddings:    {tuple(text_embeddings.shape)}")

        tokens_per_clip = TOKENS_PER_SECOND * PATCH_H * PATCH_W
        past_frames   = video_embeddings[:, :tokens_per_clip]
        future_frames = video_embeddings[:, tokens_per_clip:]
        print(f"past_frames:   {tuple(past_frames.shape)}")
        print(f"future_frames: {tuple(future_frames.shape)}")

        x0  = torch.randn_like(future_frames, device=device)
        t_s = torch.rand(B, device=device)
        x_t = (1 - t_s[:, None, None]) * x0 + t_s[:, None, None] * future_frames
        
        with autocast(device_type='cuda', dtype=torch.float16):
            v_pred  = model.predict_future(x_t, t_s, past_frames, lang=text_embeddings, l_mask=text_mask)
            print(f"v_pred:        {tuple(v_pred.shape)}")
        break
