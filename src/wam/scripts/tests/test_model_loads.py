import os
from wam.models.lewam import build_lewam_with_encoders

le_wam_root = os.environ.get('LE_WAM_ROOT')
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")

vjepa2_checkpoint = os.path.join(le_wam_root, 'weights/vjepa2_1_vitb_dist_vitG_384.pt')
t5gemma_checkpoint = os.path.join(le_wam_root, 'weights/t5gemma-s-s-prefixlm')

model = build_lewam_with_encoders(vjepa2_checkpoint=vjepa2_checkpoint,
                                  t5gemma_checkpoint=t5gemma_checkpoint)

def count_params(model, millions=True):
    multiplier = 1.
    if millions:
        multiplier = 1e-6
    return round(sum(p.numel() for p in model.parameters() if p.requires_grad)*multiplier, ndigits=0)

model.video_encoder.set_frozen(True)
print(f"Frozen vision and language: {count_params(model)}M")
model.video_encoder.set_frozen(False)
print(f"Unfrozen vision, frozen language: {count_params(model)}M")
model.language_encoder.set_frozen(False)
print(f"Unfrozen vision and language: {count_params(model)}M")

print("Everything works!")

