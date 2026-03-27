import os
from wam.models.lewam import load_t5gemma_encoder

le_wam_root = os.environ.get('LE_WAM_ROOT')
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")

t5gemma_checkpoint = os.path.join(le_wam_root, 'weights/t5gemma-s-s-prefixlm')

encoder = load_t5gemma_encoder(path=t5gemma_checkpoint)

encoder.backbone.save_pretrained(t5gemma_checkpoint, use_safetensors=True)

