"""
Precompute T5Gemma language embeddings for all unique SSv2 labels.
Saves {label_str: (embedding_cpu float16, mask_cpu bool)} to label_embeddings.pt.

Usage:
    python src/wam/training/scripts/precompute_label_embeddings.py
"""
import os
import json
import zipfile
import torch
from wam.models.encoders import load_t5gemma_encoder

le_wam_root = os.environ.get('LE_WAM_ROOT')
if not le_wam_root:
    raise ValueError("LE_WAM_ROOT environment variable not set")

data_dir    = os.path.join(le_wam_root, 'data', 'somethingsomethingv2')
weights_dir = os.path.join(le_wam_root, 'weights')
out_path    = os.path.join(data_dir, 'label_embeddings.pt')

labels_zip = os.path.join(data_dir, 'labels.zip')
with zipfile.ZipFile(labels_zip, 'r') as zf:
    with zf.open('labels/train.json') as f:
        train_annotations = json.load(f)
    with zf.open('labels/validation.json') as f:
        val_annotations = json.load(f)

unique_labels = sorted({a['label'] for a in train_annotations + val_annotations if 'label' in a})
print(f"Found {len(unique_labels)} unique labels")

if torch.cuda.is_available():
    cc_major = torch.cuda.get_device_properties(0).major
    dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
else:
    dtype = torch.float32

print(f"Using dtype: {dtype}")
encoder = load_t5gemma_encoder(
    path=os.path.join(weights_dir, 't5gemma-s-s-prefixlm'),
    torch_dtype=dtype,
    device_map='cuda' if torch.cuda.is_available() else 'cpu',
)
encoder.eval()

BATCH_SIZE = 32
cache = {}

with torch.no_grad():
    for i in range(0, len(unique_labels), BATCH_SIZE):
        batch_labels = unique_labels[i:i + BATCH_SIZE]
        embs, masks = encoder(batch_labels)
        for label, emb, mask in zip(batch_labels, embs, masks):
            seq_len = (~mask).sum().item()
            cache[label] = (emb[:seq_len].cpu(), mask[:seq_len].cpu())
        print(f"  {min(i + BATCH_SIZE, len(unique_labels))}/{len(unique_labels)}")

torch.save(cache, out_path)
print(f"Saved to {out_path}")
