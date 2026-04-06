import os
import torch
from torch.utils.data import DataLoader
from lewam.datasets.somethingsomethingv2 import SomethingSomethingV2Dataset


def custom_collate(batch):
    labels = [item['label'] for item in batch]
    return {
        'video_id': [item['video_id'] for item in batch],
        'label': torch.tensor(labels) if all(isinstance(l, int) for l in labels) else labels,
        'text': [item['text'] for item in batch],
        'placeholders': [item['placeholders'] for item in batch]
    }


def test_dataloader():
    le_wam_root = os.environ.get('LE_WAM_ROOT')
    if not le_wam_root:
        raise ValueError("LE_WAM_ROOT environment variable not set")

    data_dir = os.path.join(le_wam_root, 'data', 'somethingsomethingv2')

    print("Testing metadata loading...")
    train_dataset = SomethingSomethingV2Dataset(data_dir, split='train')
    val_dataset = SomethingSomethingV2Dataset(data_dir, split='validation')
    test_dataset = SomethingSomethingV2Dataset(data_dir, split='test')

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=custom_collate)
    batch = next(iter(train_loader))
    print(f"\nBatch: video_ids={batch['video_id']}, labels={batch['label']}")

    print("\nTesting video loading (10 samples)...")
    video_dataset = SomethingSomethingV2Dataset(data_dir, split='train', load_videos=True)

    for i in range(10):
        s = video_dataset[i]
        print(f"[{i}] id={s['video_id']} shape={s['video'].shape} label={s['label']}")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_dataloader()
