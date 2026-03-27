import os
import json
import zipfile
import tarfile
import io
import av
import torch
import numpy as np
from torch.utils.data import Dataset


class SomethingSomethingV2Dataset(Dataset):
    def __init__(self, data_dir, split='train', load_videos=False, num_frames=None, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.load_videos = load_videos
        self.num_frames = num_frames
        self.transform = transform

        labels_zip = os.path.join(data_dir, 'labels.zip')
        with zipfile.ZipFile(labels_zip, 'r') as zf:
            with zf.open(f'labels/{split}.json') as f:
                self.annotations = json.load(f)

        if load_videos:
            self.tar_path = os.path.join(data_dir, '20bn-something-something-v2.tar')
            self.tar = tarfile.open(self.tar_path, 'r')
            self.video_dict = {
                m.name.split('/')[-1].replace('.webm', ''): m
                for m in self.tar.getmembers()
                if m.name.endswith('.webm')
            }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_id = str(annotation['id'])

        result = {
            'video_id': video_id,
            'label': annotation.get('label', -1),
            'text': annotation['template'].replace('[', '').replace(']', ''),
            'placeholders': annotation.get('placeholders', [])
        }

        if self.load_videos and video_id in self.video_dict:
            member = self.video_dict[video_id]
            video_bytes = self.tar.extractfile(member).read()

            container = av.open(io.BytesIO(video_bytes))
            frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
            container.close()

            if frames:
                video = torch.from_numpy(np.stack(frames))  # (T, H, W, C)
                if self.num_frames is not None:
                    if video.shape[0] < self.num_frames:
                        return None
                    indices = torch.linspace(0, video.shape[0] - 1, self.num_frames).long()
                    video = video[indices]
                result['video'] = video.permute(0, 3, 1, 2)  # (T, C, H, W)

        return result

    def collate_fn(self, batch):
        valid = [item for item in batch if item is not None and 'video' in item]
        if not valid:
            return None
        videos = [self.transform(item['video']) if self.transform else item['video'] for item in valid]
        return {
            'video': torch.stack(videos),
            'text':  [item['text'] for item in valid],
            'label': [item['label'] for item in valid],
        }
