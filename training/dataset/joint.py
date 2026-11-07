"""Each record is one independent video/task pair, never a four-score head."""

import math
from torch.utils.data import Dataset
from .video_dataset import VideoQualityDataset, VideoInputProcessor


class PairDataset(Dataset):
    def __init__(self, config, split):
        data = config['data']
        processor = VideoInputProcessor(data)
        self.order = config['task']['order']
        self.tasks = [
            VideoQualityDataset(
                data[f'{split}_manifest'],
                task,
                config['task']['descriptions'][task],
                processor,
                data.get('video_root'),
            )
            for task in self.order
        ]
        self.video_count = len(self.tasks[0])
        ids = [r['id'] for r in self.tasks[0].records]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate video IDs')
        for dataset in self.tasks:
            if [r['id'] for r in dataset.records] != ids:
                raise ValueError('Different task video ordering')
            if any(
                not math.isfinite(float(r['labels'][dataset.label_key])) for r in dataset.records
            ):
                raise ValueError('Invalid MOS')

    def __len__(self):
        return len(self.order) * self.video_count

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        task, video = divmod(index, self.video_count)
        result = self.tasks[task][video]
        result.update(task_id=task, pair_index=index)
        return result
