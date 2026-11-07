"""Epoch-local random permutation over independent video/task pairs."""

import hashlib
import json
import math
import torch
from torch.utils.data import Sampler


class MixedPairSampler(Sampler):
    def __init__(self, count, seed=9042, batch_size=8):
        self.count, self.seed, self.batch_size, self.epoch = count, seed, batch_size, 0

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('Negative epoch')
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.count / self.batch_size)

    def batches(self):
        rng = torch.Generator(device='cpu').manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.count, generator=rng).tolist()
        return [order[i : i + self.batch_size] for i in range(0, self.count, self.batch_size)]

    def __iter__(self):
        return iter(self.batches())

    def digest(self):
        return hashlib.sha256(
            json.dumps(self.batches(), separators=(',', ':')).encode()
        ).hexdigest()


class HomogeneousTaskSampler(MixedPairSampler):
    """逐任务打乱，每轮次随机排列任务，最后四批各保留五条。"""

    def __init__(self, videos_per_task=461, task_count=4, seed=9042, batch_size=8):
        if videos_per_task < 1 or task_count < 1 or batch_size < 2:
            raise ValueError('Invalid homogeneous sampler sizes')
        super().__init__(videos_per_task * task_count, seed, batch_size)
        self.videos_per_task, self.task_count = videos_per_task, task_count

    def __len__(self):
        return self.task_count * math.ceil(self.videos_per_task / self.batch_size)

    def batches(self):
        rng = torch.Generator(device='cpu').manual_seed(self.seed + self.epoch)
        by_task = []
        for task in range(self.task_count):
            order = torch.randperm(self.videos_per_task, generator=rng).tolist()
            by_task.append([[task * self.videos_per_task + i for i in order[start:start+self.batch_size]]
                            for start in range(0, self.videos_per_task, self.batch_size)])
        return [by_task[task][step]
                for step in range(len(by_task[0]))
                for task in torch.randperm(self.task_count, generator=rng).tolist()]
