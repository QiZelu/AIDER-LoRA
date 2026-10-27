"""Shared linear maps with explicitly selected, independent task LoRAs."""
import math

import torch
from torch import nn


class LowRankAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout, seed):
        super().__init__()
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.A = nn.Linear(in_features, rank, bias=False)
            self.B = nn.Linear(rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)

    def forward(self, features):
        return self.B(self.A(self.dropout(features))) * self.scale


class TaskLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, tasks, seed, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1 or not tasks or len(set(tasks)) != len(tasks):
            raise ValueError('Invalid task LoRA configuration')
        self.in_features, self.out_features = in_features, out_features
        self.task_order = tuple(tasks)
        self.base = nn.Linear(in_features, out_features)
        # Adapter construction never changes the shared-base initialization RNG.
        self.adapters = nn.ModuleDict({
            task: LowRankAdapter(in_features, out_features, rank, alpha, dropout,
                                 seed + 1009 * (index + 1))
            for index, task in enumerate(self.task_order)
        })
        self.enabled = True

    def forward(self, features, task_ids):
        result = self.base(features)
        if not self.enabled:
            return result
        # Only present tasks execute. Absent gradients stay None, preventing
        # both Adam moment updates and decoupled weight decay for those tasks.
        values, positions = [], []
        for index, task in enumerate(self.task_order):
            selected = torch.where(task_ids == index)[0]
            if selected.numel():
                values.append(self.adapters[task](features.index_select(0, selected)))
                positions.append(selected)
        delta = torch.cat(values)[torch.cat(positions).argsort()]
        return result + delta.to(result.dtype)


def lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, TaskLoRALinear):
            yield name, module


def task_adapter_items(model):
    for name, module in lora_modules(model):
        for task, adapter in module.adapters.items():
            yield f'{name}.adapters.{task}', task, adapter
