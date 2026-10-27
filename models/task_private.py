"""任务私有 Linear：显式 task_id 分流，缺席任务不参与计算。"""

import copy
import torch
from torch import nn
from .registry import ATTRIBUTES


class TaskPrivateLinear(nn.Module):
    def __init__(self, source, task_order):
        super().__init__()
        self.task_order = tuple(task_order)
        self.in_features, self.out_features = source.in_features, source.out_features
        self.by_task = nn.ModuleDict({task: copy.deepcopy(source) for task in task_order})

    def forward(self, features, task_ids):
        from .registry import validate_task_ids
        if task_ids is None:
            raise ValueError('Private Linear requires explicit task_ids')
        validate_task_ids(task_ids, len(features), len(self.task_order))
        if features.ndim != 2 or features.shape[1] != self.in_features:
            raise ValueError('Invalid private Linear input shape')
        outputs, positions = [], []
        for index, task in enumerate(self.task_order):
            selected = torch.where(task_ids == index)[0]
            if selected.numel():
                outputs.append(self.by_task[task](features.index_select(0, selected)))
                positions.append(selected)
        return torch.cat(outputs)[torch.cat(positions).argsort()]


def task_linear(module, features, task_ids=None):
    return module(features, task_ids) if isinstance(module, TaskPrivateLinear) else module(features)


def privatize_task_modules(model):
    """深复制原未训练模块，保持参数值、注册位置及构造 RNG。"""
    for attribute in ATTRIBUTES:
        film = getattr(model, attribute)
        film.affine = TaskPrivateLinear(film.affine, model.task_order)
    model.alpha_gate = TaskPrivateLinear(model.alpha_gate, model.task_order)
    model.private_task_modules = True


def private_module_items(model):
    for name, module in model.named_modules():
        if isinstance(module, TaskPrivateLinear):
            for task, linear in module.by_task.items():
                yield name + '.' + task, task, linear
    if getattr(model, 'private_task_modules', False):
        for task, head in model.heads():
            yield 'head.' + task, task, head
