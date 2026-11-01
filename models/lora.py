"""Task-embedding-modulated shared low-rank channels; no task parameter banks."""
import math

import torch
from torch import nn


class LowRankAdapter(nn.Module):
    """B [(1 + delta(z)) * A dropout(x)], with zero-initialized linear residuals."""
    def __init__(self, in_features, out_features, rank, alpha, dropout, seed, condition_width=768):
        super().__init__()
        self.rank, self.condition_width = rank, condition_width
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # Preserve the old first-task A initialization without affecting base RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.A = nn.Linear(in_features, rank, bias=False)
            self.B = nn.Linear(rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 500000)
            self.gate = nn.Linear(condition_width, rank)
            nn.init.zeros_(self.gate.weight)
            # The Linear predicts a residual; the explicit unit baseline below
            # keeps B learnable even though B and the residual both start at zero.
            nn.init.zeros_(self.gate.bias)

    def channel_residual(self, condition):
        if condition.ndim != 2 or condition.shape[-1] != self.condition_width:
            raise ValueError('Expected task embeddings [B,condition_width]')
        if not torch.isfinite(condition).all():
            raise FloatingPointError('Nonfinite task embedding')
        with torch.autocast(condition.device.type, enabled=False):
            return self.gate(condition.float())

    def channel_weights(self, condition):
        return 1.0 + self.channel_residual(condition)

    def forward(self, features, condition):
        hidden = self.A(self.dropout(features))
        gate = self.channel_weights(condition)
        broadcast = [len(features)] + [1] * (features.ndim - 2) + [self.rank]
        # Keep task conditioning/multiplication in FP32; B uses normal forward AMP.
        modulated = hidden.float() * gate.reshape(broadcast)
        return self.B(modulated) * self.scale


class TaskLoRALinear(nn.Module):
    """One base map and one shared adapter, conditioned by z rather than task ID."""
    def __init__(self, in_features, out_features, tasks, seed, rank=8, alpha=16,
                 dropout=0.05, condition_width=768):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1 or not tasks or len(set(tasks)) != len(tasks):
            raise ValueError('Invalid shared conditional LoRA configuration')
        self.in_features, self.out_features = in_features, out_features
        self.task_order = tuple(tasks)  # Metadata only; never selects parameters.
        self.base = nn.Linear(in_features, out_features)
        self.adapter = LowRankAdapter(in_features, out_features, rank, alpha, dropout,
                                      seed + 1009, condition_width)
        self.enabled = True

    def forward(self, features, condition):
        if features.ndim < 2 or features.shape[-1] != self.in_features:
            raise ValueError('Invalid LoRA input geometry')
        if condition is None or condition.shape != (len(features), self.adapter.condition_width) or condition.device != features.device:
            raise ValueError('Task embeddings must align with the feature batch/device')
        result = self.base(features)
        if not self.enabled:
            return result
        return result + self.adapter(features, condition).to(result.dtype)


def lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, TaskLoRALinear):
            yield name, module


def adapter_items(model):
    for name, module in lora_modules(model):
        yield name + '.adapter', module.adapter
