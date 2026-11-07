"""Shared residual QK channel weights; no low-rank adapters."""
import torch
from torch import nn
from .support import digest


class QKChannelGate(nn.Module):
    def __init__(self, seed, condition_width=768, channels=128):
        super().__init__()
        self.condition_width, self.channels = condition_width, channels
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.linear = nn.Linear(condition_width, channels)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        self.enabled = True
        self.initial_state_hash = self.state_hash()

    def state_hash(self):
        return digest(self.state_dict())

    def residual(self, condition):
        if condition.ndim != 2 or condition.shape[1] != self.condition_width:
            raise ValueError('Expected task embedding [B,768]')
        with torch.autocast(condition.device.type, enabled=False):
            value = self.linear(condition.float())
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite QK gate')
        return value

    def forward(self, q, condition):
        if q.ndim != 4 or q.shape[0] != len(condition) or q.shape[-1] != self.channels:
            raise ValueError('Expected Q [B,heads,tokens,128] and aligned task condition')
        if not self.enabled:
            return q
        with torch.autocast(q.device.type, enabled=False):
            gated = q.float() * (1.0 + self.residual(condition))[:, None, None, :]
        return gated.to(q.dtype)


def gate_items(model):
    for name, module in model.named_modules():
        if isinstance(module, QKChannelGate):
            yield name, module
