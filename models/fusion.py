"""Two ordinary Pre-LN Transformer blocks with task-specific projection LoRAs."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .lora import TaskLoRALinear, lora_modules
from .registry import TASKS, validate_task_ids


class TaskSelfAttention(nn.Module):
    def __init__(self, width, heads, tasks, seed):
        super().__init__()
        if width % heads:
            raise ValueError('Attention width must divide into heads')
        self.heads, self.head_dim = heads, width // heads
        for index, name in enumerate(('q_proj', 'k_proj', 'v_proj', 'out_proj')):
            setattr(self, name, TaskLoRALinear(width, width, tasks, seed + index * 10000))
            projection = getattr(self, name).base
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, features, task_ids):
        batch, tokens, width = features.shape
        projections = [getattr(self, name)(features, task_ids).reshape(
            batch, tokens, self.heads, self.head_dim).transpose(1, 2)
            for name in ('q_proj', 'k_proj', 'v_proj')]
        attention = F.scaled_dot_product_attention(*projections, dropout_p=0.0)
        attention = attention.transpose(1, 2).contiguous().reshape(batch, tokens, width)
        return self.out_proj(attention, task_ids)


class TaskTransformerBlock(nn.Module):
    def __init__(self, seed, layer_index, width=1536, heads=12, tasks=TASKS):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 810000 + layer_index)
            self.norm1 = nn.LayerNorm(width)
            self.attention = TaskSelfAttention(width, heads, tasks, seed + 910000 + layer_index * 100000)
            self.norm2 = nn.LayerNorm(width)
            self.fc1 = TaskLoRALinear(width, width, tasks, seed + 950000 + layer_index * 100000)
            self.fc2 = TaskLoRALinear(width, width, tasks, seed + 960000 + layer_index * 100000)

    def forward(self, features, task_ids):
        features = features + self.attention(self.norm1(features.float()), task_ids)
        hidden = F.gelu(self.fc1(self.norm2(features.float()), task_ids))
        return features + self.fc2(hidden, task_ids)


class TransformerFusion(nn.Module):
    def __init__(self, seed, position_encoding, depth=2, width=1536, heads=12, tasks=TASKS):
        super().__init__()
        self.task_order = tuple(tasks)
        self.blocks = nn.ModuleList([
            TaskTransformerBlock(seed, layer_index, width, heads, tasks)
            for layer_index in range(depth)])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 300000)
            head = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.uniform_(layer.weight, -0.1, 0.1)
                    nn.init.zeros_(layer.bias)
        self.heads = nn.ModuleDict({'shared': head})
        self.final_norm = nn.LayerNorm(width)
        self.use_checkpoint = True
        self.phase = 'joint'
        self.register_buffer('position_encoding', position_encoding.clone())

    def configure_phase(self, phase):
        if phase not in ('joint', 'dense', 'adapter'):
            raise ValueError('Unknown Transformer/LoRA phase')
        self.phase = phase
        # Heads are outside the shared fusion base and stay trainable.
        for name, parameter in self.named_parameters():
            if not name.startswith('heads.'):
                parameter.requires_grad_((phase != 'dense') if '.adapters.' in name else (phase != 'adapter'))
        for _, module in lora_modules(self):
            module.enabled = phase != 'dense'

    def base_parameters(self):
        return ((name, parameter) for name, parameter in self.named_parameters()
                if not name.startswith('heads.') and '.adapters.' not in name)

    def forward(self, features, condition=None, task_ids=None):
        if task_ids is None or task_ids.device != features.device:
            raise ValueError('Explicit task IDs on feature device are required')
        validate_task_ids(task_ids, len(features), len(self.task_order))
        features = features + self.position_encoding
        for block in self.blocks:
            if self.training and torch.is_grad_enabled() and self.use_checkpoint:
                features = checkpoint(block, features, task_ids, use_reentrant=False)
            else:
                features = block(features, task_ids)
        return self.final_norm(features.float()), {}
