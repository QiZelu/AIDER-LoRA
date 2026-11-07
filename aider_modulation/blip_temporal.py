"""Temporal self-attention on BLIP's eight cross-modal CLS tokens."""
import math
import torch
from torch import nn


class BLIPTemporalSelfAttention(nn.Module):
    width = 768
    frames = 8

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(self.width)
        self.attention = nn.MultiheadAttention(self.width, 8, dropout=0., batch_first=True)
        position = torch.arange(self.frames, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0,self.width,2,dtype=torch.float32)*(-math.log(10000.)/self.width))
        encoding = torch.empty(self.frames,self.width,dtype=torch.float32)
        encoding[:,0::2] = torch.sin(position*frequency)
        encoding[:,1::2] = torch.cos(position*frequency)
        self.register_buffer('position_encoding',encoding[None],persistent=True)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.zeros_(self.attention.in_proj_bias)

    def forward(self,x):
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1:] != (self.frames,self.width):
            raise ValueError('Expected nonempty BLIP features [B,8,768]')
        values = self.norm(x.float())
        query_key = values+self.position_encoding
        return self.attention(query_key,query_key,values,need_weights=False)[0]
