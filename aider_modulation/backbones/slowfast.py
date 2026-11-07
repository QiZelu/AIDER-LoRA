"""SlowFast adapter based on the verified LAVE implementation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SlowFastEncoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        max_positions: int = 256,
    ) -> None:
        super().__init__()
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"SlowFast checkpoint does not exist: {checkpoint_file}"
            )
        # This trusted local file is a serialized PyTorchVideo model object,
        # rather than a plain state_dict.
        pretrained = torch.load(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        self.feature_extraction = nn.Sequential()
        for index in range(5):
            self.feature_extraction.add_module(str(index), pretrained[index])
        self.fast_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.feature_dim = 256
        self.max_positions = int(max_positions)
        self.is_frozen = True
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.is_frozen = True
        self.feature_extraction.eval()
        for parameter in self.feature_extraction.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extraction.eval()
        return self

    @staticmethod
    def pack_pathways(frames: torch.Tensor) -> list[torch.Tensor]:
        timesteps = frames.shape[2]
        slow_count = max(1, timesteps // 4)
        indices = torch.linspace(
            0, timesteps - 1, slow_count, device=frames.device
        ).long()
        return [frames.index_select(2, indices), frames]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        target_timesteps = frames.shape[2]
        pathways = self.pack_pathways(frames)
        with torch.no_grad():
            features = self.feature_extraction(pathways)
        fast = self.fast_pool(features[1]).squeeze(-1).squeeze(-1)
        fast = fast.permute(0, 2, 1)
        if fast.shape[1] != target_timesteps:
            fast = F.interpolate(
                fast.transpose(1, 2),
                size=target_timesteps,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return fast.contiguous()
