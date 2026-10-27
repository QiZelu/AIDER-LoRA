"""Local VITAL/InternVision backbone adapter."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .configuration_intern_vit import InternVisionConfig
from .modeling_intern_vit import InternVisionModel


class VITALVisionEncoder(nn.Module):
    """Original baseline: frozen InternViT patches without Pixel Shuffle.

    At 448px, retain the 32x32 spatial grid and 1024D patch features.
    AIDER's trainable technical projectors map each raw patch to 512D.
    """

    def __init__(
        self,
        model_dir: str,
        dtype: str = "bfloat16",
        frame_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        directory = Path(model_dir)
        if not directory.exists():
            raise FileNotFoundError(f"VITAL directory does not exist: {directory}")
        with (directory / "config.json").open("r", encoding="utf-8") as handle:
            full_config = json.load(handle)
        self.select_layer = int(full_config.get("select_layer", -1))
        vision_config = InternVisionConfig(**full_config["vision_config"])
        vision_config.use_flash_attn = False
        self.vision_model = InternVisionModel(vision_config)
        target_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self.vision_model.to(dtype=target_dtype)

        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError("VITAL loading requires the safetensors package") from error
        checkpoint = load_file(str(directory / "model.safetensors"), device="cpu")
        vision_state = {
            key.removeprefix("vision_model."): value
            for key, value in checkpoint.items()
            if key.startswith("vision_model.")
        }
        self.vision_model.load_state_dict(vision_state, strict=True)
        self.feature_dim = int(vision_config.hidden_size)
        grid = int(vision_config.image_size) // int(vision_config.patch_size)
        self.patch_grid = (grid, grid)
        self.frame_chunk_size = int(frame_chunk_size)
        if self.frame_chunk_size <= 0:
            raise ValueError("frame_chunk_size must be positive")
        self.vision_model.eval()
        self.vision_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        return self

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch, timesteps, channels, height, width = frames.shape
        flattened = frames.reshape(batch * timesteps, channels, height, width)
        patch_outputs = []
        parameter_dtype = next(self.vision_model.parameters()).dtype
        with torch.no_grad():
            for chunk in flattened.split(self.frame_chunk_size, dim=0):
                output = self.vision_model(
                    pixel_values=chunk.to(dtype=parameter_dtype),
                    output_hidden_states=self.select_layer != -1,
                    return_dict=True,
                )
                encoded = (
                    output.last_hidden_state
                    if self.select_layer == -1
                    else output.hidden_states[self.select_layer]
                )
                patches = encoded[:, 1:, :]
                grid_size = int(round(patches.shape[1] ** 0.5))
                if grid_size * grid_size != patches.shape[1]:
                    raise RuntimeError(
                        f"VITAL returned non-square patch count: {patches.shape[1]}"
                    )
                patch_grid = (grid_size, grid_size)
                patch_outputs.append(patches)
        patches = torch.cat(patch_outputs, dim=0)
        num_patches = patches.shape[1]
        return (
            patches.reshape(batch, timesteps, num_patches, -1),
            patch_grid,
        )
