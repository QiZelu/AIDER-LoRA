"""BLIP-Large text-video encoder backed by AIDER's vendored BLIP modules."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


FP32_TEXT_MODE = "fp32_master_bf16_embedding_outputs"


def bf16_embedding_output(module, inputs, output):
    """Match the validated FP32-master experiment's embedding forward path."""
    return output.to(torch.bfloat16)


def promote_text_parameters(encoder: nn.Module) -> None:
    """Preserve loaded values, but accumulate gradients/Adam state in FP32."""
    encoder.float()
    for module in encoder.modules():
        if isinstance(module, nn.Embedding) and not getattr(module, "_aider_bf16_output", False):
            module.register_forward_hook(bf16_embedding_output)
            module._aider_bf16_output = True


def tokenize_prompts(tokenizer, prompts, max_length):
    """Baseline fixed-length padding, with tokenizer-managed truncation."""
    encoded = tokenizer(prompts, padding=False, truncation=True,
                        max_length=max_length, add_special_tokens=True)
    return tokenizer.pad(encoded, padding="max_length", max_length=max_length,
                         return_tensors="pt")


def _load_required_checkpoint_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    checkpoint_path: str | Path,
) -> int:
    """Load every runtime tensor and ignore pretraining-only checkpoint heads."""
    required = model.state_dict()
    missing = sorted(set(required) - set(state))
    mismatched = sorted(
        name
        for name, value in required.items()
        if name in state and state[name].shape != value.shape
    )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched[:8]}")
        raise RuntimeError(
            f"BLIP checkpoint {checkpoint_path} is incompatible: "
            + ", ".join(details)
        )
    model.load_state_dict(
        {name: state[name] for name in required},
        strict=True,
    )
    return len(state) - len(required)


class BLIPTextVideoEncoder(nn.Module):
    """Frame-wise BLIP cross-modal text features.

    Despite the historical "BLIP2" label, the supplied checkpoint uses
    ``BLIP_Pretrain`` without a Q-Former.
    """

    def __init__(
        self,
        checkpoint_path: str,
        bert_path: str,
        med_config: str,
        image_size: int = 224,
        dtype: str = "bfloat16",
        train_text_encoder: bool = True,
        text_encoder_gradient_checkpointing: bool = True,
        max_text_length: int = 50,
        text_padding: str = "max_length",
        text_parameter_precision: str = FP32_TEXT_MODE,
    ) -> None:
        super().__init__()
        if text_parameter_precision != FP32_TEXT_MODE:
            raise ValueError(f"Unsupported text_parameter_precision: {text_parameter_precision}")
        if dtype != "bfloat16":
            raise ValueError("FP32 text master mode requires bfloat16 backbone/autocast")
        for path in (Path(checkpoint_path), Path(bert_path), Path(med_config)):
            if not path.exists():
                raise FileNotFoundError(f"BLIP dependency does not exist: {path}")

        from .blip_legacy import BLIP_Pretrain

        self.model = BLIP_Pretrain(
            image_size=image_size,
            bert_path=bert_path,
            vit="large",
            med_config=med_config,
        )
        target_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self.model.to(dtype=target_dtype)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (TypeError, RuntimeError):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        if not isinstance(state, dict):
            raise TypeError("BLIP checkpoint model state must be a mapping")
        self.ignored_checkpoint_tensors = _load_required_checkpoint_state(
            self.model,
            state,
            checkpoint_path,
        )
        self.text_parameter_precision = text_parameter_precision
        # Preserve the selected pretrained values: BF16 load, then FP32 master.
        promote_text_parameters(self.model.text_encoder)
        self.feature_dim = int(self.model.text_encoder.config.hidden_size)
        self.max_text_length = int(max_text_length)
        max_position_embeddings = int(
            self.model.text_encoder.config.max_position_embeddings
        )
        if isinstance(max_text_length, bool) or self.max_text_length != max_text_length or not 2 <= self.max_text_length <= max_position_embeddings:
            raise ValueError(
                "max_text_length must be an integer in [2, "
                f"{max_position_embeddings}], got {self.max_text_length}"
            )
        if text_padding != "max_length":
            raise ValueError("These baselines require fixed padding")
        if not isinstance(train_text_encoder, bool):
            raise TypeError("train_text_encoder must be a bool")
        self.train_text_encoder = train_text_encoder
        self.text_padding = "max_length"
        self.text_encoder_gradient_checkpointing = bool(text_encoder_gradient_checkpointing)
        self.configure_trainable_parameters()

    def _tokenize_prompts(self, prompts, device):
        return tokenize_prompts(self.model.tokenizer, prompts, self.max_text_length).to(device)

    def configure_trainable_parameters(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.text_encoder.requires_grad_(self.train_text_encoder)
        self.model.text_encoder.encoder.gradient_checkpointing = (
            self.train_text_encoder and self.text_encoder_gradient_checkpointing
        )
        if not self.train_text_encoder:
            self.train(False)

    def train(self, mode: bool = True):
        super().train(mode if self.train_text_encoder else False)
        self.model.eval()
        self.model.text_encoder.train(mode and self.train_text_encoder)
        return self

    def forward(self, frames: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        # no_grad (not inference_mode): frozen features feed trainable adapters.
        # Keep the caller's no_grad intact for evaluation of the fine-tuned path.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_text_encoder):
            return self._forward_features(frames, prompts)

    def _forward_features(self, frames: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        batch, timesteps = frames.shape[:2]
        if len(prompts) != batch:
            raise ValueError(f"Expected {batch} prompts, got {len(prompts)}")
        text = self._tokenize_prompts(prompts, frames.device)
        parameter_dtype = next(self.model.parameters()).dtype
        outputs = []
        for timestep in range(timesteps):
            with torch.no_grad():
                image_embeddings = self.model.visual_encoder(
                    frames[:, timestep].to(dtype=parameter_dtype)
                )
            image_attention = torch.ones(
                image_embeddings.shape[:-1],
                dtype=torch.long,
                device=frames.device,
            )
            encoded = self.model.text_encoder(
                text.input_ids, attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeddings,
                encoder_attention_mask=image_attention,
                return_dict=True,
            )
            outputs.append(encoded.last_hidden_state[:, 0, :])
        return torch.stack(outputs, dim=1)
