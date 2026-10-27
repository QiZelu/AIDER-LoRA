"""Load the single baseline YAML and optional dotted command-line overrides."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_BASELINE = "film_transformer"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def _apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, got: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    value = yaml.safe_load(raw_value)
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(path)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = root / config_path
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    for expression in overrides or []:
        _apply_override(config, expression)
    path_keys = {
        "train_manifest", "val_manifest", "test_manifest", "video_root",
        "cache_dir", "checkpoint_path", "model_dir", "bert_path",
        "med_config", "output_dir", "resume", "checkpoint", "initialization",
    }
    def resolve_paths(value, section=()):
        for key, item in value.items():
            if isinstance(item, dict):
                resolve_paths(item, section+(key,))
            elif key in path_keys and isinstance(item, str) and item:
                # Experiment metadata also uses 'initialization' for a method
                # label (e.g. seed42_untrained), not a filesystem location.
                if key == 'initialization' and section != ('training',):
                    continue
                p = Path(item).expanduser()
                value[key] = str(p if p.is_absolute() else root / p)
    resolve_paths(config)
    config["_config_path"] = str(config_path)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
