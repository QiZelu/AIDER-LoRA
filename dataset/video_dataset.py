"""Manifest dataset and shared input preprocessing for AIDER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .artifact_prior import ArtifactCache, ArtifactPriorExtractor
from .frame_sampling import read_anchor_pairs


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    if manifest.suffix.lower() == ".jsonl":
        records = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON at {manifest}:{line_number}: {error}"
                        ) from error
        return records
    with manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("records"))
    if not isinstance(payload, list):
        raise ValueError("JSON manifest must be a list or contain data/records")
    return payload


def _center_crop_resize(frames: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize followed by a shared center crop."""
    output = []
    for frame in frames:
        height, width = frame.shape[:2]
        scale = float(size) / float(min(height, width))
        resized_width = max(size, int(round(width * scale)))
        resized_height = max(size, int(round(height * scale)))
        resized = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC
        )
        x0 = (resized_width - size) // 2
        y0 = (resized_height - size) // 2
        output.append(resized[y0 : y0 + size, x0 : x0 + size])
    return np.stack(output, axis=0)


def _to_normalized_tensor(
    frames: np.ndarray,
    mean: Iterable[float] = IMAGENET_MEAN,
    std: Iterable[float] = IMAGENET_STD,
) -> torch.Tensor:
    tensor = torch.from_numpy(frames.copy()).float().div_(255.0)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    mean_tensor = torch.tensor(tuple(mean), dtype=tensor.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(tuple(std), dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


class VideoInputProcessor:
    """Decode one video into all aligned backbone inputs and artifact priors."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.num_frames = int(config.get("num_frames", 8))
        self.min_total_frames = int(config.get("min_total_frames", 8))
        self.vital_size = int(config.get("vital_size", 448))
        self.expert_size = int(config.get("expert_size", 224))
        artifact_config = dict(config.get("artifact", {}))
        cache_root = artifact_config.pop("cache_dir", None)
        readonly_manifest = artifact_config.pop("readonly_manifest", None)
        self.artifact_extractor = ArtifactPriorExtractor(**artifact_config)
        self.artifact_cache = ArtifactCache(cache_root, readonly_manifest=readonly_manifest)

    def __call__(self, video_path: str | Path, prompt: str) -> dict[str, Any]:
        anchors, next_frames, anchor_indices, next_indices = read_anchor_pairs(
            video_path,
            num_frames=self.num_frames,
            min_total_frames=self.min_total_frames,
        )

        # Artifact maps and VITAL patch tokens always see identical geometry.
        pair_frames = np.concatenate([anchors, next_frames], axis=0)
        vital_pair = _center_crop_resize(pair_frames, self.vital_size)
        vital_anchors = vital_pair[: self.num_frames]
        vital_next = vital_pair[self.num_frames :]

        cache_path = self.artifact_cache.path_for(
            video_path,
            self.num_frames,
            self.vital_size,
            self.artifact_extractor,
        )
        artifact_prior = self.artifact_cache.load(cache_path)
        if artifact_prior is None:
            artifact_prior = self.artifact_extractor(vital_anchors, vital_next)
            self.artifact_cache.save(cache_path, artifact_prior)

        expert_anchors = _center_crop_resize(anchors, self.expert_size)
        vital_tensor = _to_normalized_tensor(vital_anchors)
        expert_tensor = _to_normalized_tensor(expert_anchors)
        return {
            "vital_frames": vital_tensor,
            "blip_frames": expert_tensor,
            "slowfast_frames": expert_tensor.permute(1, 0, 2, 3).contiguous(),
            "artifact_prior": torch.from_numpy(artifact_prior).float(),
            "prompt": prompt,
            "anchor_indices": torch.tensor(anchor_indices, dtype=torch.long),
            "next_indices": torch.tensor(next_indices, dtype=torch.long),
        }


class VideoQualityDataset(Dataset):
    """JSON/JSONL video-quality manifest.

    Each record follows this compact schema::

        {
          "video": "relative/or/absolute.mp4",
          "prompt": "generation prompt",
          "dataset": "aigve",
          "labels": {"technical": 0.72, "temporal": 0.64, "overall": 0.68}
        }
    """

    def __init__(
        self,
        manifest: str | Path,
        label_key: str,
        dimension_text: str,
        processor: VideoInputProcessor,
        video_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.records = _read_manifest(self.manifest_path)
        self.label_key = label_key.strip()
        self.dimension_text = dimension_text.strip()
        if not self.label_key:
            raise ValueError("task.label_key must be non-empty")
        if not self.dimension_text:
            raise ValueError("task.dimension must be non-empty")
        self.processor = processor
        self.video_root = (
            Path(video_root)
            if video_root
            else self.manifest_path.resolve().parent
        )
        self._validate_target_labels()

    @staticmethod
    def _labels_source(record: dict[str, Any]) -> dict[str, Any]:
        labels_source = record.get("labels")
        if labels_source is None and "dimension" in record and "label" in record:
            labels_source = {record["dimension"]: record["label"]}
        return labels_source or {}

    def _validate_target_labels(self) -> None:
        missing = []
        for index, record in enumerate(self.records):
            value = self._labels_source(record).get(self.label_key)
            if value is None:
                missing.append(index)
                if len(missing) == 5:
                    break
        if missing:
            raise ValueError(
                f"Manifest {self.manifest_path} does not provide the explicitly "
                f"selected label_key '{self.label_key}' for record "
                f"indices {missing}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _video_path(self, record: dict[str, Any]) -> Path:
        value = record.get("video", record.get("filename"))
        if not value:
            raise KeyError("Every manifest record needs 'video' or 'filename'")
        path = Path(value)
        return path if path.is_absolute() else self.video_root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        video_path = self._video_path(record)
        prompt = str(record.get("prompt", record.get("caption", "")))
        sample = self.processor(video_path, prompt)

        labels_source = self._labels_source(record)
        value = labels_source.get(self.label_key)
        target = torch.tensor(float(value), dtype=torch.float32)
        if not torch.isfinite(target):
            raise ValueError(
                f"Record {index} has no finite '{self.label_key}' label"
            )

        sample.update(
            {
                "target": target,
                "label_key": self.label_key,
                "dimension_text": self.dimension_text,
                "dataset_name": str(record.get("dataset", "default")),
                "video_id": str(record.get("id", video_path.stem)),
                "video_path": str(video_path),
            }
        )
        return sample
