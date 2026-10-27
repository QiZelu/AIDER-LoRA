"""Single-channel artifact prior extraction and optional on-disk caching."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .artifact_features_v3 import compute_mc_residual_v3, percentile_normalize


ARTIFACT_IMPLEMENTATION = "artifact_features_v3_gaussian_before_normalize_v1"


class ArtifactPriorExtractor:
    """Extract the raw V3 texture-instability evidence for aligned frame pairs."""

    def __init__(
        self,
        normalization: str = "percentile",
        percentile_lo: float = 1.0,
        percentile_hi: float = 99.0,
        fixed_scale: float = 0.1,
        gaussian_kernel_size: int = 9,
        gaussian_sigma: float = 0.0,
        **extractor_kwargs: Any,
    ) -> None:
        if normalization not in {"percentile", "fixed", "none"}:
            raise ValueError(f"Unsupported artifact normalization: {normalization}")
        if fixed_scale <= 0:
            raise ValueError("fixed_scale must be positive")
        if gaussian_kernel_size <= 0 or gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be a positive odd integer")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")
        self.normalization = normalization
        self.percentile_lo = percentile_lo
        self.percentile_hi = percentile_hi
        self.fixed_scale = fixed_scale
        self.gaussian_kernel_size = int(gaussian_kernel_size)
        self.gaussian_sigma = float(gaussian_sigma)
        self.extractor_kwargs = extractor_kwargs

    def _normalize(self, artifact: np.ndarray) -> np.ndarray:
        if self.normalization == "percentile":
            return percentile_normalize(
                artifact, lo=self.percentile_lo, hi=self.percentile_hi
            )
        if self.normalization == "fixed":
            return np.clip(artifact / self.fixed_scale, 0.0, 1.0).astype(np.float32)
        return np.clip(artifact, 0.0, 1.0).astype(np.float32)

    def __call__(
        self, anchor_frames_rgb: np.ndarray, next_frames_rgb: np.ndarray
    ) -> np.ndarray:
        if anchor_frames_rgb.shape != next_frames_rgb.shape:
            raise ValueError(
                "Anchor and next-frame arrays must have identical shapes, got "
                f"{anchor_frames_rgb.shape} and {next_frames_rgb.shape}"
            )
        maps = []
        for anchor_rgb, next_rgb in zip(anchor_frames_rgb, next_frames_rgb):
            anchor_bgr = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2BGR)
            next_bgr = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2BGR)
            result = compute_mc_residual_v3(
                anchor_bgr, next_bgr, **self.extractor_kwargs
            )
            artifact = cv2.GaussianBlur(
                result["A_tex"],
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                sigmaX=self.gaussian_sigma,
                sigmaY=self.gaussian_sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )
            maps.append(self._normalize(artifact))
        return np.stack(maps, axis=0).astype(np.float32)

    def fingerprint(self) -> str:
        values = (
            ARTIFACT_IMPLEMENTATION,
            self.normalization,
            self.percentile_lo,
            self.percentile_hi,
            self.fixed_scale,
            self.gaussian_kernel_size,
            self.gaussian_sigma,
            sorted(self.extractor_kwargs.items()),
        )
        return hashlib.sha1(repr(values).encode("utf-8")).hexdigest()[:12]


class ArtifactCache:
    """Atomic ``.npy`` cache keyed by video identity and extraction settings."""

    def __init__(self, root: str | Path | None, readonly_manifest=None) -> None:
        self.root = Path(root).expanduser() if root else None
        self.readonly = None
        if readonly_manifest is not None:
            import json
            self.readonly = json.loads(Path(readonly_manifest).read_text())['files']

    def path_for(
        self,
        video_path: str | Path,
        num_frames: int,
        image_size: int,
        extractor: ArtifactPriorExtractor,
    ) -> Path | None:
        if self.root is None:
            return None
        video = Path(video_path).resolve()
        stat = video.stat()
        identity = (
            f"{video}|{stat.st_size}|{stat.st_mtime_ns}|{num_frames}|"
            f"{image_size}|{extractor.fingerprint()}"
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
        return self.root / digest[:2] / f"{digest}.npy"

    def load(self, path: Path | None) -> np.ndarray | None:
        if self.readonly is not None:
            import io
            if path is None or str(path.resolve()) not in self.readonly:
                raise ValueError('Cache path absent from read-only audit: ' + str(path))
            expected = self.readonly[str(path.resolve())]
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('Read-only Prior cache changed: ' + str(path))
            value = np.load(io.BytesIO(data), allow_pickle=False)
            if (list(value.shape) != expected['shape'] or value.dtype != np.float32
                    or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1):
                raise ValueError('Invalid read-only Prior cache: ' + str(path))
            return value.astype(np.float32, copy=False)
        if path is None or not path.exists():
            return None
        return np.load(path, allow_pickle=False).astype(np.float32, copy=False)

    def save(self, path: Path | None, value: np.ndarray) -> None:
        if self.readonly is not None:
            raise RuntimeError('Read-only cache cannot be rebuilt')
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
        np.save(temp, value.astype(np.float32), allow_pickle=False)
        os.replace(temp, path)
