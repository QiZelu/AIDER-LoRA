"""Frame sampling and decoding with the exact AIDER temporal contract."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def anchor_pair_indices(total_frames: int, num_frames: int = 16) -> tuple[list[int], list[int]]:
    """Return T anchors in [0, F-2] and their immediate next frames.

    The implementation intentionally follows the formula in ``pipeline.md``:
    ``round(t * (F - 2) / (T - 1))``. Videos shorter than two frames cannot
    form an adjacent pair. The paper protocol expects F >= 32; callers can
    enforce that separately with ``min_total_frames``.
    """
    if total_frames < 2:
        raise ValueError(f"At least 2 frames are required, got {total_frames}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_frames == 1:
        anchors = [0]
    else:
        anchors = np.rint(
            np.arange(num_frames, dtype=np.float64)
            * float(total_frames - 2)
            / float(num_frames - 1)
        ).astype(np.int64).tolist()
    return anchors, [index + 1 for index in anchors]


def probe_video(video_path: str | Path) -> tuple[int, int, int]:
    """Return ``(frame_count, width, height)`` without decoding the video."""
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, width, height


def read_frames(video_path: str | Path, indices: Sequence[int]) -> np.ndarray:
    """Read exact indices and return RGB uint8 frames in the requested order."""
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames: list[np.ndarray] = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {index} from {path}")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames, axis=0)


def read_anchor_pairs(
    video_path: str | Path,
    num_frames: int = 16,
    min_total_frames: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Decode aligned anchor/next-frame arrays as RGB uint8."""
    total_frames, _, _ = probe_video(video_path)
    if total_frames < min_total_frames:
        raise ValueError(
            f"{video_path} has {total_frames} frames, but the protocol requires "
            f"at least {min_total_frames}"
        )
    anchors, next_frames = anchor_pair_indices(total_frames, num_frames)
    decoded = read_frames(video_path, anchors + next_frames)
    return decoded[:num_frames], decoded[num_frames:], anchors, next_frames
