"""Motion-compensated, unmasked texture-instability artifact features."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    flow=None,
) -> np.ndarray:
    previous_u8 = (np.clip(previous_gray, 0, 1) * 255).astype(np.uint8)
    current_u8 = (np.clip(current_gray, 0, 1) * 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(
        previous_u8,
        current_u8,
        flow,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def standardize_relative(
    artifact: np.ndarray,
    z_clip: float = 3.0,
) -> np.ndarray:
    """Map positive per-frame z-scores to [0, 1] without percentiles."""
    artifact = np.asarray(artifact, dtype=np.float32)
    finite = np.isfinite(artifact)
    if not np.any(finite):
        return np.zeros_like(artifact, dtype=np.float32)
    values = artifact[finite]
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    if standard_deviation < 1e-10 or z_clip <= 0:
        return np.zeros_like(artifact, dtype=np.float32)
    standardized = np.zeros_like(artifact, dtype=np.float32)
    standardized[finite] = (values - mean) / standard_deviation
    return np.clip(standardized / z_clip, 0, 1).astype(np.float32)


def percentile_normalize(
    artifact: np.ndarray,
    lo: float = 1.0,
    hi: float = 99.0,
) -> np.ndarray:
    """Compatibility wrapper for the per-frame z-score visualization."""
    del lo, hi
    return standardize_relative(artifact)


def _gray01(frame_bgr: np.ndarray) -> np.ndarray:
    return (
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        / 255.0
    )


def _dog_bands(gray: np.ndarray) -> dict[str, np.ndarray]:
    blurred_1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
    blurred_2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
    blurred_4 = cv2.GaussianBlur(gray, (0, 0), sigmaX=4.0, sigmaY=4.0)
    return {
        "H1": gray - blurred_1,
        "H2": blurred_1 - blurred_2,
        "H3": blurred_2 - blurred_4,
        "Llow": blurred_4,
    }


def _warp_backward(
    source: np.ndarray,
    target_to_source_flow: np.ndarray,
) -> np.ndarray:
    height, width = target_to_source_flow.shape[:2]
    y_coordinates, x_coordinates = np.mgrid[0:height, 0:width].astype(
        np.float32
    )
    map_x = x_coordinates + target_to_source_flow[..., 0]
    map_y = y_coordinates + target_to_source_flow[..., 1]
    return cv2.remap(
        source.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _estimate_geometric_flow(
    target_gray: np.ndarray,
    source_gray: np.ndarray,
) -> np.ndarray:
    target_low = cv2.GaussianBlur(
        target_gray, (0, 0), sigmaX=2.0, sigmaY=2.0
    )
    source_low = cv2.GaussianBlur(
        source_gray, (0, 0), sigmaX=2.0, sigmaY=2.0
    )
    return estimate_flow(target_low, source_low)


def compute_mc_residual_v3(
    frame_current: np.ndarray,
    frame_next: np.ndarray,
    fb_alpha: float = 0.05,
    fb_beta: float = 1.5,
    tex_percentile: float = 50.0,
    tex_abs_thr: float = 0.03,
    structure_percentile: float = 90.0,
) -> dict[str, np.ndarray]:
    """Compute the unmasked texture residual in current-frame coordinates."""
    # Retain obsolete keyword arguments so older callers remain compatible.
    del fb_alpha, fb_beta, tex_percentile, tex_abs_thr, structure_percentile

    current_gray = _gray01(frame_current)
    next_gray = _gray01(frame_next)
    current_to_next = _estimate_geometric_flow(current_gray, next_gray)
    flow_magnitude = np.sqrt(np.sum(current_to_next**2, axis=2)).astype(
        np.float32
    )

    current_bands = _dog_bands(current_gray)
    next_bands = _dog_bands(next_gray)
    residuals = []
    for band_name in ("H1", "H2", "H3"):
        warped_band = _warp_backward(
            next_bands[band_name], current_to_next
        )
        residuals.append(
            np.abs(current_bands[band_name] - warped_band).astype(np.float32)
        )

    texture_residual = (
        0.5 * residuals[0] + 0.3 * residuals[1] + 0.2 * residuals[2]
    ).astype(np.float32)
    artifact = texture_residual.copy()
    return {
        "R_tex": texture_residual,
        "A_tex": artifact,
        "flow_mag": flow_magnitude,
    }
