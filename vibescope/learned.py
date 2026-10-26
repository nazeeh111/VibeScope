"""Noise-aware subpixel translation estimator with portable, trained NumPy weights.

Positive x is right and positive y is down. Measurements require a textured,
approximately translating patch and small inter-frame displacement. This model
is a synthetic-trained calibration experiment, not a diagnostic instrument.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_MODEL = Path(__file__).with_name("models") / "noise_fusion.npz"
FEATURE_NAMES = (
    "middle_dx", "middle_dy", "fine_delta_x", "fine_delta_y",
    "coarse_delta_x", "coarse_delta_y", "log_noise_gradient_ratio",
    "log_gradient_rms", "eigenvalue_ratio", "gradient_correlation",
    "relative_residual", "brightness_offset",
)


def _images(reference: np.ndarray, current: np.ndarray):
    a, b = np.asarray(reference, np.float64), np.asarray(current, np.float64)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 16:
        raise ValueError("Use matching grayscale patches at least 16 × 16 pixels.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Motion patches must contain finite values.")
    if min(a.min(), b.min()) < -1e-5 or max(a.max(), b.max()) > 1.00001:
        raise ValueError("Motion patches must be normalized to 0..1.")
    return a, b


def pair_features(reference: np.ndarray, current: np.ndarray):
    """Return measured features, uncalibrated translation, and observability data."""
    a, b = _images(reference, current)
    # Cropping suppresses the artificial reflected boundaries of Gaussian derivatives.
    margin = min(6, min(a.shape) // 4)
    sl = (slice(margin, -margin), slice(margin, -margin))
    avg = (a + b) * 0.5
    shifts, diagnostics = [], []
    for scale in (0.65, 1.25, 2.0):
        gx = gaussian_filter(avg, scale, order=(0, 1))[sl]
        gy = gaussian_filter(avg, scale, order=(1, 0))[sl]
        diff = gaussian_filter(b - a, scale)[sl]
        # A constant lighting offset is a nuisance term, not displacement.
        offset = float(diff.mean())
        diff = diff - offset
        gx, gy = gx - gx.mean(), gy - gy.mean()
        matrix = np.array([[np.mean(gx * gx), np.mean(gx * gy)],
                           [np.mean(gx * gy), np.mean(gy * gy)]])
        eig = np.linalg.eigvalsh(matrix)
        ridge = max(1e-12, float(np.trace(matrix)) * 1e-7)
        rhs = -np.array([np.mean(gx * diff), np.mean(gy * diff)])
        shift = np.linalg.solve(matrix + np.eye(2) * ridge, rhs)
        residual = diff + gx * shift[0] + gy * shift[1]
        shifts.append(shift)
        diagnostics.append((matrix, eig, float(np.sqrt(np.mean(residual**2))), offset))
    fine, middle, coarse = shifts
    matrix, eig, residual, offset = diagnostics[1]
    grad = float(np.sqrt(np.trace(matrix)))
    high = (b - a) - gaussian_filter(b - a, 0.8)
    noise = float(np.median(np.abs(high - np.median(high))) / 0.67449 / np.sqrt(2))
    eigen_ratio = float(eig[0] / max(eig[1], 1e-14))
    corr = float(matrix[0, 1] / max(np.sqrt(matrix[0, 0] * matrix[1, 1]), 1e-14))
    features = np.array([
        *middle, *(fine - middle), *(coarse - middle),
        np.log10(max(noise / max(grad, 1e-8), 1e-5)),
        np.log10(max(grad, 1e-8)), eigen_ratio, corr,
        min(residual / max(grad, 1e-8), 10), offset / max(grad, 1e-8),
    ], np.float64)
    return features, middle, {"gradient_rms": grad, "eigenvalue_ratio": eigen_ratio,
                              "noise_estimate": noise, "residual": residual}


@lru_cache(maxsize=4)
def _load_model(path: str, version: int = 0):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _forward(features: np.ndarray, model):
    x = np.clip((features - model["mean"]) / model["std"], -8, 8)
    return np.tanh(x @ model["w1"] + model["b1"]) @ model["w2"] + model["b2"]


def baseline_pair(reference: np.ndarray, current: np.ndarray):
    """Middle-scale differential Lucas–Kanade estimate, without learned calibration."""
    _, shift, info = pair_features(reference, current)
    return {"dx": float(shift[0]), "dy": float(shift[1]), **info}


def estimate_pair(reference: np.ndarray, current: np.ndarray,
                  model_path=None, device="cpu") -> dict:
    """Estimate patch translation; sigma_px is a synthetic error scale, not a CI.

    ``device='cpu'`` is the portable path. No temporal filtering or magnification
    occurs here. The quality gate rejects aperture/flat patches and shifts outside
    the trained small-motion domain. NumPy weights never execute serialized code.
    """
    if device != "cpu":
        raise ValueError("Portable noise-fusion inference supports device='cpu' only.")
    features, base, info = pair_features(reference, current)
    path = Path(model_path or DEFAULT_MODEL).resolve()
    model = _load_model(str(path), path.stat().st_mtime_ns)
    correction = _forward(features, model)
    # The network can only apply a bounded calibration to a physical estimate.
    correction = np.clip(correction, -0.3, 0.3)
    shift = base + correction
    observable = info["gradient_rms"] >= 0.001 and info["eigenvalue_ratio"] >= 0.01
    in_domain = np.max(np.abs(base)) <= 1.3
    residual_ratio = info["residual"] / max(info["gradient_rms"], 1e-8)
    noise_ratio = info["noise_estimate"] / max(info["gradient_rms"], 1e-8)
    quality = float(np.clip(np.exp(-residual_ratio * 2) / (1 + noise_ratio * 0.15), 0, 1))
    if not observable or not in_domain:
        quality = 0.0
    scale = float(model["error_scale_px"])
    sigma = min(10.0, scale * (1 + residual_ratio + noise_ratio * 0.2))
    if not observable or not in_domain:
        sigma = 10.0
    return {"dx": float(shift[0]), "dy": float(shift[1]), "sigma_px": sigma,
            "quality": quality, "model": "noise-fusion-mlp-v1",
            "uncertainty_kind": "synthetic_error_scale_not_confidence_interval",
            "within_training_domain": bool(observable and in_domain), **info}
