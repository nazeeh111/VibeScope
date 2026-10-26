"""Controlled rendered target. Synthetic evidence only, never a real recording."""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, shift, spline_filter
from .video import encode_video


def texture(seed, shape=(64, 64), smooth=1.3):
    rng = np.random.default_rng(seed)
    im = gaussian_filter(rng.normal(size=shape), smooth)
    im = 0.50 + 0.18 * im / max(np.std(im), 1e-8)
    return np.clip(im, 0.02, 0.98).astype(np.float32)


def patch_sequence(
    seed=404,
    frames=180,
    fps=60.0,
    frequency=7.3,
    amplitude=0.25,
    noise=0.01,
    camera=0.0,
    damping=0.0,
    axis="x",
    harmonic=0.0,
):
    rng = np.random.default_rng(seed + 100000)
    reference = texture(seed)
    t = np.arange(frames) / fps
    true = amplitude * np.sin(2 * np.pi * frequency * t) * np.exp(
        -damping * t
    ) + harmonic * np.sin(2 * np.pi * frequency * 2 * t)
    camera_signal = camera * np.sin(2 * np.pi * 1.7 * t + 0.4)
    out = []
    for displacement in true + camera_signal:
        dy, dx = (0, displacement) if axis == "x" else (displacement, 0)
        image = shift(reference, (dy, dx), order=3, mode="reflect")
        out.append(np.clip(image + rng.normal(0, noise, reference.shape), 0, 1))
    return np.asarray(out, dtype=np.float32), true, camera_signal


def make_demo(directory, seed=812, frames=240, fps=60.0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "controlled.mp4").exists() or (directory / "truth.json").exists():
        raise ValueError(
            "Controlled demo output already exists. Choose a fresh folder; existing recordings are preserved."
        )
    rng = np.random.default_rng(seed)
    h, w = 192, 320
    bg = np.full((h, w, 3), 230, dtype=np.uint8)
    for x in range(0, w, 16):
        cv2.line(bg, (x, 0), (x, h), (220, 223, 221), 1)
    for y in range(0, h, 16):
        cv2.line(bg, (0, y), (w, y), (220, 223, 221), 1)
    # Textured stationary reference target for independent camera-motion subtraction.
    ref = (texture(seed + 1, (96, 64), 1) * 170 + 30).astype(np.uint8)
    bg[32:128, 24:88] = ref[..., None]
    cv2.rectangle(bg, (140, 22), (198, 48), (46, 57, 60), -1)
    target = np.zeros((h, w, 3), np.uint8)
    metal = (texture(seed + 2, (132, 56), 1.0) * 120 + 65).astype(np.uint8)
    target[40:172, 140:196] = metal[..., None]
    for y in range(48, 166, 8):
        cv2.line(target, (141, y), (150 if y % 16 else 157, y), (234, 235, 224), 1)
    cv2.line(target, (190, 44), (190, 166), (225, 225, 217), 1)
    mask = np.zeros((h, w), np.float32)
    mask[40:172, 140:196] = 1
    # Precompute spatial spline coefficients once. Treat RGB as three 2D fields,
    # never as a third spatial axis; compose camera and target transforms directly.
    bg_coeff = [spline_filter(bg[:, :, c].astype(float), order=3) for c in range(3)]
    target_coeff = spline_filter(target[:, :, 0].astype(float), order=3)
    mask_coeff = spline_filter(mask, order=3)
    t = np.arange(frames) / fps
    truth = 0.28 * np.sin(2 * np.pi * 7.3 * t)
    cam = 0.12 * np.sin(2 * np.pi * 1.7 * t)
    videos = []
    for i in range(frames):
        moved = shift(
            target_coeff,
            (0, truth[i] + cam[i]),
            order=3,
            mode="constant",
            prefilter=False,
        )
        moved_mask = np.clip(
            shift(
                mask_coeff,
                (0, truth[i] + cam[i]),
                order=3,
                mode="constant",
                prefilter=False,
            ),
            0,
            1,
        )
        background = np.stack(
            [
                shift(c, (0, cam[i]), order=3, mode="constant", prefilter=False)
                for c in bg_coeff
            ],
            axis=2,
        )
        image = background * (1 - moved_mask[..., None]) + moved[..., None]
        image = np.clip(image + rng.normal(0, 0.6, image.shape), 0, 255).astype(
            np.uint8
        )
        cv2.putText(
            image,
            "CONTROLLED SYNTHETIC / 60 fps",
            (8, 184),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (54, 65, 62),
            1,
            cv2.LINE_AA,
        )
        videos.append(image)
    output = directory / "controlled.mp4"
    encode_video(output, videos, fps, lossless=True)
    truth_data = dict(
        source_kind="synthetic",
        seed=seed,
        fps=fps,
        frequency_hz=7.3,
        amplitude_px=0.28,
        roi=[152, 108, 32, 48],
        reference_roi=[32, 40, 48, 72],
        axis="x",
        time_s=t.tolist(),
        displacement_px=truth.tolist(),
        camera_px=cam.tolist(),
        limitations="Rendered rigid translation, direct combined camera/target cubic sampling, independent additive pixel noise, 8-bit quantization and CRF0 YUV video coding; not a captured cantilever or physical test.",
    )
    (directory / "truth.json").write_text(json.dumps(truth_data, indent=2))
    return output, truth_data
