"""Analysis orchestration. Original-frame estimates never depend on display gain."""

from __future__ import annotations
import csv
import hashlib
import json
import time
import os
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import shift as image_shift
from scipy import signal
from .motion import track, validate_roi, analytic_band
from .signals import analyze_signal
from .video import metadata, encode_video


def _figure(path, result, title):
    cache = Path(tempfile.gettempdir()) / "vibescope-matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = result["signal"]
    m = result["metrics"]
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, axs = plt.subplots(2, 1, figsize=(10, 5.8), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.955))
        axs[0].plot(
            s["time_s"],
            s["raw_px"],
            color="#a9b6b1",
            lw=0.8,
            label="Original-frame displacement",
        )
        axs[0].plot(
            s["time_s"],
            s["filtered_px"],
            color="#12615b",
            lw=1.2,
            label="Selected band",
        )
        axs[0].set(xlabel="Capture time (s)", ylabel="Displacement (px)", title=title)
        axs[0].legend(frameon=False, loc="upper right")
        axs[1].semilogy(
            s["frequency_hz"][1:],
            np.maximum(s["psd_px2_hz"][1:], 1e-12),
            color="#12615b",
        )
        low, high = result["parameters"]["low_hz"], result["parameters"]["high_hz"]
        axs[1].axvspan(low, high, alpha=0.1, color="#12615b")
        axs[1].set(
            xlabel="Frequency (Hz)",
            ylabel="PSD (px²/Hz)",
            xlim=(0, result["metadata"]["fps"] / 2),
        )
        fig.text(
            0.99,
            0.005,
            "Measurement from originals • visualization gain excluded • synthetic ≠ real validation",
            ha="right",
            fontsize=8,
            color="#66736e",
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)


def render_visuals(
    frames,
    fps,
    roi,
    filtered,
    camera_x,
    camera_y,
    gain,
    directory,
    method="classical",
    low=4,
    high=12,
    axis="x",
):
    directory = Path(directory)
    x, y, w, h = roi
    stabilized = []
    for i, image in enumerate(frames):
        stabilized.append(
            np.clip(
                image_shift(
                    image.astype(np.float32),
                    (-camera_y[i], -camera_x[i], 0),
                    order=1,
                    mode="nearest",
                ),
                0,
                255,
            ).astype(np.uint8)
        )
    stabilized = np.asarray(stabilized)
    enhanced = stabilized.copy()
    local = stabilized[:, y : y + h, x : x + w].astype(float) / 255
    if method == "phase":
        gray = np.mean(local, axis=3)
        z = np.array([analytic_band(im, axis, 12) for im in gray])
        phases = np.unwrap(np.angle(z), axis=0)
        sos = signal.butter(3, [low, high], btype="bandpass", fs=fps, output="sos")
        phases = signal.sosfiltfilt(sos, phases, axis=0)
        delta = (np.real(z * np.exp(1j * (gain - 1) * phases)) - np.real(z))[..., None]
        visual = np.clip(local + delta, 0, 1)
    else:
        visual = []
        for i in range(len(frames)):
            displacement = (gain - 1) * filtered[i]
            delta = (0, displacement, 0) if axis == "x" else (displacement, 0, 0)
            visual.append(
                np.clip(image_shift(local[i], delta, order=1, mode="nearest"), 0, 1)
            )
        visual = np.asarray(visual)
    # Soft edge blending avoids a hard paste boundary; it is display, not dense flow.
    edge = np.minimum(
        np.minimum(np.arange(w), np.arange(w)[::-1])[None, :],
        np.minimum(np.arange(h), np.arange(h)[::-1])[:, None],
    )
    blend = np.clip(edge / 4, 0, 1)[None, :, :, None]
    enhanced[:, y : y + h, x : x + w] = np.clip(
        (visual * blend + local * (1 - blend)) * 255, 0, 255
    ).astype(np.uint8)
    for image in enhanced:
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (18, 97, 91), 1)
        cv2.putText(
            image,
            f"VISUALIZATION x{gain:g}",
            (8, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (20, 70, 65),
            1,
            cv2.LINE_AA,
        )
    encode_video(directory / "original.mp4", frames, fps)
    encode_video(directory / "enhanced.mp4", enhanced, fps)
    # Explicitly intensity activity, not a purported dense displacement measurement.
    sos = signal.butter(3, [low, high], btype="bandpass", fs=fps, output="sos")
    pixels = stabilized.reshape(len(frames), -1, 3)
    activity_flat = np.zeros(pixels.shape[1], dtype=np.float32)
    block = max(64, int(2_000_000 / len(frames)))
    for start in range(0, pixels.shape[1], block):
        gray = np.mean(
            pixels[:, start : start + block].astype(np.float32) / 255, axis=2
        )
        filtered_intensity = signal.sosfiltfilt(sos, gray, axis=0)
        activity_flat[start : start + block] = np.sqrt(
            np.mean(filtered_intensity**2, axis=0)
        )
    activity = activity_flat.reshape(frames.shape[1:3])
    scale = max(float(np.percentile(activity, 99)), 1e-6)
    colored = cv2.applyColorMap(
        np.uint8(np.clip(activity / scale, 0, 1) * 255), cv2.COLORMAP_INFERNO
    )
    cv2.imwrite(str(directory / "heatmap.png"), colored)
    np.save(
        directory / "intensity_activity.npy",
        activity.astype(np.float32),
        allow_pickle=False,
    )
    return dict(
        activity_scale_normalized_intensity=scale,
        activity_kind="Band-limited intensity RMS, normalized for display; not displacement or damage.",
    )


def _analyze(
    path,
    directory,
    roi,
    reference_roi=None,
    axis="x",
    low_hz=4.0,
    high_hz=12.0,
    gain=20.0,
    mm_per_px=None,
    method="classical",
    ringdown=False,
    fps_override=None,
    assume_cfr=False,
    source_kind="user_recording",
    baseline=None,
    baseline_scope=None,
    **_,
):
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if axis not in ("x", "y"):
        raise ValueError("Axis must be x or y.")
    if not np.isfinite(gain) or not 1 <= gain <= 100:
        raise ValueError("Visualization gain must be between 1 and 100.")
    if mm_per_px is not None and (not np.isfinite(mm_per_px) or mm_per_px <= 0):
        raise ValueError("Calibration must be a finite positive mm/px scale.")
    frames, meta = metadata(path, fps_override)
    fps = meta["fps"]
    roi = validate_roi(roi, frames.shape[1:3])
    if meta["timing"].get("variable_rate"):
        raise ValueError(
            "Variable-rate or dropped-frame timing detected. Resample with a documented capture-time mapping before analysis; overriding FPS cannot repair this."
        )
    if source_kind != "synthetic" and not assume_cfr:
        raise ValueError(
            "Confirm constant capture frame rate and verify the effective capture fps before measuring. Playback fps can differ for slow motion."
        )
    # Validate all signal controls before expensive tracking.
    analyze_signal(np.zeros(len(frames)), fps, low_hz, high_hz)
    gray = np.array(
        [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) / 255 for im in frames], dtype=np.float32
    )
    raw, q, sigma = track(gray, roi, method, axis)
    warnings = []
    if not meta["timing"].get("verified"):
        warnings.append(
            "Video presentation timing is unverified: "
            + meta["timing"].get("reason", "incomplete timestamps")
            + " Capture rate is a user declaration, not an independent measurement."
        )
    camera_x = np.zeros(len(frames))
    camera_y = np.zeros(len(frames))
    if reference_roi is not None:
        reference_roi = validate_roi(reference_roi, frames.shape[1:3])
        x, y, w, h = roi
        rx, ry, rw, rh = reference_roi
        if min(x + w, rx + rw) > max(x, rx) and min(y + h, ry + rh) > max(y, ry):
            raise ValueError(
                "Measurement and stationary reference regions must not overlap."
            )
        camera_x, rqx, _ = track(gray, reference_roi, "classical", "x")
        camera_y, rqy, _ = track(gray, reference_roi, "classical", "y")
        q = np.minimum(q, np.minimum(rqx, rqy))
        raw = raw - (camera_x if axis == "x" else camera_y)
        warnings.append(
            "Stationary-reference subtraction corrects image-plane translation only. Rotation, parallax and rolling shutter remain uncorrected."
        )
    else:
        warnings.append(
            "No stationary reference: measured motion includes camera motion. Do not interpret it as structure-only displacement."
        )
    quality = float(np.median(q))
    bad = float(np.mean(q < 0.25))
    if quality < 0.25 or bad > 0:
        raise ValueError(
            f"Motion not observable reliably (quality {quality:.2f}, {bad:.1%} unobservable frames). Choose a textured rigid region, smaller motion, or improve lighting. No measurement was issued."
        )
    s = analyze_signal(raw, fps, low_hz, high_hz, ringdown)
    metrics = s.pop("metrics")
    warnings.extend(s.pop("warnings"))
    metrics.update(
        quality=quality,
        rejected_frame_fraction=bad,
        unobservable_frame_fraction=bad,
        amplitude_mm=metrics["amplitude_px"] * mm_per_px
        if mm_per_px and metrics["amplitude_px"] is not None
        else None,
        tracking_error_scale_px=float(np.median(sigma)),
        tracking_error_kind="Synthetic error scale for learned method; local image-fit scatter for classical; neither is physical confidence.",
        anomaly=dict(
            status="not_assessed",
            explanation="No healthy reference matched to this acquisition setup. No diagnostic inference.",
        ),
    )
    if baseline is not None:
        from .anomaly import score, load_reference

        metrics["anomaly"] = score(
            raw, fps, load_reference(baseline), confirmed_scope=baseline_scope
        )
    if mm_per_px is not None:
        warnings.append(
            "mm conversion uses a user-supplied in-plane scale. Calibration uncertainty, perspective and out-of-plane motion are not included."
        )
    warnings += [
        "Subpixel estimates assume a textured rigid patch, fixed focus/exposure and small in-plane translation. Inspect original video and residuals.",
        "Nyquist is not an anti-aliasing guarantee: motion above fps/2 may appear at a lower frequency. Exposure blurs and attenuates displacement.",
        "Bandpass uses the full clip (noncausal). Edge transients affect the beginning and end; sine-fit amplitude comes from unfiltered original-frame displacement.",
        "Magnified video and intensity activity map are visualizations. No physical measurement is extracted from them.",
    ]
    if source_kind == "synthetic":
        warnings.insert(
            0,
            "CONTROLLED SYNTHETIC INPUT. Real-world displacement accuracy and diagnostic reliability are unverified.",
        )
    params = dict(
        roi=list(roi),
        reference_roi=list(reference_roi) if reference_roi else None,
        axis=axis,
        low_hz=low_hz,
        high_hz=high_hz,
        gain=gain,
        mm_per_px=mm_per_px,
        method=method,
        ringdown=ringdown,
        assume_cfr=assume_cfr,
        fps_override=fps_override,
    )
    result = dict(
        metadata={
            **meta,
            "source_kind": source_kind,
            "source_sha256": _source_digest(path),
            "effective_fps": fps,
        },
        metrics=metrics,
        parameters=params,
        signal={k: v.tolist() for k, v in s.items()},
        warnings=warnings,
    )
    result["visualization"] = render_visuals(
        frames,
        fps,
        roi,
        s["filtered_px"],
        camera_x,
        camera_y,
        gain,
        directory,
        method,
        low_hz,
        high_hz,
        axis,
    )
    with (directory / "displacement.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "raw_displacement_px",
                "band_displacement_px",
                "raw_displacement_mm",
                "tracking_quality",
                "tracking_error_scale_px",
            ]
        )
        for t, r, b, qi, si in zip(
            s["time_s"], s["raw_px"], s["filtered_px"], q, sigma
        ):
            writer.writerow([t, r, b, r * mm_per_px if mm_per_px else "", qi, si])
    _figure(
        directory / "measurement.png",
        result,
        "VibeScope • "
        + (
            "controlled synthetic source"
            if source_kind == "synthetic"
            else "user recording, unverified calibration"
        ),
    )
    result["metadata"]["analysis_seconds"] = time.perf_counter() - started
    result["metadata"]["software_version"] = "0.1.0"
    (directory / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    return result


def _source_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def analyze(path, directory, *args, **kwargs):
    """Publish one complete fresh result; never overwrite originals/old results.

    A failed run removes only its own temporary stage. The requested final output
    appears atomically after all artifacts and the report have succeeded.
    """
    destination = Path(directory).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(
            "Output already exists. Choose a fresh output directory; existing files and recordings are never overwritten."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".vibescope-stage-", dir=destination.parent))
    try:
        result = _analyze(path, stage, *args, **kwargs)
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                "Output was created by another process. Choose a fresh output directory."
            )
        stage.rename(destination)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
