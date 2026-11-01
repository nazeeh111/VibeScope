"""Reproduce one external rotor frequency-content check; no amplitude calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

from vibescope import pipeline


VIDEO_SHA = "0360bfa8fa39f45c4bb530085deda4b77c87ef9f6e40e5eeb73cadc18f016fd6"
SENSOR_SHA = "eb95278187f0d926f49b77203cb917509f86fac2421e2191e489486f558aab29"
CROP = (840, 350, 440, 220)
ROI = (32, 100, 112, 70)
REFERENCE = (375, 42, 52, 64)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def crop_recording(source, destination):
    """Keep all 500 frames and verify every decoded crop pixel, without scaling."""
    capture = cv2.VideoCapture(str(source))
    x, y, width, height = CROP
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"FFV1"), 100, (width, height)
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise ValueError("Cannot open the source or lossless FFV1 output.")
    expected = hashlib.sha256()
    count = 0
    first = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cropped = frame[y : y + height, x : x + width].copy()
            if first is None:
                first = cropped
            expected.update(cropped.tobytes())
            writer.write(cropped)
            count += 1
    finally:
        capture.release()
        writer.release()
    actual = hashlib.sha256()
    capture = cv2.VideoCapture(str(destination))
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            actual.update(frame.tobytes())
            decoded += 1
    finally:
        capture.release()
    if decoded != count or count != 500 or expected.digest() != actual.digest():
        raise ValueError("The crop did not preserve all 500 decoded frames exactly.")
    return first, expected.hexdigest()


def sensor_spectra(path):
    columns = np.loadtxt(path, delimiter=",")
    if columns.shape != (13000, 4) or not np.isfinite(columns).all():
        raise ValueError("Expected the published 13,000-row, four-channel sensor file.")
    records, spectra = [], []
    for index, values in enumerate(columns.T, start=1):
        frequencies, psd = signal.periodogram(
            values, fs=1000, window="hann", detrend="linear", scaling="density"
        )
        band = np.flatnonzero((frequencies >= 10) & (frequencies <= 80))
        peak = band[np.argmax(psd[band])]
        records.append({"channel": index, "peak_bin_hz": float(frequencies[peak])})
        spectra.append((frequencies, psd))
    return records, spectra


def figure(path, frame, result, spectra):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (photo, plot) = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.76))
        photo.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for box, color, label in [(ROI, "#12615b", "Collar patch"), (REFERENCE, "#ad4e1e", "Reference patch")]:
            x, y, w, h = box
            photo.add_patch(Rectangle((x, y), w, h, fill=False, color=color, linewidth=2))
            right = box == REFERENCE
            photo.text(x + w if right else x, y - 7, label, ha="right" if right else "left", color=color, fontsize=9, backgroundcolor="white")
        photo.set(title="Decoded original pixels; crop coordinates", xlabel="x (px)", ylabel="y (px)")
        vf = np.asarray(result["signal"]["frequency_hz"])
        vp = np.asarray(result["signal"]["psd_px2_hz"])
        chosen = (vf >= 10) & (vf <= 80)
        plot.plot(vf, vp / vp[chosen].max(), color="#12615b", linewidth=2, label="Video, 1 s")
        colors = ["#687a8c", "#ad4e1e", "#887ca1", "#a5a5a5"]
        for index, ((frequencies, psd), color) in enumerate(zip(spectra, colors), start=1):
            chosen = (frequencies >= 10) & (frequencies <= 80)
            plot.plot(frequencies, psd / psd[chosen].max(), color=color, linewidth=1, alpha=0.8, label=f"Sensor {index}, 13 s")
        plot.set(xlim=(10, 80), ylim=(0, 1.08), xlabel="Frequency (Hz)", ylabel="PSD / each trace's in-band maximum", title="Frequency content, not amplitude agreement")
        plot.legend(frameon=False, fontsize=8)
        fig.suptitle("One external rotor recording: a bounded frequency check", fontsize=14)
        fig.text(0.01, 0.075, "Capture: 500 fps (AVI playback: 100 fps). Video resolution: 1 Hz. No time alignment or physical scale established.", fontsize=9)
        fig.text(0.01, 0.025, "Data / photograph: Kang, Zheng, Liu & Pei (2024), doi:10.17632/47y7rwc2yg.1, CC BY 4.0. Cropped and annotated.", fontsize=8, color="#555555")
        fig.savefig(path, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("Output exists; choose a fresh folder. Existing results are preserved.")
    for path, expected in [(args.video, VIDEO_SHA), (args.sensors, SENSOR_SHA)]:
        try:
            actual = digest(path)
        except OSError as error:
            parser.error(f"Cannot read input: {error}")
        if actual != expected:
            parser.error(f"Published input SHA-256 mismatch: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rotor-case-", dir=output.parent) as temporary:
        stage = Path(temporary)
        cropped = stage / "crop.avi"
        first, pixels = crop_recording(args.video, cropped)
        result = pipeline.analyze(
            cropped, stage / "measurement", ROI, REFERENCE, axis="y",
            low_hz=10, high_hz=80, gain=1, fps_override=500, assume_cfr=True,
        )
        sensors, spectra = sensor_spectra(args.sensors)
        summary = {
            "dataset_doi": "10.17632/47y7rwc2yg.1", "dataset_license": "CC BY 4.0",
            "input_sha256": {"video": VIDEO_SHA, "sensors": SENSOR_SHA},
            "crop_xywh": CROP, "all_500_decoded_crop_frames_exact": True,
            "decoded_crop_bgr_sha256": pixels,
            "video_capture_fps_documented": 500, "video_playback_fps": 100,
            "sensor_sample_rate_hz_documented": 1000, "sensor_duration_s": 13,
            "sensor_fft_bin_spacing_hz": 1000 / 13000,
            "engine_files_sha256": {
                name: digest(Path(pipeline.__file__).parent / name)
                for name in ["pipeline.py", "motion.py", "signals.py", "video.py"]
            },
            "parameters": result["parameters"], "video_metrics": result["metrics"],
            "sensor_peak_bins_10_to_80_hz": sensors,
            "scope": "One recording and geometry-selected ROI, not a calibrated accuracy benchmark. Sensor and video are not time-aligned. No in-plane scale or channel-to-ROI correspondence established. Rotating features and lighting can produce matching spectral components without proving shaft translation. No amplitude, resonance, diagnostic or general accuracy claim.",
        }
        (stage / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
        figure(stage / "comparison.png", first, result, spectra)
        if output.exists() or output.is_symlink():
            parser.error("Output appeared during processing; existing results are preserved.")
        stage.rename(output)
    print(json.dumps(summary["video_metrics"], indent=2))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
