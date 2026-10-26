"""Bounded local decoding and browser-compatible exports. No remote video URLs."""

from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

MAX_FRAMES = 900
MAX_PIXELS = 50_000_000  # aggregate decoded image pixels (~150 MB uint8 RGB)
MAX_DIMENSION = 1920


def ffmpeg_binary():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_video(path, frames, fps, lossless=False):
    """Encode playback frames; optional CRF0 is lossless YUV coding, not RGB identity.

    Browser exports use CRF8/H264420. Odd dimensions receive one padding pixel.
    Every physical estimate precedes this display encoding.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    command = [
        ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "0" if lossless else "8",
        "-preset",
        "fast",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait() != 0:
            raise ValueError(f"Video encoding failed: {error[-400:]}")
    except Exception:
        process.kill()
        process.wait()
        raise
    return str(path)


def inspect_timing(path):
    """Probe presentation timestamps when ffprobe exists; otherwise require CFR declaration."""
    binary = shutil.which("ffprobe")
    if not binary:
        return dict(
            verified=False,
            reason="ffprobe unavailable; constant-rate timestamps unverified.",
        )
    try:
        data = json.loads(
            subprocess.check_output(
                [
                    binary,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "frame=best_effort_timestamp_time",
                    "-of",
                    "json",
                    str(path),
                ],
                timeout=30,
            )
        )
        stamps = np.array(
            [
                float(f["best_effort_timestamp_time"])
                for f in data["frames"]
                if "best_effort_timestamp_time" in f
            ]
        )
        intervals = np.diff(stamps)
        if np.any(intervals <= 0):
            return dict(
                verified=False,
                variable_rate=True,
                invalid_timestamps=True,
                reason="Nonmonotonic or duplicate video timestamps detected.",
            )
        if len(intervals) < 5:
            return dict(
                verified=False, reason="Too few video timestamps for inspection."
            )
        median = np.median(intervals)
        deviation = float(np.max(np.abs(intervals - median)))
        # Allow integer timestamp rounding but reject genuinely variable rate/dropped frames.
        uniform = deviation < max(0.0001, median * 0.02)
        return dict(
            verified=bool(uniform),
            variable_rate=not bool(uniform),
            max_interval_deviation_s=deviation,
            fps_from_timestamps=float(1 / np.mean(intervals)),
        )
    except (subprocess.SubprocessError, ValueError, KeyError):
        return dict(
            verified=False,
            reason="Timestamp inspection failed; constant-rate timing unverified.",
        )


def read_video(path, fps_override=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(
            "Video could not be decoded. Use a local MP4, MOV or AVI file."
        )
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps_override is not None:
        fps = float(fps_override)
    if not np.isfinite(fps) or fps <= 0 or fps > 2000:
        cap.release()
        raise ValueError(
            "A valid capture frame rate is required (not slow-motion playback rate)."
        )
    frames = []
    total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            total += h * w
            if (
                max(h, w) > MAX_DIMENSION
                or len(frames) >= MAX_FRAMES
                or total > MAX_PIXELS
            ):
                raise ValueError(
                    "Video exceeds the local processing limit (900 frames, 1920 px side, 50 million total pixels). Trim/crop a copy first; no frames were silently skipped."
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(frames) < 40:
        raise ValueError("Use at least 40 decoded frames.")
    return np.asarray(frames), fps


def metadata(path, fps_override=None):
    frames, fps = read_video(path, fps_override)
    n, h, w = frames.shape[:3]
    timing = inspect_timing(path)
    return frames, dict(
        fps=fps, frame_count=n, width=w, height=h, duration=n / fps, timing=timing
    )
