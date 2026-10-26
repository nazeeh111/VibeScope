# Installation and local use

VibeScope uses Python, NumPy/SciPy and OpenCV for measurement, FFmpeg for playback exports, and a small local FastAPI server. A GPU is not required. The interface is static HTML/CSS/JavaScript with no CDN or build step.

## macOS or Linux

Use Python 3.11–3.14 and Git. Python 3.12 is the locally tested runtime.

```bash
git clone https://github.com/nazeeh111/VibeScope.git
cd VibeScope
./setup.sh
./launch.sh
```

Open `http://127.0.0.1:8765`. If it is already occupied, use `./launch.sh --port 8767` and open the matching address. To select a Python executable for setup, use `PYTHON=python3.12 ./setup.sh`.

The launcher reuses `.venv`; it does not install packages on each run. The app binds to loopback only and rejects cross-origin browser requests. Ctrl+C stops it. A successful health check is available at `/api/health`.

## Windows (PowerShell)

```powershell
git clone https://github.com/nazeeh111/VibeScope.git
cd VibeScope
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m vibescope.cli serve
```

Use `.venv\Scripts\python` in place of `.venv/bin/python` in the remaining examples. The CPU implementation is portable; Windows-specific runtime behavior has not been tested locally.

## Data and limits

Uploads, previews and results stay in `runs/` by default. Set `--data-dir PATH` to select another local folder. No video is uploaded to an external service. Treat original recordings as private data and back them up independently.

Each result receives a new directory. CLI output directories must not already exist. Failed analysis removes only its own temporary stage. There is no automatic deletion of successful recordings/results; manage `runs/` yourself when no longer needed.

The default limits are 120 MiB per uploaded video, 900 decoded frames, 1920 pixels on the longest side, and 50 million aggregate decoded pixels. Clips are never silently resized or decimated. Trim/crop a copy of a large recording first. No audio is analyzed or exported.

`ffprobe` is optional but recommended for independent presentation-timestamp inspection. Install a free FFmpeg distribution that includes `ffprobe` and put it on PATH. Without it, presentation timing remains unverified and real input requires an explicit constant-capture-rate declaration. A detected variable-rate, duplicate-timestamp or dropped-frame gap is rejected even when that declaration is checked. Capture timing can still differ from file playback timing.

`imageio-ffmpeg` supplies a separately installed FFmpeg binary for generated playback videos. These browser exports use H264420 CRF8. They are display encodes rather than pixel-identical archival copies; every quantitative estimate precedes re-encoding. Synthetic source generation uses CRF0 coding after RGB→YUV conversion, which is not bit-exact RGB preservation. Odd dimensions are padded by one pixel for video playback; measurement coordinates and units retain the original decoded dimensions.

## Optional neural baseline and acceleration

Core inference does not import PyTorch. To run the external EulerMormer baseline:

```bash
.venv/bin/python -m pip install -e '.[accelerated]'
```

Follow [the pinned baseline protocol](FOUNDATIONS.md#optional-eulermormer-reproduction). `--device cpu` is the reproducible default. `--device mps` or `--device cuda` selects an available accelerator for the external neural model. Unsupported devices fail rather than silently switching hardware. Availability and performance depend on the local PyTorch build; an accelerator is not a claim of improved measurement accuracy.

## Troubleshooting

| Symptom | Action |
|---|---|
| Address already in use | Choose another `--port`; do not stop an unrelated server. |
| Import failure / too many frames | Trim a copy, retain its actual capture fps and re-import. |
| Variable-rate timing rejected | Export a controlled constant-rate acquisition with documented frame-time mapping. Changing an FPS label cannot repair gaps or aliasing. |
| No resolved frequency | Widen the selected band, include more cycles, improve lighting/texture, check raw displacement and inspect the resolver policy. |
| Unobservable frame | Use a stable textured region without occlusions. The instrument does not insert zero values or silently interpolate missing measurements. |
| Empty baseline / mismatch | Use the current baseline command and the same capture fps, duration, camera/ROI/axis/scale/exposure/method. |
| Slow CPU run | Use a smaller textured ROI and a trimmed clip. Timings depend on system load; NumPy thread counts default to 1 in the launcher. |

For tests and reproduction, see [CONTRIBUTING.md](../CONTRIBUTING.md) and [BENCHMARKS.md](BENCHMARKS.md).
