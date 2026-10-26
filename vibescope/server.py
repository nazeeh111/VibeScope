"""Loopback-only, local-file measurement service with bounded work and uploads."""

from __future__ import annotations
import json
import hashlib
import shutil
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from .video import metadata
from .pipeline import analyze
from .synthetic import make_demo

MAX_UPLOAD = 120 * 1024 * 1024


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    video_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    roi: list[int] = Field(min_length=4, max_length=4)
    reference_roi: list[int] | None = Field(default=None, min_length=4, max_length=4)
    axis: str = "x"
    low_hz: float = 4.0
    high_hz: float = 12.0
    gain: float = 20.0
    mm_per_px: float | None = None
    method: str = "classical"
    ringdown: bool = False
    fps_override: float | None = Field(default=None, gt=0, le=2000)
    assume_cfr: bool = False
    baseline_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    baseline_scope: str | None = Field(default=None, max_length=512)


def create_app(data_dir=None):
    root = Path(data_dir or Path.cwd() / "runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="VibeScope", docs_url=None, redoc_url=None)
    lock = threading.Lock()

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # No CORS. Reject browser requests from other origins, including mutation by forms.
        origin = request.headers.get("origin")
        host = request.headers.get("host", "")
        host_name = host.rsplit(":", 1)[0] if ":" in host else host
        if host_name not in ("127.0.0.1", "localhost", "testserver", "[::1]"):
            return JSONResponse(
                {"detail": "Use the local loopback address."}, status_code=403
            )
        if origin and urlparse(origin).netloc != host:
            return JSONResponse(
                {"detail": "Cross-origin requests are not allowed."}, status_code=403
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def register(path, name, kind="user_recording", truth=None, video_id=None):
        video_id = video_id or uuid.uuid4().hex
        folder = root / video_id
        folder.mkdir(exist_ok=True)
        dest = folder / "input.video"
        if Path(path).resolve() != dest:
            shutil.copyfile(path, dest)
        frames, info = metadata(dest)
        cv2.imwrite(
            str(folder / "preview.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        )
        warnings = []
        if kind != "synthetic":
            warnings = [
                "Confirm actual constant capture fps. Metadata/playback fps can be wrong for slow motion.",
                "Rolling shutter, lens distortion, exposure blur and out-of-plane motion are not corrected.",
            ]
        info.update(
            id=video_id,
            name=name,
            source_kind=kind,
            preview_url=f"/data/{video_id}/preview.png",
            warnings=warnings,
        )
        if truth:
            info.update(
                recommended_roi=truth["roi"],
                recommended_reference_roi=truth["reference_roi"],
                recommended_band=[4, 12],
                recommended_axis="x",
            )
        (folder / "source.json").write_text(json.dumps(info))
        return info

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "scope": "local experimental video measurement",
        }

    @app.get("/api/demo")
    def demo():
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Another job is running. Wait for it to finish.")
        try:
            # Fixed ID only for our generated disposable demo, no supplied files overwritten.
            video_id = "f" * 32
            folder = root / video_id
            source = folder / "source.json"
            if source.exists():
                return json.loads(source.read_text())
            bundled = Path(__file__).with_name("data")
            if (bundled / "controlled.mp4").is_file() and (
                bundled / "truth.json"
            ).is_file():
                truth = json.loads((bundled / "truth.json").read_text())
                video = bundled / "controlled.mp4"
                with video.open("rb") as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                if actual != truth.get("video_sha256"):
                    raise HTTPException(
                        500,
                        "Packaged demonstration checksum failed. Reinstall VibeScope.",
                    )
            else:
                video, truth = make_demo(folder)
            return register(
                video,
                "Controlled target · 7.3 Hz",
                kind="synthetic",
                truth=truth,
                video_id=video_id,
            )
        finally:
            lock.release()

    @app.post("/api/upload")
    def upload(file: UploadFile = File(...)):
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Another job is running. Wait for it to finish.")
        video_id = uuid.uuid4().hex
        folder = root / video_id
        folder.mkdir()
        try:
            dest = folder / "input.video"
            total = 0
            with dest.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise ValueError(
                            "Upload exceeds 120 MiB. Trim a copy of the source first."
                        )
                    output.write(chunk)
            return register(
                dest, Path(file.filename or "User recording").name, video_id=video_id
            )
        except ValueError as e:
            # Only the newly created failed upload copy is disposable.
            shutil.rmtree(folder)
            raise HTTPException(422, str(e)) from e
        finally:
            file.file.close()
            lock.release()

    @app.post("/api/baseline")
    def import_baseline(file: UploadFile = File(...)):
        from .anomaly import validate_reference

        try:
            content = file.file.read(64 * 1024 + 1)
            if len(content) > 64 * 1024:
                raise ValueError("Reference JSON exceeds64KiB.")
            model = validate_reference(json.loads(content))
            reference_id = uuid.uuid4().hex
            folder = root / "references"
            folder.mkdir(exist_ok=True)
            (folder / (reference_id + ".json")).write_text(
                json.dumps(model, allow_nan=False)
            )
            return dict(
                id=reference_id,
                scope=model["scope"],
                fps=model["fps"],
                training_records=model["training_records"],
                calibration_records=model["calibration_records"],
                sample_count_range=model["sample_count_range"],
            )
        except (ValueError, UnicodeDecodeError, TypeError) as e:
            raise HTTPException(422, str(e)) from e
        finally:
            file.file.close()

    @app.post("/api/analyze")
    def measurement(request: AnalysisRequest):
        folder = root / request.video_id
        source = folder / "source.json"
        if not source.exists():
            raise HTTPException(404, "Source not found. Import the recording again.")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Another job is running. Wait for it to finish.")
        try:
            info = json.loads(source.read_text())
            run_id = uuid.uuid4().hex
            destination = folder / run_id
            params = request.model_dump(exclude={"video_id", "baseline_id"})
            if request.baseline_id:
                baseline = root / "references" / (request.baseline_id + ".json")
                if not baseline.exists():
                    raise ValueError(
                        "Reference not found. Load the baseline JSON again."
                    )
                if not request.baseline_scope:
                    raise ValueError(
                        "Confirm the same acquisition setup before reference screening."
                    )
                params["baseline"] = baseline
            elif request.baseline_scope:
                raise ValueError("Load a reference JSON before confirming its setup.")
            result = analyze(
                folder / "input.video",
                destination,
                source_kind=info["source_kind"],
                **params,
            )
            prefix = f"/data/{request.video_id}/{run_id}/"
            result.update(
                id=run_id,
                artifacts={
                    k: prefix + v
                    for k, v in dict(
                        original="original.mp4",
                        enhanced="enhanced.mp4",
                        heatmap="heatmap.png",
                        csv="displacement.csv",
                        plot="measurement.png",
                        report="report.json",
                    ).items()
                },
            )
            (destination / "report.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )
            return result
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(422, str(e)) from e
        finally:
            lock.release()

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).with_name("static")),
        name="static",
    )
    app.mount("/data", StaticFiles(directory=root), name="data")
    return BodyLimitMiddleware(app)


class BodyLimitMiddleware:
    """Cap request bytes before multipart spooling, including chunked uploads."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        limit = MAX_UPLOAD + 1024 * 1024 if path == "/api/upload" else 1024 * 1024
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = limit + 1

        async def reject():
            response = JSONResponse(
                {
                    "detail": "Request exceeds the upload size limit. Trim a copy of the video before importing."
                },
                status_code=413,
            )
            await response(scope, receive, send)

        if length < 0 or length > limit:
            return await reject()
        total = 0
        exceeded = False

        class TooLarge(Exception):
            pass

        async def capped_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    exceeded = True
                    raise TooLarge()
            return message

        async def guarded_send(message):
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, capped_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded:
            await reject()
