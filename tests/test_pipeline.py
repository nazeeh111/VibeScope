import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from vibescope.synthetic import make_demo
from vibescope.pipeline import analyze
from vibescope.server import create_app
from vibescope.video import metadata


@pytest.fixture(scope="module")
def fixture_video(tmp_path_factory):
    root = tmp_path_factory.mktemp("video")
    p, truth = make_demo(root, frames=120)
    return p, truth, root


def test_decode_encode_and_independent_measurements(fixture_video):
    path, truth, root = fixture_video
    frames, info = metadata(path)
    assert info["frame_count"] == 120 and info["width"] == 320
    result = analyze(
        path,
        root / "measurement",
        truth["roi"],
        truth["reference_roi"],
        low_hz=4,
        high_hz=12,
        source_kind="synthetic",
        gain=10,
        mm_per_px=0.1,
    )
    m = result["metrics"]
    assert abs(m["frequency_hz"] - truth["frequency_hz"]) < 0.08
    assert abs(m["amplitude_px"] - truth["amplitude_px"]) < 0.035
    assert m["amplitude_mm"] == pytest.approx(m["amplitude_px"] * 0.1)
    assert result["metadata"]["source_kind"] == "synthetic"
    assert (root / "measurement" / "displacement.csv").exists()
    assert "not displacement" in result["visualization"]["activity_kind"]
    report = json.loads((root / "measurement" / "report.json").read_text())
    assert report["parameters"]["gain"] == 10
    assert report["metrics"]["anomaly"]["status"] == "not_assessed"


def test_real_source_requires_capture_declaration(fixture_video):
    path, truth, root = fixture_video
    with pytest.raises(ValueError, match="Confirm constant capture"):
        analyze(path, root / "invalid", truth["roi"])


def test_overlap_and_flat_roi_fail(fixture_video):
    path, truth, root = fixture_video
    with pytest.raises(ValueError, match="overlap"):
        analyze(
            path, root / "overlap", truth["roi"], truth["roi"], source_kind="synthetic"
        )


def test_local_api_controls_and_invalid_requests(tmp_path):
    app = create_app(tmp_path / "runs")
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 200
        assert (
            client.get("/api/health", headers={"Host": "evil.example"}).status_code
            == 403
        )
        assert (
            client.post(
                "/api/analyze", headers={"Origin": "https://evil.example"}, json={}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/analyze", json={"video_id": "../../secret", "roi": [0, 0, 24, 24]}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/upload", files={"file": ("bad.mp4", b"not a video", "video/mp4")}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/analyze", json={"video_id": "a" * 32, "roi": [0, 0, 24, 24]}
            ).status_code
            == 404
        )


def test_output_collision_never_changes_original(fixture_video):
    import hashlib

    path, truth, root = fixture_video
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Output already exists"):
        analyze(path, root, truth["roi"], source_kind="synthetic")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_api_body_limits_before_multipart_parsing(tmp_path):
    from vibescope.server import MAX_UPLOAD

    with TestClient(create_app(tmp_path / "runs")) as client:
        response = client.post(
            "/api/upload",
            content=b"x",
            headers={"Content-Length": str(MAX_UPLOAD + 2 * 1024 * 1024)},
        )
        assert response.status_code == 413
        # A stream without declared Content-Length is also bounded.
        parts = [
            b'--vs\r\nContent-Disposition: form-data; name="file"; filename="baseline.json"\r\nContent-Type: application/json\r\n\r\n',
            b"x" * 600_000,
            b"y" * 600_000,
            b"\r\n--vs--\r\n",
        ]
        response = client.post(
            "/api/baseline",
            content=iter(parts),
            headers={"Content-Type": "multipart/form-data; boundary=vs"},
        )
        assert response.status_code == 413
        assert not list((tmp_path / "runs").rglob("input.video"))


def test_reference_api_and_mismatched_fps(tmp_path):
    from vibescope.anomaly import train_reference

    t = np.arange(120) / 60
    records = [(0.2 + i * 0.001) * np.sin(2 * np.pi * 7.3 * t) for i in range(20)]
    model = train_reference(
        records[:10], records[10:], 60, "cameraA regionA x axis,1px units"
    )
    with TestClient(create_app(tmp_path / "runs")) as client:
        response = client.post(
            "/api/baseline",
            files={"file": ("baseline.json", json.dumps(model), "application/json")},
        )
        assert response.status_code == 200
        assert response.json()["scope"] == model["scope"]
        assert response.json()["fps"] == 60
        bad = client.post(
            "/api/baseline",
            files={"file": ("baseline.json", '{"version":1}', "application/json")},
        )
        assert bad.status_code == 422


def test_unobservable_frames_fail_instead_of_inserting_zeros(
    fixture_video, monkeypatch
):
    import vibescope.pipeline as pipeline

    path, truth, root = fixture_video
    n = 120
    t = np.arange(n) / 60

    def damaged_track(*args, **kwargs):
        q = np.ones(n)
        q[10] = 0
        return 0.28 * np.sin(2 * np.pi * 7.3 * t), q, np.ones(n) * 0.01

    monkeypatch.setattr(pipeline, "track", damaged_track)
    destination = root / "damaged"
    with pytest.raises(ValueError, match="unobservable frames"):
        pipeline.analyze(path, destination, truth["roi"], source_kind="synthetic")
    assert not destination.exists()


def test_nonmonotonic_timing_is_always_rejected(fixture_video, monkeypatch):
    import vibescope.pipeline as pipeline

    path, truth, root = fixture_video
    frames, info = metadata(path)
    info["timing"] = {
        "verified": False,
        "variable_rate": True,
        "invalid_timestamps": True,
    }
    monkeypatch.setattr(pipeline, "metadata", lambda *args: (frames, info))
    with pytest.raises(ValueError, match="Variable-rate"):
        pipeline.analyze(path, root / "badtiming", truth["roi"], assume_cfr=True)


@pytest.mark.parametrize("method", ["learned", "phase"])
def test_optional_estimators_complete_inference_and_visual_exports(
    fixture_video, method
):
    path, truth, root = fixture_video
    result = analyze(
        path,
        root / ("method-" + method),
        truth["roi"],
        truth["reference_roi"],
        low_hz=4,
        high_hz=12,
        source_kind="synthetic",
        gain=8,
        method=method,
    )
    assert result["parameters"]["method"] == method
    assert result["metrics"]["status"] == "resolved"
    assert abs(result["metrics"]["frequency_hz"] - truth["frequency_hz"]) < 0.12
    assert result["metrics"]["unobservable_frame_fraction"] == 0
    assert (root / ("method-" + method) / "enhanced.mp4").stat().st_size > 1000
    assert result["metrics"]["amplitude_px"] > 0
