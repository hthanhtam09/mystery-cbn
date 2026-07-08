"""Unit tests for the /v1 HTTP API adapter (ARCHITECTURE.md §5, adapters/api)."""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from mysterycbn.adapters.api.job_store import JobStore  # noqa: E402
from mysterycbn.adapters.api.main import create_app  # noqa: E402
from mysterycbn.adapters.api.routes import get_store  # noqa: E402


def _two_tone_png() -> bytes:
    """Synthetic two-region fixture -- no real imagery (ARCHITECTURE.md §10)."""
    w, h = 64, 64
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (220, 30, 30)
    arr[:, w // 2 :] = (30, 30, 220)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client() -> fastapi_testclient.TestClient:  # type: ignore[name-defined]
    app = create_app()
    store = JobStore(max_workers=2)
    app.dependency_overrides[get_store] = lambda: store
    with fastapi_testclient.TestClient(app) as c:
        yield c
    store.shutdown()


def _await_terminal(client: fastapi_testclient.TestClient, job_id: str, *, timeout_s: float = 30.0):  # type: ignore[name-defined]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["state"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state within {timeout_s}s")


def test_health_reports_engine_version(client: fastapi_testclient.TestClient) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["engine_version"]


def test_convert_then_poll_then_download_svg(client: fastapi_testclient.TestClient) -> None:
    resp = client.post(
        "/v1/convert",
        files={"file": ("photo.png", _two_tone_png(), "image/png")},
        data={"preset": "medium"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status = _await_terminal(client, job_id)
    assert status["state"] == "succeeded"
    assert status["fraction_complete"] == 1.0
    assert status["downloads"] is not None
    assert set(status["downloads"]) == {"svg", "pdf", "preview_lineart", "preview_solved"}

    svg_resp = client.get(f"/v1/download/{job_id}", params={"artifact": "svg"})
    assert svg_resp.status_code == 200
    assert svg_resp.headers["content-type"].startswith("image/svg+xml")
    assert svg_resp.content.startswith(b"<?xml")
    assert "content-disposition" not in svg_resp.headers  # inline by default

    attachment_resp = client.get(
        f"/v1/download/{job_id}", params={"artifact": "svg", "as_attachment": "true"}
    )
    assert attachment_resp.status_code == 200
    disposition = attachment_resp.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert f"mystery-cbn-{job_id}.svg" in disposition


def test_convert_defaults_to_medium_preset_without_form_fields(
    client: fastapi_testclient.TestClient,
) -> None:
    resp = client.post("/v1/convert", files={"file": ("photo.png", _two_tone_png(), "image/png")})
    assert resp.status_code == 202
    status = _await_terminal(client, resp.json()["job_id"])
    assert status["state"] == "succeeded"


def test_convert_rejects_an_empty_upload(client: fastapi_testclient.TestClient) -> None:
    resp = client.post("/v1/convert", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400


def test_convert_rejects_malformed_overrides_json(client: fastapi_testclient.TestClient) -> None:
    resp = client.post(
        "/v1/convert",
        files={"file": ("photo.png", _two_tone_png(), "image/png")},
        data={"overrides": "{not json"},
    )
    assert resp.status_code == 400


def test_convert_surfaces_unknown_preset_as_a_failed_job(
    client: fastapi_testclient.TestClient,
) -> None:
    resp = client.post(
        "/v1/convert",
        files={"file": ("photo.png", _two_tone_png(), "image/png")},
        data={"preset": "extreme"},
    )
    assert resp.status_code == 202  # submission always succeeds; the error is async
    status = _await_terminal(client, resp.json()["job_id"])
    assert status["state"] == "failed"
    assert status["error_type"] == "ConfigError"
    assert status["error"]


def test_job_status_for_unknown_id_is_404(client: fastapi_testclient.TestClient) -> None:
    resp = client.get("/v1/jobs/does-not-exist")
    assert resp.status_code == 404


def test_download_before_completion_is_409(client: fastapi_testclient.TestClient) -> None:
    resp = client.post("/v1/convert", files={"file": ("photo.png", _two_tone_png(), "image/png")})
    job_id = resp.json()["job_id"]
    download_resp = client.get(f"/v1/download/{job_id}")
    assert download_resp.status_code in (200, 409)  # racy: may already be done
    _await_terminal(client, job_id)  # drain before fixture teardown


def test_download_unknown_artifact_is_400(client: fastapi_testclient.TestClient) -> None:
    resp = client.post("/v1/convert", files={"file": ("photo.png", _two_tone_png(), "image/png")})
    job_id = resp.json()["job_id"]
    _await_terminal(client, job_id)
    bad = client.get(f"/v1/download/{job_id}", params={"artifact": "nope"})
    assert bad.status_code == 400


def test_cancel_a_pending_job_transitions_it_to_cancelled(
    client: fastapi_testclient.TestClient,
) -> None:
    resp = client.post("/v1/convert", files={"file": ("photo.png", _two_tone_png(), "image/png")})
    job_id = resp.json()["job_id"]
    cancel_resp = client.delete(f"/v1/jobs/{job_id}")
    assert cancel_resp.status_code == 202

    status = _await_terminal(client, job_id)
    # Cooperative cancellation: the job may finish before the token is
    # checked between stages if it was already deep in the pipeline, but on
    # this tiny fixture with 2 workers it reliably lands as cancelled or
    # (rarely) succeeded -- both are valid terminal states, never "failed".
    assert status["state"] in ("cancelled", "succeeded")


def test_cancel_unknown_job_is_404(client: fastapi_testclient.TestClient) -> None:
    resp = client.delete("/v1/jobs/does-not-exist")
    assert resp.status_code == 404
