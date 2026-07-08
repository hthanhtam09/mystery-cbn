"""``/v1`` router: submit conversions, poll status, fetch artifacts (ARCHITECTURE.md §5)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from mysterycbn import __version__ as ENGINE_VERSION
from mysterycbn.adapters.api.job_store import JobRecord, JobStore
from mysterycbn.adapters.api.schemas import (
    ConvertResponse,
    HealthResponse,
    JobStatusResponse,
    job_state_name,
)
from mysterycbn.app.jobs import JobState

router = APIRouter(prefix="/v1")

# One process-wide store; FastAPI dependency indirection keeps routes testable
# with a substitute store (see tests/unit/test_api.py's app.dependency_overrides).
_store = JobStore()

_DOWNLOAD_KEYS = ("svg", "pdf", "preview_lineart", "preview_solved")


def get_store() -> JobStore:
    return _store


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(engine_version=ENGINE_VERSION)


@router.post("/convert", response_model=ConvertResponse, status_code=202)
async def submit_convert(
    file: UploadFile = File(...),
    preset: str = Form("medium"),
    seed: int = Form(0),
    overrides: str | None = Form(None, description="JSON object of dotted-key config overrides."),
    store: JobStore = Depends(get_store),
) -> ConvertResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    parsed_overrides: dict[str, object] | None = None
    if overrides is not None:
        try:
            parsed_overrides = json.loads(overrides)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"overrides is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed_overrides, dict):
            raise HTTPException(status_code=400, detail="overrides must be a JSON object")

    # ConfigError/InputError (e.g. an unknown preset) surface asynchronously
    # via GET /v1/jobs/{id} — convert() raises them on the worker thread, not
    # here, so submission itself always succeeds once the upload is read.
    job_id = store.submit(data, preset=preset, overrides=parsed_overrides, seed=seed)
    return ConvertResponse(job_id=job_id)


def _require_job(store: JobStore, job_id: str) -> JobRecord:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    return record


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, store: JobStore = Depends(get_store)) -> JobStatusResponse:
    record = _require_job(store, job_id)
    downloads = None
    if record.state is JobState.SUCCEEDED:
        downloads = {key: f"/v1/download/{job_id}?artifact={key}" for key in _DOWNLOAD_KEYS}
    return JobStatusResponse(
        job_id=record.job_id,
        state=job_state_name(record.state),
        fraction_complete=record.fraction_complete,
        current_stage=record.current_stage,
        error=record.error,
        error_type=record.error_type,
        downloads=downloads,
    )


@router.delete("/jobs/{job_id}", status_code=202)
def cancel_job(job_id: str, store: JobStore = Depends(get_store)) -> dict[str, bool]:
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    cancelled = store.cancel(job_id)
    return {"cancellation_requested": cancelled}


_CONTENT_TYPES = {
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "preview_lineart": "image/png",
    "preview_solved": "image/png",
}

_EXTENSIONS = {
    "svg": "svg",
    "pdf": "pdf",
    "preview_lineart": "png",
    "preview_solved": "png",
}


@router.get("/download/{job_id}")
def download(
    job_id: str,
    artifact: str = "svg",
    as_attachment: bool = False,
    store: JobStore = Depends(get_store),
) -> Response:
    """Serve a job's artifact bytes.

    Inline by default (``<img src>``-friendly for the two preview PNGs);
    pass ``as_attachment=true`` for a browser "Save As" download — a plain
    cross-origin ``<a download>`` is unreliable without ``Content-Disposition:
    attachment`` (browsers ignore the attribute across origins), so the
    frontend's download buttons set this explicitly.
    """
    record = _require_job(store, job_id)
    if record.state is not JobState.SUCCEEDED or record.result is None:
        raise HTTPException(
            status_code=409,
            detail=f"job {job_id!r} is {job_state_name(record.state)}, not succeeded",
        )
    if artifact not in _DOWNLOAD_KEYS:
        raise HTTPException(
            status_code=400, detail=f"unknown artifact {artifact!r}; choose from {_DOWNLOAD_KEYS}"
        )

    bundle = record.result
    if artifact == "svg":
        content: bytes | None = bundle.svg
    elif artifact == "pdf":
        content = bundle.pdf
    else:
        content = bundle.previews.get(artifact.removeprefix("preview_"))
    if content is None:
        raise HTTPException(
            status_code=404, detail=f"artifact {artifact!r} was not produced for this job"
        )

    headers = {}
    if as_attachment:
        headers["Content-Disposition"] = (
            f'attachment; filename="mystery-cbn-{job_id}.{_EXTENSIONS[artifact]}"'
        )
    return Response(content=content, media_type=_CONTENT_TYPES[artifact], headers=headers)
