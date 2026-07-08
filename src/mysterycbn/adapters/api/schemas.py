"""Pydantic request/response models for the /v1 API (ARCHITECTURE.md §5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mysterycbn.app.jobs import JobState

JobStateName = Literal["pending", "running", "succeeded", "failed", "cancelled"]

_STATE_NAMES: dict[JobState, JobStateName] = {
    JobState.PENDING: "pending",
    JobState.RUNNING: "running",
    JobState.SUCCEEDED: "succeeded",
    JobState.FAILED: "failed",
    JobState.CANCELLED: "cancelled",
}


def job_state_name(state: JobState) -> JobStateName:
    return _STATE_NAMES[state]


class ConvertResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    state: JobStateName
    fraction_complete: float = Field(ge=0.0, le=1.0)
    current_stage: str | None = None
    error: str | None = None
    error_type: str | None = None
    downloads: dict[str, str] | None = Field(
        default=None,
        description="Available artifact keys → GET /v1/download URLs (SUCCEEDED only).",
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    engine_version: str


class ErrorResponse(BaseModel):
    error_type: str
    message: str
