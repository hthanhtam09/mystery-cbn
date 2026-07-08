"""In-process async job store for the API adapter.

No external queue/broker: conversions run on a bounded thread pool inside
the API process (ARCHITECTURE.md's "async job model from day one" is about
the *HTTP contract*, not the execution backend — a single process is the
simplest thing that satisfies it, and swapping in a real queue later only
touches this file, per adapters/api's "high replaceability by design").

Jobs are addressed by ``job_id`` (uuid4 hex) and never removed automatically;
callers own retention (ARCHITECTURE.md doesn't specify a TTL, so none is
invented here).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from mysterycbn.app.api import convert
from mysterycbn.app.jobs import JobState
from mysterycbn.foundation.errors import CancelledError, EngineError
from mysterycbn.kernel.cancellation import ManualCancelToken
from mysterycbn.kernel.progress import ProgressEvent, ProgressKind
from mysterycbn.model.reports import OutputBundle

_MAX_WORKERS = 4


@dataclass
class JobRecord:
    """Mutable job state polled by ``GET /v1/jobs/{id}`` and consumed by
    ``GET /v1/download/{id}``; guarded by ``JobStore._lock``."""

    job_id: str
    state: JobState = JobState.PENDING
    fraction_complete: float = 0.0
    current_stage: str | None = None
    result: OutputBundle | None = None
    error: str | None = None
    error_type: str | None = None
    cancel_token: ManualCancelToken = field(default_factory=ManualCancelToken)


class _JobProgressListener:
    """Bridges kernel progress events onto a ``JobRecord`` under the store's lock."""

    def __init__(self, store: JobStore, job_id: str) -> None:
        self._store = store
        self._job_id = job_id

    def on_progress(self, event: ProgressEvent) -> None:
        self._store._update_progress(
            self._job_id,
            stage_name=event.stage_name,
            fraction_complete=event.fraction_complete,
            running=event.kind is ProgressKind.STAGE_STARTED,
        )


class JobStore:
    """Owns job records and the thread pool that runs conversions."""

    def __init__(self, *, max_workers: int = _MAX_WORKERS) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cbn-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[None]] = {}

    def submit(
        self,
        source: bytes,
        *,
        preset: str,
        overrides: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> str:
        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = record

        future = self._executor.submit(
            self._run, job_id, source, preset=preset, overrides=overrides, seed=seed
        )
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return None if record is None else _copy(record)

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation; ``False`` if the job is unknown
        or already finished (mirrors ``ManualCancelToken.cancel``'s
        idempotence — calling twice is harmless)."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state not in (JobState.PENDING, JobState.RUNNING):
                return False
            record.cancel_token.cancel()
            return True

    def _update_progress(
        self, job_id: str, *, stage_name: str, fraction_complete: float, running: bool
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.state = JobState.RUNNING
            record.current_stage = stage_name if running else None
            record.fraction_complete = fraction_complete

    def _run(
        self,
        job_id: str,
        source: bytes,
        *,
        preset: str,
        overrides: Mapping[str, Any] | None,
        seed: int,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.state = JobState.RUNNING
            cancel_token = record.cancel_token
        listener = _JobProgressListener(self, job_id)
        try:
            bundle = convert(
                source,
                preset=preset,
                overrides=overrides,
                seed=seed,
                on_progress=listener,
                cancel_token=cancel_token,
            )
        except CancelledError:
            with self._lock:
                record = self._jobs[job_id]
                record.state = JobState.CANCELLED
                record.current_stage = None
            return
        except EngineError as exc:
            with self._lock:
                record = self._jobs[job_id]
                record.state = JobState.FAILED
                record.error = str(exc)
                record.error_type = type(exc).__name__
                record.current_stage = None
            return
        with self._lock:
            record = self._jobs[job_id]
            record.state = JobState.SUCCEEDED
            record.fraction_complete = 1.0
            record.current_stage = None
            record.result = bundle

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _copy(record: JobRecord) -> JobRecord:
    """Shallow snapshot so callers reading outside the lock never observe a
    torn write from ``_run``/``_update_progress``."""
    return JobRecord(
        job_id=record.job_id,
        state=record.state,
        fraction_complete=record.fraction_complete,
        current_stage=record.current_stage,
        result=record.result,
        error=record.error,
        error_type=record.error_type,
        cancel_token=record.cancel_token,
    )
