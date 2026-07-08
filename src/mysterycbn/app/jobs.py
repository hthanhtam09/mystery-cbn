"""Job model shared by all adapters.

JobStore (the implementation) lives in mystery-cbn-api (the backend repository).
This module defines the protocol contracts only (JobState, JobSpec, JobStatus).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class JobState(enum.Enum):
    PENDING = enum.auto()
    RUNNING = enum.auto()
    SUCCEEDED = enum.auto()
    FAILED = enum.auto()
    CANCELLED = enum.auto()


@runtime_checkable
class JobSpec(Protocol):
    """Everything needed to run one conversion."""

    @property
    def source(self) -> Path | bytes: ...

    @property
    def preset(self) -> str: ...

    @property
    def overrides(self) -> Mapping[str, Any]:
        """Programmatic config-layer overrides (dotted keys)."""
        ...

    @property
    def seed(self) -> int: ...


@runtime_checkable
class JobStatus(Protocol):
    """Polled by the API job-status endpoint."""

    @property
    def job_id(self) -> str: ...

    @property
    def state(self) -> JobState: ...

    @property
    def fraction_complete(self) -> float: ...

    @property
    def current_stage(self) -> str | None: ...
