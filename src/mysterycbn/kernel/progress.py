"""Progress events consumed by the API job-status endpoint and future GUIs."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class ProgressKind(enum.Enum):
    STAGE_STARTED = enum.auto()
    STAGE_FINISHED = enum.auto()


@runtime_checkable
class ProgressEvent(Protocol):
    @property
    def kind(self) -> ProgressKind: ...

    @property
    def stage_name(self) -> str: ...

    @property
    def fraction_complete(self) -> float:
        """Overall pipeline completion in [0, 1]."""
        ...


@runtime_checkable
class ProgressListener(Protocol):
    def on_progress(self, event: ProgressEvent) -> None: ...


@dataclass(frozen=True)
class ProgressUpdate:
    """Concrete progress event emitted by the executor."""

    kind: ProgressKind
    stage_name: str
    fraction_complete: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction_complete <= 1.0:
            raise ValueError(f"fraction_complete must be in [0, 1], got {self.fraction_complete}")
