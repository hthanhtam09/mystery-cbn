"""Per-run tracing: stage timings, artifact sizes, debug artifacts (ARCHITECTURE.md §12).

Tracing is never load-bearing — output must be identical with tracing disabled (I2).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from types import MappingProxyType
from typing import Any


class Tracer(ABC):
    """Collects per-stage timings and metrics into the run report."""

    @abstractmethod
    def span(self, stage_name: str) -> AbstractContextManager[None]:
        """Time a stage execution as a named span."""

    @abstractmethod
    def record_metric(self, stage_name: str, key: str, value: float) -> None:
        """Attach a scalar metric to a stage span."""

    @abstractmethod
    def record_artifact_size(self, artifact_name: str, size_bytes: int) -> None:
        """Record the serialized size of a produced artifact."""

    @abstractmethod
    def snapshot(self) -> Mapping[str, Any]:
        """Return an immutable view of everything collected so far."""


class InMemoryTracer(Tracer):
    """Default tracer: monotonic-clock spans and metrics in process memory.

    Never load-bearing (I2): consumers embed the snapshot in the run report;
    disabling tracing must not change any output byte.
    """

    def __init__(self) -> None:
        self._timings: dict[str, float] = {}
        self._metrics: dict[str, dict[str, float]] = {}
        self._artifact_sizes: dict[str, int] = {}

    @contextmanager
    def _span(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings[stage_name] = self._timings.get(stage_name, 0.0) + (
                time.perf_counter() - start
            )

    def span(self, stage_name: str) -> AbstractContextManager[None]:
        return self._span(stage_name)

    def record_metric(self, stage_name: str, key: str, value: float) -> None:
        self._metrics.setdefault(stage_name, {})[key] = value

    def record_artifact_size(self, artifact_name: str, size_bytes: int) -> None:
        self._artifact_sizes[artifact_name] = size_bytes

    def snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "timings_s": MappingProxyType(dict(self._timings)),
                "metrics": MappingProxyType(
                    {k: MappingProxyType(dict(v)) for k, v in self._metrics.items()}
                ),
                "artifact_sizes": MappingProxyType(dict(self._artifact_sizes)),
            }
        )
