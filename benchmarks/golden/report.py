"""Golden-test quality report generation (docs/GOLDEN_TEST_STANDARDS.md §7).

Produces one JSON-serializable report per run covering every fixture
compared, mirroring BENCHMARK_SPEC.md §11's report-schema discipline
(machine-readable, one file per run, pass/fail is derived not eyeballed).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from benchmarks.datasets.loaders import DatasetFixture
from benchmarks.framework.pipeline import PipelineRun
from benchmarks.golden.compare import GoldenReport, compare_run_to_golden
from benchmarks.golden.runner import run_full_ladder, run_golden_ladder
from benchmarks.golden.tolerances import DEFAULT_TOLERANCES, GoldenTolerances
from mysterycbn import __version__ as ENGINE_VERSION


@dataclass(frozen=True)
class GoldenSuiteReport:
    """All per-fixture ``GoldenReport``s for one golden-test run."""

    run_id: str
    engine_version: str
    timestamp: float
    tolerances: GoldenTolerances
    fixture_reports: tuple[GoldenReport, ...]

    @property
    def accepted(self) -> bool:
        return all(r.passed for r in self.fixture_reports)

    @property
    def failures(self) -> tuple[GoldenReport, ...]:
        return tuple(r for r in self.fixture_reports if not r.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "engine_version": self.engine_version,
            "timestamp": self.timestamp,
            "accepted": self.accepted,
            "tolerances": self.tolerances.to_dict(),
            "fixtures": [r.to_dict() for r in self.fixture_reports],
            "summary": {
                "total": len(self.fixture_reports),
                "passed": sum(1 for r in self.fixture_reports if r.passed),
                "failed": len(self.failures),
            },
        }


def _build_report(
    pairs: Iterable[tuple[DatasetFixture, PipelineRun]], *, tolerances: GoldenTolerances
) -> GoldenSuiteReport:
    reports = [
        compare_run_to_golden(run, category=fx.category, tolerances=tolerances) for fx, run in pairs
    ]
    return GoldenSuiteReport(
        run_id=uuid.uuid4().hex[:12],
        engine_version=ENGINE_VERSION,
        timestamp=time.time(),
        tolerances=tolerances,
        fixture_reports=tuple(reports),
    )


def run_golden_suite(*, tolerances: GoldenTolerances = DEFAULT_TOLERANCES) -> GoldenSuiteReport:
    """Run the frozen one-per-category golden subset and compare to stored goldens."""
    return _build_report(run_golden_ladder(), tolerances=tolerances)


def run_full_dataset_suite(
    *, tolerances: GoldenTolerances = DEFAULT_TOLERANCES
) -> GoldenSuiteReport:
    """Run every dataset fixture and compare to stored goldens."""
    return _build_report(run_full_ladder(), tolerances=tolerances)
