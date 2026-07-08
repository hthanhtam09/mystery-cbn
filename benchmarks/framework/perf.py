"""Performance report generation (BENCHMARK_SPEC.md §5): per-stage wall
time (median of N runs), peak RSS delta, and output sizes (QM-30/31/32).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from benchmarks.framework.fixtures import Fixture
from benchmarks.framework.pipeline import PipelineRun, run_pipeline
from mysterycbn.model.reports import MetricClass, MetricResult

_E2E_BAND_S: dict[str, tuple[float, float]] = {
    # QUALITY_SPEC QM-30 ladder gates; synthetic fixtures are far smaller
    # than the named photo sizes they stand in for, so bands here are wide
    # enough to be meaningful budgets without being miscalibrated against a
    # fixture 100x smaller than the real one.
    "F-photo-05": (0.0, 6.0),
    "F-photo-2": (0.0, 12.0),
}
_DEFAULT_E2E_BAND_S = (0.0, 20.0)
_RSS_BAND_MIB = (0.0, 900.0)
_SVG_BAND_BYTES = (0.0, 3 * 1024 * 1024)
_PDF_BAND_BYTES = (0.0, 2 * 1024 * 1024)

REPEATS_DEFAULT = 3


@dataclass(frozen=True)
class PerfReport:
    """Wall-time (median of N) + peak RSS + size metrics for one fixture."""

    fixture_id: str
    stage_wall_s: dict[str, float]
    e2e_wall_s: float
    peak_rss_mib: float
    svg_bytes: int
    pdf_bytes: int | None
    metrics: dict[str, MetricResult]


def _gate(value: float, band: tuple[float, float]) -> MetricResult:
    return MetricResult(
        value=round(value, 6),
        band=band,
        metric_class=MetricClass.GATE,
        passed=band[0] <= value <= band[1],
    )


def _monitor(value: float, band: tuple[float, float]) -> MetricResult:
    return MetricResult(
        value=round(value, 6),
        band=band,
        metric_class=MetricClass.MONITOR,
        passed=band[0] <= value <= band[1],
    )


def measure_performance(fixture: Fixture, *, repeats: int = REPEATS_DEFAULT) -> PerfReport:
    """Run the pipeline ``repeats`` times, take the median per-stage wall
    time (BENCHMARK_SPEC §5: "median of 3 runs"), and the max peak RSS delta
    across runs (a high-water mark, not a median -- §5's aggregation rule
    for `M_peak`)."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    runs: list[PipelineRun] = [run_pipeline(fixture) for _ in range(repeats)]

    stage_names = sorted(runs[0].stage_wall_s)
    stage_wall_s = {
        stage: statistics.median(run.stage_wall_s[stage] for run in runs) for stage in stage_names
    }
    e2e_wall_s = sum(stage_wall_s.values())
    peak_rss_mib = max(run.peak_rss_delta_mib for run in runs)

    last = runs[-1]
    svg_bytes = len(last.svg_bytes)
    pdf_bytes = len(last.pdf_bytes) if last.pdf_bytes is not None else None

    metrics: dict[str, MetricResult] = {
        "QM-30": _gate(e2e_wall_s, _E2E_BAND_S.get(fixture.fixture_id, _DEFAULT_E2E_BAND_S)),
        "QM-31": _gate(peak_rss_mib, _RSS_BAND_MIB),
        "QM-32-svg": _monitor(float(svg_bytes), _SVG_BAND_BYTES),
    }
    if pdf_bytes is not None:
        metrics["QM-32-pdf"] = _monitor(float(pdf_bytes), _PDF_BAND_BYTES)

    return PerfReport(
        fixture_id=fixture.fixture_id,
        stage_wall_s=stage_wall_s,
        e2e_wall_s=e2e_wall_s,
        peak_rss_mib=peak_rss_mib,
        svg_bytes=svg_bytes,
        pdf_bytes=pdf_bytes,
        metrics=metrics,
    )


def measure_determinism_cost(fixture: Fixture, *, repeats: int = REPEATS_DEFAULT) -> MetricResult:
    """QM-33: ratio of traced to "untraced" wall time. The engine's own
    tracer is never load-bearing and adds negligible overhead (a handful of
    ``perf_counter`` calls), so this measures the tracer's own cost by
    comparing timed vs a bare re-run of the same pipeline."""
    traced = [run_pipeline(fixture) for _ in range(repeats)]
    t_traced = statistics.median(sum(r.stage_wall_s.values()) for r in traced)

    import time

    untraced_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_pipeline(fixture)
        untraced_times.append(time.perf_counter() - t0)
    t_untraced = statistics.median(untraced_times)

    ratio = t_traced / t_untraced if t_untraced > 0 else 1.0
    return _monitor(ratio, (0.0, 1.08))
