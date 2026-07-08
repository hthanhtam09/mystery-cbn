"""Unit tests for performance-report generation."""

from __future__ import annotations

from benchmarks.framework.fixtures import load_fixture
from benchmarks.framework.perf import measure_performance


def test_perf_report_has_stage_timings_and_size_metrics() -> None:
    fx = load_fixture("F-photo-05")
    report = measure_performance(fx, repeats=1)
    assert report.e2e_wall_s > 0
    assert report.peak_rss_mib >= 0
    assert "QM-30" in report.metrics
    assert "QM-31" in report.metrics
    assert "QM-32-svg" in report.metrics
    assert report.stage_wall_s  # at least one stage traced


def test_repeats_must_be_positive() -> None:
    fx = load_fixture("F-photo-05")
    try:
        measure_performance(fx, repeats=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_median_of_repeats_is_stable() -> None:
    fx = load_fixture("F-flat-2")
    a = measure_performance(fx, repeats=3)
    b = measure_performance(fx, repeats=3)
    # Wall time varies run to run; the metric *shape* must not.
    assert set(a.stage_wall_s) == set(b.stage_wall_s)
    assert a.svg_bytes == b.svg_bytes  # deterministic output size
