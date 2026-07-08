"""Unit tests for quality-report generation: correct QM ids, correct Gate
vs Monitor classification, and a fixture purpose-built to fail a Gate."""

from __future__ import annotations

from benchmarks.framework.fixtures import load_fixture
from benchmarks.framework.pipeline import run_pipeline
from benchmarks.framework.quality import compute_quality_report, gate_metrics, monitor_metrics
from mysterycbn.model.reports import MetricClass


def test_quality_report_covers_expected_qm_ids() -> None:
    fx = load_fixture("F-photo-05")
    run = run_pipeline(fx)
    report = compute_quality_report(run)
    expected = {
        "QM-01", "QM-02", "QM-10", "QM-11", "QM-13",
        "QM-16", "QM-18", "QM-21", "QM-24", "QM-26", "QM-28",
    }  # fmt: skip
    assert expected.issubset(report.metrics.keys())


def test_gate_and_monitor_split() -> None:
    fx = load_fixture("F-flat-2")
    run = run_pipeline(fx)
    report = compute_quality_report(run)
    gates = gate_metrics(report.metrics)
    monitors = monitor_metrics(report.metrics)
    assert set(gates) | set(monitors) == set(report.metrics)
    assert set(gates) & set(monitors) == set()
    assert gates["QM-01"].metric_class is MetricClass.GATE
    assert monitors["QM-13"].metric_class is MetricClass.MONITOR


def test_clean_fixture_passes_every_gate() -> None:
    """F-flat-2 has no sub-floor regions, so every printability/topology
    gate must pass -- a purpose-built "known good" case."""
    fx = load_fixture("F-flat-2")
    run = run_pipeline(fx)
    report = compute_quality_report(run)
    for metric_id, result in gate_metrics(report.metrics).items():
        assert result.passed, (
            f"{metric_id} unexpectedly failed: {result.value} not in {result.band}"
        )


def test_thin_structure_fixture_fails_printability_gate() -> None:
    """F-thin-2's 2px lines are below the printability floor by
    construction -- this proves the harness actually catches a real
    violation rather than always reporting green."""
    fx = load_fixture("F-thin-2")
    run = run_pipeline(fx)
    report = compute_quality_report(run)
    assert not report.metrics["QM-10"].passed or not report.metrics["QM-11"].passed
