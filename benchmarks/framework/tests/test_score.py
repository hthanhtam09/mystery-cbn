"""Unit tests for the Engine Score (BENCHMARK_SPEC §10.2)."""

from __future__ import annotations

from benchmarks.framework.score import compute_score
from mysterycbn.model.reports import MetricClass, MetricResult


def _metrics(qm_id: str, value: float, band: tuple[float, float]) -> dict:
    return {"F-x": {"medium": {qm_id: MetricResult(value, band, MetricClass.GATE, True)}}}


def test_score_is_100_when_every_dimension_hits_target() -> None:
    metrics = {
        "F-x": {
            "medium": {
                "QM-18": MetricResult(0.995, (0.99, 1.0), MetricClass.GATE, True),
                "QM-02": MetricResult(0.0, (0.0, 1e-4), MetricClass.GATE, True),
                "QM-11": MetricResult(0.0, (0.0, 0.0), MetricClass.GATE, True),
                "QM-16": MetricResult(12.0, (12.0, 1e18), MetricClass.GATE, True),
                "QM-30": MetricResult(12.0, (0.0, 15.0), MetricClass.GATE, True),
                "QM-31": MetricResult(600.0, (0.0, 900.0), MetricClass.GATE, True),
            }
        }
    }
    total, dims = compute_score(metrics)
    assert total == 100.0
    assert all(v == 1.0 for v in dims.values())


def test_score_defaults_to_neutral_when_metric_missing() -> None:
    total, dims = compute_score({"F-x": {"medium": {}}})
    assert total == 100.0
    assert all(v == 1.0 for v in dims.values())


def test_worse_metric_lowers_score() -> None:
    good, _ = compute_score(_metrics("QM-30", 12.0, (0.0, 15.0)))
    bad, _ = compute_score(_metrics("QM-30", 24.0, (0.0, 15.0)))
    assert bad < good


def test_score_is_bounded_0_to_100() -> None:
    total, dims = compute_score(_metrics("QM-02", 1.0, (0.0, 1e-4)))
    assert 0.0 <= total <= 100.0
    assert all(0.0 <= v <= 1.0 for v in dims.values())
