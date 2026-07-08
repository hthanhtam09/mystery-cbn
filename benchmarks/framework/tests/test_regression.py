"""Unit tests for regression detection: baseline round-trip, within/outside
tolerance, and the Gate-vs-Monitor decision split (BENCHMARK_SPEC §7.2)."""

from __future__ import annotations

from benchmarks.framework.regression import (
    baseline_path,
    check_regressions,
    load_baselines,
    update_baseline,
)
from mysterycbn.model.reports import MetricClass, MetricResult


def test_baseline_round_trips_through_disk(tmp_path, monkeypatch) -> None:
    import benchmarks.framework.regression as regression_mod

    monkeypatch.setattr(regression_mod, "BASELINES_ROOT", tmp_path)
    update_baseline("F-x", "stage_wall_s", 2.5, run_id="run-1", machine_class_name="test-class")
    baselines = load_baselines("test-class")
    assert baselines["F-x"]["stage_wall_s"].value == 2.5
    assert baseline_path("test-class").is_file()


def test_gate_metric_fails_on_its_own_band_regardless_of_baseline() -> None:
    metrics = {"QM-01": MetricResult(1.0, (0.0, 0.0), MetricClass.GATE, passed=False)}
    failures = check_regressions(fixture_id="F-x", preset="medium", metrics=metrics, baselines={})
    assert len(failures) == 1
    assert failures[0].metric == "QM-01"


def test_monitor_metric_without_baseline_is_not_a_regression() -> None:
    metrics = {"QM-13": MetricResult(500.0, (150.0, 1500.0), MetricClass.MONITOR, passed=True)}
    failures = check_regressions(fixture_id="F-x", preset="medium", metrics=metrics, baselines={})
    assert failures == []


def test_monitor_metric_within_tolerance_passes(tmp_path, monkeypatch) -> None:
    import benchmarks.framework.regression as regression_mod

    monkeypatch.setattr(regression_mod, "BASELINES_ROOT", tmp_path)
    update_baseline(
        "F-x", "stage_wall_s", 1.0, run_id="run-0", tolerance=0.2, machine_class_name="t"
    )
    baselines = load_baselines("t")
    metrics = {"stage_wall_s": MetricResult(1.1, (0.0, 999.0), MetricClass.MONITOR, passed=True)}
    failures = check_regressions(
        fixture_id="F-x", preset="medium", metrics=metrics, baselines=baselines
    )
    assert failures == []


def test_monitor_metric_outside_tolerance_is_flagged(tmp_path, monkeypatch) -> None:
    import benchmarks.framework.regression as regression_mod

    monkeypatch.setattr(regression_mod, "BASELINES_ROOT", tmp_path)
    update_baseline(
        "F-x", "stage_wall_s", 1.0, run_id="run-0", tolerance=0.2, machine_class_name="t"
    )
    baselines = load_baselines("t")
    metrics = {"stage_wall_s": MetricResult(1.5, (0.0, 999.0), MetricClass.MONITOR, passed=True)}
    failures = check_regressions(
        fixture_id="F-x", preset="medium", metrics=metrics, baselines=baselines
    )
    assert len(failures) == 1
    assert failures[0].value == 1.5


def test_baselines_change_only_via_explicit_update(tmp_path, monkeypatch) -> None:
    """Reading + checking never writes a baseline back (§7.1: explicit only)."""
    import benchmarks.framework.regression as regression_mod

    monkeypatch.setattr(regression_mod, "BASELINES_ROOT", tmp_path)
    assert not baseline_path("t2").exists()
    baselines = load_baselines("t2")
    metrics = {"stage_wall_s": MetricResult(1.0, (0.0, 999.0), MetricClass.MONITOR, passed=True)}
    check_regressions(fixture_id="F-x", preset="medium", metrics=metrics, baselines=baselines)
    assert not baseline_path("t2").exists()


def test_save_baselines_is_reviewable_json(tmp_path, monkeypatch) -> None:
    import benchmarks.framework.regression as regression_mod

    monkeypatch.setattr(regression_mod, "BASELINES_ROOT", tmp_path)
    update_baseline("F-x", "QM-13", 640.0, run_id="run-2", machine_class_name="t3")
    text = baseline_path("t3").read_text()
    assert '"F-x"' in text
    assert '"QM-13"' in text
