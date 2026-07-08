"""Unit tests for full report assembly, including the two-generation
regression-detection demo: seed a baseline, then prove a worse run is
caught by ``check_regressions`` through the full ``build_report`` path."""

from __future__ import annotations

from benchmarks.framework.regression import update_baseline
from benchmarks.framework.report import build_report
from mysterycbn.model.reports import BenchmarkReport


def test_smoke_report_is_a_valid_benchmark_report() -> None:
    report = build_report(suite="smoke", repeats=1)
    assert isinstance(report, BenchmarkReport)
    assert report.dataset_version >= 1
    assert set(report.metrics.keys()) == {"F-photo-05", "F-flat-2"}


def test_full_report_covers_every_fixture() -> None:
    report = build_report(suite="full", repeats=1)
    from benchmarks.framework.fixtures import available_fixture_ids

    assert set(report.metrics.keys()) == set(available_fixture_ids())


def test_thin_structure_fixture_causes_full_suite_rejection() -> None:
    """F-thin-2's Gate violation must propagate all the way to the
    top-level accepted/failures verdict, not just the per-metric report."""
    report = build_report(suite="full", repeats=1)
    assert not report.accepted
    assert any(f.fixture == "F-thin-2" for f in report.failures)


def test_regression_against_a_seeded_baseline_is_detected(tmp_path, monkeypatch) -> None:
    """The full regression-detection loop: seed an artificially tight
    baseline for QM-32-svg (output size, Monitor), then prove a real run
    against it is flagged when the current output exceeds tolerance."""
    import benchmarks.framework.regression as regression_mod

    monkeypatch.setattr(regression_mod, "BASELINES_ROOT", tmp_path)
    monkeypatch.setattr(regression_mod, "machine_class", lambda: "test-class")

    # Seed a baseline far below any real SVG size, tolerance 1% -- any real
    # run's actual output size will overshoot it.
    update_baseline("F-photo-05", "QM-32-svg", 10.0, run_id="seed", tolerance=0.01)

    report = build_report(suite="smoke", repeats=1)
    assert not report.accepted
    assert any(f.metric == "QM-32-svg" and f.fixture == "F-photo-05" for f in report.failures)
