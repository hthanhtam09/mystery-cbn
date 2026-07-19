"""Unit tests for report/bundle objects (DATA_MODEL_SPEC §18–§20)."""

from __future__ import annotations

import json

import pytest

from mysterycbn.model.reports import (
    BenchmarkReport,
    FailureTuple,
    Finding,
    GoldenOutcome,
    MachineFingerprint,
    MetricClass,
    MetricResult,
    OutputBundle,
    QualityMetricsReport,
    RunReport,
    Severity,
    ValidationReport,
)

_PREVIEWS_OK = {"lineart": b"x", "solved": b"y", "colored": b"z", "palette": b"p"}


def _quality_report() -> QualityMetricsReport:
    return QualityMetricsReport(metrics={})


def _finding(severity: Severity = Severity.WARNING) -> Finding:
    return Finding(severity, "I3", "gap detected", "arc 12")


def test_validation_report_passed_is_derived() -> None:
    ok = ValidationReport("topology", (_finding(Severity.INFO),), {"residual": 0.0})
    assert ok.passed
    bad = ValidationReport("topology", (_finding(Severity.FATAL),), {})
    assert not bad.passed
    assert json.dumps(bad.to_dict())


def _passing_reports() -> tuple[ValidationReport, ...]:
    return tuple(
        ValidationReport(name, (), {})
        for name in ("fidelity", "topology", "printability", "palette")
    )


def _run_report(validation: tuple[ValidationReport, ...]) -> RunReport:
    return RunReport(
        resolved_config={"quantize": {"n_colors": 16}},
        engine_version="0.1.0",
        input_hash="ab" * 32,
        seed=0,
        warnings=("auto-tune set preprocess.smooth_passes=3",),
        stage_timings_s={"quantize": 1.2},
        validation=validation,
        renumber_map=(1, 0),
    )


def test_run_report_to_dict_is_json_safe_with_nested_mappingproxy() -> None:
    """RunReport.resolved_config commonly comes from LayeredResolver.as_mapping(),
    a MappingProxyType tree nested arbitrarily deep -- to_dict() must
    recursively convert it, not just the top level (regression: a shallow
    dict(self.resolved_config) leaves nested mappingproxy values, which
    json.dumps cannot serialize)."""
    from types import MappingProxyType

    nested = MappingProxyType({"quantize": MappingProxyType({"n_colors": 16})})
    report = RunReport(
        resolved_config=nested,
        engine_version="0.1.0",
        input_hash="ab" * 32,
        seed=0,
        warnings=(),
        stage_timings_s={},
        validation=_passing_reports(),
        renumber_map=(),
    )
    assert json.dumps(report.to_dict())
    assert report.to_dict()["resolved_config"] == {"quantize": {"n_colors": 16}}


def test_output_bundle_atomicity_rules() -> None:
    quality = _quality_report()
    bundle = OutputBundle(
        svg=b"<svg/>",
        pdf=None,
        previews=_PREVIEWS_OK,
        report=_run_report(_passing_reports()),
        quality=quality,
    )
    assert json.dumps(bundle.to_dict())
    with pytest.raises(ValueError, match="non-empty"):
        OutputBundle(b"", None, _PREVIEWS_OK, bundle.report, quality)
    with pytest.raises(ValueError, match="keys"):
        OutputBundle(b"<svg/>", None, {"lineart": b"x"}, bundle.report, quality)
    failed = (*_passing_reports()[:3], ValidationReport("palette", (_finding(Severity.FATAL),), {}))
    with pytest.raises(ValueError, match="every validator passed"):
        OutputBundle(
            b"<svg/>", None, _PREVIEWS_OK, _run_report(failed), quality
        )
    with pytest.raises(ValueError, match="exactly 4"):
        OutputBundle(
            b"<svg/>",
            None,
            _PREVIEWS_OK,
            _run_report(_passing_reports()[:2]),
            quality,
        )


def _benchmark_report(failures: tuple[FailureTuple, ...] = ()) -> BenchmarkReport:
    return BenchmarkReport(
        run_id="run-1",
        timestamp_utc="2026-07-06T00:00:00Z",
        git_sha="deadbeef",
        engine_version="0.1.0",
        machine=MachineFingerprint("cpu", 8, 32.0, "sha256:abc", "6.1", "lock", 1.5),
        dataset_version=1,
        score_version=1,
        report_schema=3,
        metrics={
            "F-photo-2": {
                "medium": {"QM-17": MetricResult(0.991, (0.985, 1.0), MetricClass.GATE, True)}
            }
        },
        stages={"F-photo-2": {"medium": {"quantize": {"wall_s": 1.2, "rss_mib": 220.0}}}},
        golden={"F-photo-2/medium": GoldenOutcome.IDENTICAL},
        score_total=97.3,
        score_dimensions={"fidelity": 0.999},
        accepted=len(failures) == 0,
        failures=failures,
    )


def test_benchmark_report_roundtrips_to_json() -> None:
    report = _benchmark_report()
    doc = report.to_dict()
    assert json.dumps(doc)
    assert doc["verdict"] == {"accepted": True, "failures": []}
    assert doc["golden"] == {"F-photo-2/medium": "identical"}


def test_benchmark_report_verdict_consistency() -> None:
    fail = FailureTuple("QM-17", "F-photo-2", "medium", 0.97, (0.985, 1.0))
    ok = _benchmark_report((fail,))
    assert not ok.accepted
    with pytest.raises(ValueError, match="iff"):
        BenchmarkReport(
            **{
                **ok.__dict__,
                "accepted": True,  # inconsistent with non-empty failures
            }
        )


def test_metric_result_band_ordering() -> None:
    with pytest.raises(ValueError, match="lo ≤ hi"):
        MetricResult(1.0, (2.0, 1.0), MetricClass.MONITOR, True)
