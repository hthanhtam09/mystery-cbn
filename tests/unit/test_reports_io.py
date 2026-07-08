"""Unit tests for the Sprint 23 metrics.json/report.json writers
(app/reports_io.py). Pure serialization of an existing OutputBundle -- no
new computation, no rendering."""

from __future__ import annotations

import json
from pathlib import Path

from mysterycbn.app.reports_io import write_bundle_reports, write_metrics_json, write_report_json
from mysterycbn.model.reports import (
    MetricClass,
    MetricResult,
    OutputBundle,
    QualityMetricsReport,
    RunReport,
    ValidationReport,
)


def _bundle() -> OutputBundle:
    validation = tuple(
        ValidationReport(name, (), {})
        for name in ("fidelity", "topology", "printability", "palette")
    )
    report = RunReport(
        resolved_config={"quantize": {"n_colors": 16}},
        engine_version="0.1.0",
        input_hash="ab" * 32,
        seed=0,
        warnings=(),
        stage_timings_s={"quantize": 1.2},
        validation=validation,
        renumber_map=(),
    )
    quality = QualityMetricsReport(
        metrics={
            "QM-13": MetricResult(300.0, (150.0, 1500.0), MetricClass.MONITOR, True),
            "printability_score": MetricResult(0.9, (0.5, 1.0), MetricClass.MONITOR, True),
        }
    )
    return OutputBundle(
        svg=b"<svg/>",
        pdf=None,
        previews={"lineart": b"x", "solved": b"y"},
        report=report,
        quality=quality,
    )


def test_write_metrics_json_matches_quality_to_dict(tmp_path: Path) -> None:
    bundle = _bundle()
    path = tmp_path / "metrics.json"
    write_metrics_json(bundle, path)
    written = json.loads(path.read_text())
    assert written == bundle.quality.to_dict()


def test_write_report_json_matches_run_report_to_dict(tmp_path: Path) -> None:
    bundle = _bundle()
    path = tmp_path / "report.json"
    write_report_json(bundle, path)
    written = json.loads(path.read_text())
    assert written == bundle.report.to_dict()


def test_write_bundle_reports_creates_both_files_in_output_dir(tmp_path: Path) -> None:
    bundle = _bundle()
    output_dir = tmp_path / "nested" / "output"
    metrics_path, report_path = write_bundle_reports(bundle, output_dir)

    assert metrics_path == output_dir / "metrics.json"
    assert report_path == output_dir / "report.json"
    assert metrics_path.is_file()
    assert report_path.is_file()
    assert json.loads(metrics_path.read_text()) == bundle.quality.to_dict()
    assert json.loads(report_path.read_text()) == bundle.report.to_dict()


def test_written_json_has_no_infinity_literal(tmp_path: Path) -> None:
    """MetricResult bands can carry the 1e18 sentinel for 'unbounded', never
    float('inf') -- Infinity is not valid JSON."""
    bundle = _bundle()
    path = tmp_path / "metrics.json"
    write_metrics_json(bundle, path)
    text = path.read_text()
    assert "Infinity" not in text
