"""Unit tests for the JSON/CSV/HTML exporters: valid JSON (no Infinity/NaN),
correct CSV row counts, and a well-formed HTML document."""

from __future__ import annotations

import json

from benchmarks.framework.exporters import (
    render_html_dashboard,
    write_csv_report,
    write_html_dashboard,
    write_json_report,
    write_stage_timings_csv,
)
from benchmarks.framework.report import build_report


def _sample_report():
    return build_report(suite="smoke", repeats=1)


def test_json_report_is_strictly_valid_json(tmp_path) -> None:
    report = _sample_report()
    path = tmp_path / "report.json"
    write_json_report(report, path)
    doc = json.loads(path.read_text())  # raises if Infinity/NaN leaked through
    assert doc["run_id"] == report.run_id
    assert "Infinity" not in path.read_text()
    assert "NaN" not in path.read_text()


def test_json_report_schema_shape(tmp_path) -> None:
    report = _sample_report()
    path = tmp_path / "report.json"
    write_json_report(report, path)
    doc = json.loads(path.read_text())
    for key in ("run_id", "timestamp_utc", "git_sha", "engine_version", "machine",
                "dataset_version", "score_version", "report_schema", "metrics",
                "stages", "golden", "score", "verdict"):  # fmt: skip
        assert key in doc


def test_csv_report_has_one_row_per_metric(tmp_path) -> None:
    report = _sample_report()
    path = tmp_path / "metrics.csv"
    write_csv_report(report, path)
    lines = path.read_text().strip().splitlines()
    n_metrics = sum(
        len(per_preset)
        for per_fixture in report.metrics.values()
        for per_preset in per_fixture.values()
    )
    assert len(lines) - 1 == n_metrics  # minus header


def test_stage_csv_has_one_row_per_stage_measurement(tmp_path) -> None:
    report = _sample_report()
    path = tmp_path / "stages.csv"
    write_stage_timings_csv(report, path)
    lines = path.read_text().strip().splitlines()
    n_stages = sum(
        len(per_preset)
        for per_fixture in report.stages.values()
        for per_preset in per_fixture.values()
    )
    assert len(lines) - 1 == n_stages


def test_html_dashboard_is_well_formed_and_contains_verdict() -> None:
    report = _sample_report()
    html = render_html_dashboard(report)
    assert html.strip().startswith("<!doctype html>")
    assert "</html>" in html
    assert report.run_id in html
    assert ("ACCEPTED" in html) or ("REJECTED" in html)


def test_html_dashboard_writes_to_disk(tmp_path) -> None:
    report = _sample_report()
    path = tmp_path / "dashboard.html"
    write_html_dashboard(report, path)
    assert path.is_file()
    assert path.read_text().startswith("<!doctype html>")
