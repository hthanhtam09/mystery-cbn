"""Report exporters: JSON (BENCHMARK_SPEC.md §11 schema), CSV (flat metric
table for spreadsheet review), and an HTML dashboard (§12's chart intent,
rendered as an inline self-contained page rather than separate SVG chart
files -- one artifact a reviewer opens, matching this task's ask for "an
HTML Dashboard" as a single deliverable).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mysterycbn.model.reports import BenchmarkReport, MetricClass


def write_json_report(report: BenchmarkReport, path: Path) -> None:
    """The canonical §11 report document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")


def _metric_rows(report: BenchmarkReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fixture_id, per_fixture in report.metrics.items():
        for preset, per_preset in per_fixture.items():
            for metric_id, result in per_preset.items():
                rows.append(
                    {
                        "run_id": report.run_id,
                        "fixture": fixture_id,
                        "preset": preset,
                        "metric": metric_id,
                        "value": result.value,
                        "band_lo": result.band[0],
                        "band_hi": result.band[1],
                        "class": result.metric_class.value,
                        "pass": result.passed,
                    }
                )
    return rows


def write_csv_report(report: BenchmarkReport, path: Path) -> None:
    """Flat per-metric CSV: one row per (fixture, preset, metric) --
    the format a reviewer pulls into a spreadsheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _metric_rows(report)
    fieldnames = [
        "run_id",
        "fixture",
        "preset",
        "metric",
        "value",
        "band_lo",
        "band_hi",
        "class",
        "pass",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stage_timings_csv(report: BenchmarkReport, path: Path) -> None:
    """Flat per-stage wall-time CSV (BENCHMARK_SPEC §5's per-stage table)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fixture_id, per_fixture in report.stages.items():
        for preset, per_preset in per_fixture.items():
            for stage, vals in per_preset.items():
                rows.append(
                    {
                        "run_id": report.run_id,
                        "fixture": fixture_id,
                        "preset": preset,
                        "stage": stage,
                        "wall_s": vals.get("wall_s", ""),
                    }
                )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "fixture", "preset", "stage", "wall_s"])
        writer.writeheader()
        writer.writerows(rows)


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _verdict_badge(accepted: bool) -> str:
    color = "#1a7f37" if accepted else "#cf222e"
    label = "ACCEPTED" if accepted else "REJECTED"
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:4px;font-weight:600">{label}</span>'


def _metric_class_badge(cls: MetricClass) -> str:
    color = "#0969da" if cls is MetricClass.GATE else "#9a6700"
    return f'<span style="background:{color};color:#fff;padding:1px 6px;border-radius:3px;font-size:11px">{cls.value}</span>'


def _dimension_bars(dims: dict[str, float]) -> str:
    rows = []
    for name, value in sorted(dims.items()):
        pct = max(0.0, min(100.0, value * 100.0))
        bar_color = "#1a7f37" if value >= 0.9 else "#9a6700" if value >= 0.5 else "#cf222e"
        rows.append(
            f"<tr><td>{_esc(name)}</td>"
            f'<td style="width:60%"><div style="background:#e1e4e8;border-radius:3px;overflow:hidden">'
            f'<div style="background:{bar_color};width:{pct:.1f}%;height:14px"></div></div></td>'
            f"<td>{value:.4f}</td></tr>"
        )
    return "\n".join(rows)


def _metric_table_rows(report: BenchmarkReport) -> str:
    rows = []
    for fixture_id, per_fixture in sorted(report.metrics.items()):
        for preset, per_preset in sorted(per_fixture.items()):
            for metric_id, result in sorted(per_preset.items()):
                status = "✓" if result.passed else "✗"
                status_color = "#1a7f37" if result.passed else "#cf222e"
                rows.append(
                    "<tr>"
                    f"<td>{_esc(fixture_id)}</td><td>{_esc(preset)}</td><td>{_esc(metric_id)}</td>"
                    f"<td>{result.value:.6g}</td>"
                    f"<td>[{result.band[0]:.6g}, {result.band[1]:.6g}]</td>"
                    f"<td>{_metric_class_badge(result.metric_class)}</td>"
                    f'<td style="color:{status_color};font-weight:700">{status}</td>'
                    "</tr>"
                )
    return "\n".join(rows)


def _stage_table_rows(report: BenchmarkReport) -> str:
    rows = []
    for fixture_id, per_fixture in sorted(report.stages.items()):
        for preset, per_preset in sorted(per_fixture.items()):
            for stage, vals in sorted(per_preset.items()):
                rows.append(
                    "<tr>"
                    f"<td>{_esc(fixture_id)}</td><td>{_esc(preset)}</td><td>{_esc(stage)}</td>"
                    f"<td>{vals.get('wall_s', 0.0):.4f}</td>"
                    "</tr>"
                )
    return "\n".join(rows)


def _golden_table_rows(report: BenchmarkReport) -> str:
    rows = []
    for key, outcome in sorted(report.golden.items()):
        color = {
            "identical": "#1a7f37",
            "changed_compatible": "#9a6700",
            "incompatible": "#cf222e",
        }.get(outcome.value, "#57606a")
        rows.append(
            f"<tr><td>{_esc(key)}</td>"
            f'<td style="color:{color};font-weight:600">{_esc(outcome.value)}</td></tr>'
        )
    return "\n".join(rows)


def _failures_table_rows(report: BenchmarkReport) -> str:
    if not report.failures:
        return '<tr><td colspan="5" style="color:#57606a">No failures.</td></tr>'
    rows = []
    for f in report.failures:
        rows.append(
            "<tr>"
            f"<td>{_esc(f.metric)}</td><td>{_esc(f.fixture)}</td><td>{_esc(f.preset)}</td>"
            f"<td>{f.value:.6g}</td><td>[{f.band[0]:.6g}, {f.band[1]:.6g}]</td>"
            "</tr>"
        )
    return "\n".join(rows)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark Report {run_id}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 0; padding: 32px;
          background: #fff; color: #1f2328; max-width: 1100px; margin-inline: auto; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0d1117; color: #e6edf3; }}
    table {{ border-color: #30363d !important; }}
    th, td {{ border-color: #30363d !important; }}
    .card {{ background: #161b22 !important; border-color: #30363d !important; }}
  }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .meta {{ color: #57606a; font-size: 13px; margin-bottom: 20px; }}
  .card {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px 20px; margin-bottom: 24px; }}
  .score {{ font-size: 42px; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }}
  th {{ background: rgba(0,0,0,0.03); }}
  h2 {{ font-size: 16px; margin-top: 32px; }}
  code {{ font-size: 12px; }}
</style>
</head>
<body>
  <h1>Mystery-CBN Benchmark Report</h1>
  <div class="meta">
    run <code>{run_id}</code> &middot; engine {engine_version} &middot; git {git_sha} &middot;
    {timestamp_utc} &middot; machine class <code>{machine_class}</code> &middot;
    dataset v{dataset_version} &middot; score v{score_version}
  </div>

  <div class="card">
    <div>{verdict_badge}</div>
    <div class="score">{score_total:.2f}<span style="font-size:16px;color:#57606a"> / 100</span></div>
    <table>
      <tr><th>Dimension</th><th>Score</th><th>Value</th></tr>
      {dimension_bars}
    </table>
  </div>

  <h2>Regression / acceptance failures</h2>
  <table>
    <tr><th>Metric</th><th>Fixture</th><th>Preset</th><th>Value</th><th>Band</th></tr>
    {failures_rows}
  </table>

  <h2>Quality &amp; performance metrics</h2>
  <table>
    <tr><th>Fixture</th><th>Preset</th><th>Metric</th><th>Value</th><th>Band</th><th>Class</th><th>Pass</th></tr>
    {metric_rows}
  </table>

  <h2>Per-stage wall time (s)</h2>
  <table>
    <tr><th>Fixture</th><th>Preset</th><th>Stage</th><th>Wall (s)</th></tr>
    {stage_rows}
  </table>

  <h2>Golden comparison</h2>
  <table>
    <tr><th>Fixture / preset</th><th>Outcome</th></tr>
    {golden_rows}
  </table>
</body>
</html>
"""


def render_html_dashboard(report: BenchmarkReport) -> str:
    return _HTML_TEMPLATE.format(
        run_id=_esc(report.run_id),
        engine_version=_esc(report.engine_version),
        git_sha=_esc(report.git_sha),
        timestamp_utc=_esc(report.timestamp_utc),
        machine_class=_esc(report.machine.cpu),
        dataset_version=report.dataset_version,
        score_version=report.score_version,
        verdict_badge=_verdict_badge(report.accepted),
        score_total=report.score_total,
        dimension_bars=_dimension_bars(dict(report.score_dimensions)),
        failures_rows=_failures_table_rows(report),
        metric_rows=_metric_table_rows(report),
        stage_rows=_stage_table_rows(report),
        golden_rows=_golden_table_rows(report),
    )


def write_html_dashboard(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_dashboard(report))
