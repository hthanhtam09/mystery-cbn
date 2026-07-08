"""Writes ``metrics.json`` and ``report.json`` for one ``OutputBundle``
(Sprint 23). Pure serialization -- no rendering, no new computation; both
files are ``OutputBundle.quality``/``OutputBundle.report`` already computed
by ``ConcreteOrchestrator.convert()``, written to disk on request.

Distinct from ``benchmarks/framework/exporters.py``'s ``report.json``: that
one is the ladder-wide BENCHMARK_SPEC §11 schema across many fixtures x
presets. This ``report.json`` is per-``convert()``-call, scoped to a single
``RunReport`` (config, timings, the 4 canonical validation reports).
"""

from __future__ import annotations

import json
from pathlib import Path

from mysterycbn.model.reports import OutputBundle


def write_metrics_json(bundle: OutputBundle, path: Path) -> None:
    """Sprint 23 quality metrics (region stats, tiny regions, boundary
    smoothness, compactness, palette quality, label fit/overlap rate,
    SVG/PDF validity, printability score) as a standalone JSON document."""
    path.write_text(json.dumps(bundle.quality.to_dict(), indent=2, sort_keys=True) + "\n")


def write_report_json(bundle: OutputBundle, path: Path) -> None:
    """The run's reproducibility record: resolved config, engine version,
    input hash, stage timings, and the 4 canonical validation reports."""
    path.write_text(json.dumps(bundle.report.to_dict(), indent=2, sort_keys=True) + "\n")


def write_bundle_reports(bundle: OutputBundle, output_dir: Path) -> tuple[Path, Path]:
    """Convenience: write both files into ``output_dir``, returning their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.json"
    write_metrics_json(bundle, metrics_path)
    write_report_json(bundle, report_path)
    return metrics_path, report_path
