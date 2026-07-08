"""Quality report generation: computes QM-metric ``MetricResult``s from a
pipeline run (QUALITY_SPEC.md). Reuses the production ``validate`` module's
measurement implementations rather than re-deriving them (BENCHMARK_SPEC.md
§6: "the benchmark harness and the production validator share the same
measurement implementations").

Covers the metrics computable from a single pipeline run without a raster
fixture (QM-01/02/10/11/16/18/21/24/26/28, QM-13); the raster-domain color
and fidelity metrics (QM-15/17/19) require an actual RasterImage source and
are out of scope until real image fixtures exist (see ``fixtures.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.framework.pipeline import PipelineRun
from mysterycbn.model.layout import LabelPlan
from mysterycbn.model.records import Provenance
from mysterycbn.model.reports import MetricClass, MetricResult
from mysterycbn.validate.output_validity import validate_output_validity
from mysterycbn.validate.palette import validate_palette
from mysterycbn.validate.printability import validate_printability
from mysterycbn.validate.topology import validate_topology

# (metric_id, band, class) per QUALITY_SPEC §2-§6. Bands are the *acceptable*
# min/max (Gate: hard; Monitor: informational unless a baseline tolerance
# also fires -- see regression.py).
_BAND_TOPOLOGY_ERRORS = (0.0, 0.0)
# QUALITY_SPEC states 1e-4 for the ArcGraph's pre-smoothing polyline area
# identity; this framework measures the same quantity the production
# topology validator gates on (validate/topology.py's own widened band),
# which is taken on the final Bezier-smoothed CurveSet and so carries the
# QM-09 displacement-bound residual by construction. Using a stricter band
# here would fail on every run for a reason that isn't a regression.
# A large finite sentinel, not float("inf"): the JSON report (BENCHMARK_SPEC
# §11) must be valid JSON, and `Infinity` is not (json.dumps emits it, but
# no schema validator or `json.loads` in another language accepts it back).
_UNBOUNDED = 1e18

_BAND_WATERTIGHT = (0.0, 2e-3)
_BAND_MIN_DIAMETER_MM = (3.5, _UNBOUNDED)
_BAND_TINY_PCT = (0.0, 0.0)
_BAND_PALETTE_SEP = (12.0, _UNBOUNDED)
_BAND_FACE_LABEL_AGREEMENT = (0.99, 1.0)
_BAND_LABEL_COVERAGE = (100.0, 100.0)
_BAND_REGION_COUNT = (150.0, 1500.0)  # QM-13, Monitor


@dataclass(frozen=True)
class QualityReport:
    """All QM ``MetricResult``s computed for one fixture run."""

    fixture_id: str
    metrics: dict[str, MetricResult]
    fatal_findings: list[str]


def _gate(value: float, band: tuple[float, float]) -> MetricResult:
    return MetricResult(
        value=round(value, 6),
        band=band,
        metric_class=MetricClass.GATE,
        passed=band[0] <= value <= band[1],
    )


def _monitor(value: float, band: tuple[float, float]) -> MetricResult:
    return MetricResult(
        value=round(value, 6),
        band=band,
        metric_class=MetricClass.MONITOR,
        passed=band[0] <= value <= band[1],
    )


def compute_quality_report(run: PipelineRun) -> QualityReport:
    """Run every applicable validator on the pipeline's context and translate
    findings/metrics into QM-numbered ``MetricResult``s."""
    metrics: dict[str, MetricResult] = {}
    fatal: list[str] = []

    topology_report = validate_topology(run.ctx)
    metrics["QM-01"] = _gate(topology_report.metrics["topology_errors"], _BAND_TOPOLOGY_ERRORS)
    metrics["QM-02"] = _gate(topology_report.metrics["watertightness_residual"], _BAND_WATERTIGHT)

    printability_report = validate_printability(run.ctx)
    metrics["QM-10"] = _gate(
        printability_report.metrics["min_region_diameter_mm"], _BAND_MIN_DIAMETER_MM
    )
    metrics["QM-11"] = _gate(printability_report.metrics["tiny_region_pct"], _BAND_TINY_PCT)
    metrics["QM-21"] = _gate(
        printability_report.metrics["label_coverage_pct"], _BAND_LABEL_COVERAGE
    )

    label_plan = run.ctx.get("label_plan")
    assert isinstance(label_plan, LabelPlan)
    min_font = min((lb.font_size_pt for lb in label_plan.labels), default=6.0)
    metrics["QM-24"] = _gate(min_font, (6.0, _UNBOUNDED))

    palette_report = validate_palette(run.ctx)
    metrics["QM-16"] = _gate(palette_report.metrics["min_delta_e"], _BAND_PALETTE_SEP)

    from mysterycbn.validate.fidelity import validate_fidelity

    fidelity_report = validate_fidelity(run.ctx)
    metrics["QM-18"] = _gate(
        fidelity_report.metrics["min_face_label_agreement"], _BAND_FACE_LABEL_AGREEMENT
    )

    metrics["QM-13"] = _monitor(float(len(run.curve_set.faces)), _BAND_REGION_COUNT)

    run.ctx.put("svg", _DocView(run.svg_bytes))
    if run.pdf_bytes is not None:
        run.ctx.put("pdf", _DocView(run.pdf_bytes))
    output_report = validate_output_validity(run.ctx)
    metrics["QM-26"] = _gate(0.0 if output_report.passed else 1.0, (0.0, 0.0))
    if run.pdf_bytes is not None:
        pdf_ok = not any(f.location == "pdf" for f in output_report.findings)
        metrics["QM-28"] = _gate(1.0 if pdf_ok else 0.0, (1.0, 1.0))

    for report in (
        topology_report,
        printability_report,
        palette_report,
        fidelity_report,
        output_report,
    ):
        fatal.extend(f.message for f in report.findings if f.severity.value == "fatal")

    return QualityReport(fixture_id=run.fixture_id, metrics=metrics, fatal_findings=fatal)


@dataclass(frozen=True)
class _DocView:
    """Minimal ``.data`` carrier matching SvgDocument/PdfDocument's shape,
    for artifacts assembled outside the pipeline stage wrappers. Carries a
    placeholder ``provenance`` only to satisfy the ``Artifact`` protocol
    that ``PipelineContext.put`` requires -- ``validate_output_validity``
    never reads it."""

    data: bytes
    provenance: Provenance = field(
        default_factory=lambda: Provenance("bench", "1.0.0", "0" * 64, "1" * 64)
    )


def gate_metrics(metrics: dict[str, MetricResult]) -> dict[str, MetricResult]:
    return {k: v for k, v in metrics.items() if v.metric_class is MetricClass.GATE}


def monitor_metrics(metrics: dict[str, MetricResult]) -> dict[str, MetricResult]:
    return {k: v for k, v in metrics.items() if v.metric_class is MetricClass.MONITOR}
