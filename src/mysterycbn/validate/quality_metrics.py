"""Quality Metrics Validator (Sprint 23, QUALITY_SPEC.md): measures output
quality and reports it -- unlike the four canonical validators (fidelity,
topology, printability, palette), this validator never blocks a run. It
reads the same final artifacts the canonical validators already established
as correct (an ``OutputBundle`` only exists once those four passed) and
reports Monitor-class measurements for visibility: region statistics, tiny
regions, boundary smoothness, mean compactness, palette quality, label fit
rate, label overlap rate, SVG/PDF validity, and a printability score.

No rendering or engine-stage code changes: every metric here is derived
independently from bound artifacts (``curve_set``, ``label_plan``,
``palette``, ``svg``, ``pdf``), following the same "independent
double-entry" convention the canonical validators use (ARCHITECTURE.md
§0) -- this module re-flattens Bézier geometry itself rather than trusting
construction-time areas/counts.

Findings are always INFO/WARNING, never FATAL -- ``QualityMetricsReport``
is a measurement, not a gate (that distinction is exactly QUALITY_SPEC
§1.2's Gate vs Monitor split; every metric here is Monitor-class except the
two structural-validity flags, which mirror QM-26/28's existing Gate
verdicts without re-raising).
"""

from __future__ import annotations

import math

import numpy as np

from mysterycbn.foundation.errors import StageError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.layout import LabelMode, LabelPlan
from mysterycbn.model.records import Palette
from mysterycbn.model.reports import MetricClass, MetricResult
from mysterycbn.model.vector import CurveSet
from mysterycbn.render.pdf import validate_pdf
from mysterycbn.render.svg import validate_svg
from mysterycbn.stages.layout.labels import _bbox_rect, _rects_overlap, text_bbox_pt
from mysterycbn.validate.common import (
    face_area_pt2,
    flatten_arc_polyline,
    flatten_face_rings,
    point_in_rings,
)

VALIDATOR_NAME = "quality_metrics"
_FLATTEN_MM = 0.1
_MM_TO_PT = 72.0 / 25.4
_UNBOUNDED = 1e18
_CORNER_THRESHOLD_RAD = math.radians(60.0)


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


def _ring_perimeter_pt(ring: np.ndarray) -> float:
    edges = np.roll(ring, -1, axis=0) - ring
    return float(np.sum(np.linalg.norm(edges, axis=1)))


def _region_stats(curve_set: CurveSet, tolerance_pt: float) -> dict[str, MetricResult]:
    """QM-13 (region count band) and mean compactness (QM-14): both
    independently re-derived from flattened final geometry, not the
    construction-time raster ``RegionGraph`` (post-merge/post-simplify
    counts are what actually ships)."""
    faces = curve_set.faces
    n = len(faces)
    compactness_values = []
    for face in faces:
        rings = flatten_face_rings(face, curve_set, tolerance_pt)
        area = abs(face_area_pt2(face, curve_set, tolerance_pt))
        perimeter = sum(_ring_perimeter_pt(r) for r in rings)
        if perimeter <= 0.0:
            continue
        compactness_values.append(4.0 * math.pi * area / (perimeter**2))
    mean_compactness = float(np.mean(compactness_values)) if compactness_values else 0.0
    return {
        "QM-13": _monitor(float(n), (150.0, 1500.0)),
        "QM-14": _monitor(mean_compactness, (0.25, 1.0)),
    }


def _tiny_region_pct(printability_metrics: dict[str, float] | None) -> MetricResult:
    """QM-11: reused verbatim from ``printability``'s own measurement (that
    validator already independently re-derives every face's inscribed
    diameter against ``d_min_mm`` -- recomputing it here would just be a
    second, less precise copy of the same geometry)."""
    pct = printability_metrics.get("tiny_region_pct", 0.0) if printability_metrics else 0.0
    return _gate(pct, (0.0, 0.0))


def _boundary_smoothness(curve_set: CurveSet, tolerance_pt: float) -> MetricResult:
    """QM-08: scale-invariant angular noise. Flatten every arc, sum
    turn-angle² at interior vertices (corners > 60° excluded as intentional,
    not noise), normalize by total flattened length in mm. Sums over each
    arc's own interior vertices per QUALITY_SPEC's ``Σ_arcs Σ_k θ_k²``
    formula -- whether the originating arc is closed doesn't change which
    vertices are interior to *this* polyline, so ``Curve`` need not carry
    that flag (it lives on ``vector.Arc``, one layer up)."""
    total_energy = 0.0
    total_length_mm = 0.0
    for curve in curve_set.curves:
        control_chain = [seg.control for seg in curve.segments]
        pts = flatten_arc_polyline(control_chain, tolerance_pt)
        if len(pts) < 3:
            continue
        edges = np.diff(pts, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        total_length_mm += float(np.sum(lengths)) / _MM_TO_PT
        cross = edges[:-1, 0] * edges[1:, 1] - edges[:-1, 1] * edges[1:, 0]
        dot = np.einsum("ij,ij->i", edges[:-1], edges[1:])
        theta = np.arctan2(cross, dot)
        theta = theta[np.abs(theta) <= _CORNER_THRESHOLD_RAD]
        total_energy += float(np.sum(theta**2))
    value = total_energy / total_length_mm if total_length_mm > 0.0 else 0.0
    return _monitor(value, (0.0, 0.09))


def _palette_quality(palette: Palette) -> MetricResult:
    """QM-16 proxy: minimum pairwise ΔE00 separation, reusing the palette's
    own cached ΔE table rather than recomputing it (the table is itself a
    cached derived value, not a construction-time shortcut -- ΔE00 has one
    correct formula, so there is nothing to independently re-derive here)."""
    k = palette.size
    if k < 2:
        return _gate(_UNBOUNDED, (12.0, _UNBOUNDED))
    off_diagonal = palette.delta_e_table[~np.eye(k, dtype=bool)]
    min_delta_e = float(off_diagonal.min())
    return _gate(min_delta_e, (12.0, _UNBOUNDED))


def _label_fit_rate(
    curve_set: CurveSet, label_plan: LabelPlan, tolerance_pt: float
) -> MetricResult:
    """QM-22: fraction of labels whose printed bounding box fits entirely
    within its own face's rings (in-region placements only -- a leader-mode
    label is definitionally outside its face, so it is excluded from the
    denominator rather than counted as a miss)."""
    faces_by_id = {face.face_id: face for face in curve_set.faces}
    in_region_labels = [label for label in label_plan.labels if label.mode is LabelMode.IN_REGION]
    if not in_region_labels:
        return _monitor(100.0, (90.0, 100.0))

    fitted = 0
    for label in in_region_labels:
        face = faces_by_id.get(label.region_id)
        if face is None:
            continue
        rings = flatten_face_rings(face, curve_set, tolerance_pt)
        w, h = text_bbox_pt(label.printed_number, label.font_size_pt)
        cx, cy = label.anchor
        corners = [
            np.array([cx - w / 2, cy - h / 2]),
            np.array([cx + w / 2, cy - h / 2]),
            np.array([cx - w / 2, cy + h / 2]),
            np.array([cx + w / 2, cy + h / 2]),
        ]
        if all(point_in_rings(corner, rings) for corner in corners):
            fitted += 1
    rate_pct = 100.0 * fitted / len(in_region_labels)
    return _monitor(rate_pct, (90.0, 100.0))


def _label_overlap_rate(label_plan: LabelPlan) -> MetricResult:
    """Label-vs-label overlap rate: fraction of placed labels whose printed
    bounding box overlaps another label's (a rate generalization of QM-20's
    raw collision count), independently re-derived via the same
    ``_bbox_rect``/``_rects_overlap`` primitives the labels stage itself
    uses to avoid collisions at placement time."""
    labels = label_plan.labels
    if len(labels) < 2:
        return _gate(0.0, (0.0, 0.0))
    rects = [_bbox_rect(label.anchor, label.printed_number, label.font_size_pt) for label in labels]
    overlapping = set()
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rects_overlap(rects[i], rects[j]):
                overlapping.add(i)
                overlapping.add(j)
    rate_pct = 100.0 * len(overlapping) / len(labels)
    return _gate(rate_pct, (0.0, 0.0))


def _svg_validity(ctx: PipelineContext) -> MetricResult:
    """QM-26, surfaced as a metric here (the gate itself lives in
    ``output_validity``, whose report ``OutputBundle`` doesn't embed --
    this restates the same pass/fail as a monitorable value)."""
    if not ctx.has("svg"):
        return _gate(0.0, (1.0, 1.0))
    curve_set = ctx.get("curve_set") if ctx.has("curve_set") else None
    curve_set = curve_set if isinstance(curve_set, CurveSet) else None
    data = ctx.get("svg").data  # type: ignore[attr-defined]
    try:
        validate_svg(bytes(data), curve_set)
        return _gate(1.0, (1.0, 1.0))
    except StageError:
        return _gate(0.0, (1.0, 1.0))


def _pdf_validity(ctx: PipelineContext) -> MetricResult:
    """QM-28, surfaced as a metric (see ``_svg_validity``). PDF is
    optional -- absent PDF is reported as not-applicable via a passing
    1.0, matching ``OutputBundle.pdf: bytes | None``'s own optionality."""
    if not ctx.has("pdf"):
        return _gate(1.0, (1.0, 1.0))
    data = ctx.get("pdf").data  # type: ignore[attr-defined]
    try:
        validate_pdf(bytes(data))
        return _gate(1.0, (1.0, 1.0))
    except StageError:
        return _gate(0.0, (1.0, 1.0))


def _printability_score(tiny_region_pct: float) -> float:
    """BENCHMARK_SPEC.md §10.2's Engine Score printability dimension:
    ``1 - QM-12/100`` (leader ratio complement), floored at 0.5. QM-12
    (leader-line ratio) isn't separately tracked by any validator today;
    ``tiny_region_pct`` is used as the input here since every tiny region
    is exactly the population QM-12 measures (faces demoted to a leader),
    so this is the same quantity under the label already available."""
    return max(0.5, 1.0 - tiny_region_pct / 100.0)


def compute_quality_metrics(
    ctx: PipelineContext,
    *,
    printability_metrics: dict[str, float] | None = None,
) -> dict[str, MetricResult]:
    """Compute every Sprint 23 quality metric from the pipeline's bound
    artifacts. Call after the canonical validators have run (so
    ``label_plan`` reflects any printability repair) but this function
    itself never mutates ctx and never raises on a quality shortfall --
    only on a missing required artifact, which is a caller error."""
    curve_set = ctx.get("curve_set")
    label_plan = ctx.get("label_plan")
    palette = ctx.get("palette")
    assert isinstance(curve_set, CurveSet)
    assert isinstance(label_plan, LabelPlan)
    assert isinstance(palette, Palette)

    tolerance_pt = _FLATTEN_MM * _MM_TO_PT

    metrics: dict[str, MetricResult] = {}
    metrics.update(_region_stats(curve_set, tolerance_pt))
    metrics["QM-11"] = _tiny_region_pct(printability_metrics)
    metrics["QM-08"] = _boundary_smoothness(curve_set, tolerance_pt)
    metrics["QM-16"] = _palette_quality(palette)
    metrics["QM-22"] = _label_fit_rate(curve_set, label_plan, tolerance_pt)
    metrics["label_overlap_rate_pct"] = _label_overlap_rate(label_plan)
    metrics["QM-26"] = _svg_validity(ctx)
    metrics["QM-28"] = _pdf_validity(ctx)

    tiny_pct = metrics["QM-11"].value
    printability_value = _printability_score(tiny_pct)
    metrics["printability_score"] = _monitor(printability_value, (0.5, 1.0))

    return metrics
