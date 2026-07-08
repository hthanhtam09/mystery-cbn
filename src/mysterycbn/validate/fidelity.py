"""Fidelity Validator (I1): every output region maps to a connected pixel set
of the quantized input; composition preserved (ENGINE_SPEC.md §25.1;
QM-18 Face-Label Agreement).

Correspondence audit: rasterize each face id at working resolution
(scanline fill, even-odd over its rings) and compare against the region's
own ``component_map`` (the label raster produced by §9-11, independent of
the ArcGraph/CurveSet geometry that would otherwise "grade its own
homework"). Agreement must be >= ``fidelity_min_agreement`` (default 99%)
for every face.

Note: the §24 PNG Preview module (solved-preview SSIM probe, QM-17) is not
yet implemented in this codebase, so that half of §25.1 is out of scope
here; this validator covers the face<->label correspondence audit only.
"""

from __future__ import annotations

import numpy as np

from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import RegionGraph
from mysterycbn.model.reports import Finding, Severity, ValidationReport
from mysterycbn.model.vector import ArcGraph, CurveSet
from mysterycbn.validate.common import flatten_face_rings, segments_of_ring

VALIDATOR_NAME = "fidelity"
FIDELITY_MIN_AGREEMENT_DEFAULT = 0.99
_FLATTEN_MM = 0.1
_MM_TO_PT = 72.0 / 25.4


def _rasterize_face_mask(rings_px: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Even-odd scanline fill of a face's rings (in *working px* coords) over
    a boolean (H, W) mask -- the same primitive the §24 preview would use."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    all_edges = [segments_of_ring(r) for r in rings_px]
    ys = np.arange(h) + 0.5
    for row_idx, y in enumerate(ys):
        xs_hit: list[float] = []
        for seg_a, seg_b in all_edges:
            cond = (seg_a[:, 1] > y) != (seg_b[:, 1] > y)
            if not cond.any():
                continue
            a, b = seg_a[cond], seg_b[cond]
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
            xs_hit.extend(xi.tolist())
        xs_hit.sort()
        for i in range(0, len(xs_hit) - 1, 2):
            x0 = max(0, int(np.ceil(xs_hit[i] - 0.5)))
            x1 = min(w, int(np.floor(xs_hit[i + 1] - 0.5)) + 1)
            if x1 > x0:
                mask[row_idx, x0:x1] = True
    return mask


def validate_fidelity(
    ctx: PipelineContext,
    *,
    fidelity_min_agreement: float = FIDELITY_MIN_AGREEMENT_DEFAULT,
) -> ValidationReport:
    """Run the QM-18 correspondence audit against the bound ``region_graph``
    (authoritative label raster) and ``curve_set`` (final vector geometry)."""
    curve_set = ctx.get("curve_set")
    region_graph = ctx.get("region_graph")
    assert isinstance(curve_set, CurveSet)
    assert isinstance(region_graph, RegionGraph)

    work_scale = _infer_work_scale(ctx)
    tolerance_pt = _FLATTEN_MM * _MM_TO_PT
    component_map = region_graph.component_map
    h, w = component_map.shape

    # Φ's letterbox origin (m_x, m_y) is not itself stored on the ArcGraph;
    # it is recoverable exactly as the min corner of the painted geometry
    # (every face ring's min x/y), since Φ maps working px (0, 0) there.
    all_rings = [
        ring
        for face in curve_set.faces
        for ring in flatten_face_rings(face, curve_set, tolerance_pt)
    ]
    origin_x = min(float(ring[:, 0].min()) for ring in all_rings)
    origin_y = min(float(ring[:, 1].min()) for ring in all_rings)

    # component_map stores dense region_ids, not palette labels; face.label
    # is a palette label (DATA_MODEL_SPEC §7/§11) -- compare through the
    # region record, not the raw component_map value.
    label_of_region = [region.label for region in region_graph.regions]

    findings: list[Finding] = []
    agreements: list[float] = []
    for face in curve_set.faces:
        rings_pt = flatten_face_rings(face, curve_set, tolerance_pt)
        rings_px = [
            (ring - np.array([origin_x, origin_y])) / work_scale for ring in rings_pt
        ]  # pt -> working px, still (x, y)
        mask = _rasterize_face_mask(rings_px, (h, w))
        covered = component_map[mask] if mask.any() else np.empty(0, dtype=component_map.dtype)
        if covered.size == 0:
            agreements.append(1.0)
            continue
        covered_labels = np.take(np.asarray(label_of_region), covered)
        matches = int(np.count_nonzero(covered_labels == face.label))
        agreement = matches / covered.size
        agreements.append(agreement)
        if agreement < fidelity_min_agreement:
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I1",
                    message=(
                        f"face-label agreement {agreement:.4f} below floor {fidelity_min_agreement}"
                    ),
                    location=f"face {face.face_id}",
                )
            )

    metrics = {"min_face_label_agreement": min(agreements) if agreements else 1.0}
    return ValidationReport(
        validator_name=VALIDATOR_NAME, findings=tuple(findings), metrics=metrics
    )


def _infer_work_scale(ctx: PipelineContext) -> float:
    """The Φ scale (pt per working px) recorded on the ArcGraph, if bound."""
    if ctx.has("arc_graph"):
        arc_graph = ctx.get("arc_graph")
        assert isinstance(arc_graph, ArcGraph)
        return arc_graph.work_scale
    raster = ctx.get("raster_image") if ctx.has("raster_image") else None
    raster_scale = getattr(raster, "work_scale", 0.0)
    if raster is not None and raster_scale > 0.0:
        return float(raster_scale)
    return 1.0
