"""Topology Validator (I3): independent re-proof of the planar-partition
invariant (ENGINE_SPEC.md §25.2; QM-01 Topology Errors, QM-02 Watertightness).

Re-derives, by a method different from the ArcGraph construction (§15) and
the CurveSet construction (§17-18): every arc borders exactly 2 faces; the
sum of flattened face areas equals the content-box area within tolerance;
no arc self-intersects and no two arcs intersect except at shared junction
endpoints. Never repaired — a topology repair would be a lie (§25.2).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import RegionGraph
from mysterycbn.model.reports import Finding, Severity, ValidationReport
from mysterycbn.model.vector import ArcGraph, CurveSet
from mysterycbn.validate.common import face_area_pt2, flatten_face_rings, segments_of_ring

VALIDATOR_NAME = "topology"
_FLATTEN_MM = 0.1
_MM_TO_PT = 72.0 / 25.4
# QUALITY_SPEC QM-02 states 1e-4 for the ArcGraph's own exact/near-exact
# polyline area identity. This validator instead re-proves watertightness
# on the *final*, Bézier-smoothed CurveSet (the artifact that actually
# ships) -- smoothing perturbs the outer frame boundary within the
# QM-09 displacement bound (0.20 mm), which necessarily shows up here as a
# small residual even on a perfectly-constructed ArcGraph. The floor below
# is sized to that budget, not to the pre-smoothing identity.
_WATERTIGHT_MAX_REL = 2e-3


def _arc_side_counts(curve_set: CurveSet) -> Counter[int]:
    counts: Counter[int] = Counter()
    for face in curve_set.faces:
        for walk in face.all_walks():
            for arc_id, _ in walk:
                counts[arc_id] += 1
    return counts


def _segments_intersect(
    p: np.ndarray, q: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray
) -> np.ndarray:
    """Proper-intersection mask (open interior) of segment p->q vs each (a, b).

    ``p``/``q`` may be a single 2-vector (broadcast against many segments)
    or an (N, 2) array matched element-wise against ``seg_a``/``seg_b``.
    """
    d = q - p
    e = seg_b - seg_a
    w = seg_a - p
    d0, d1 = (d[..., 0], d[..., 1])
    e0, e1 = (e[..., 0], e[..., 1])
    w0, w1 = (w[..., 0], w[..., 1])
    denom = d0 * e1 - d1 * e0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w0 * e1 - w1 * e0) / denom
        s = (w0 * d1 - w1 * d0) / denom
    return np.asarray((denom != 0) & (t > 1e-9) & (t < 1 - 1e-9) & (s > 1e-9) & (s < 1 - 1e-9))


def _bboxes(seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """(N, 4) axis-aligned bounding boxes (xmin, ymin, xmax, ymax) per edge."""
    lo = np.minimum(seg_a, seg_b)
    hi = np.maximum(seg_a, seg_b)
    return np.concatenate([lo, hi], axis=1)


def _bbox_overlap(boxes_i: np.ndarray, boxes_j: np.ndarray) -> np.ndarray:
    """(len(boxes_i), len(boxes_j)) overlap mask -- the spatial-hash pre-filter
    (ENGINE_SPEC §25.2's "segment sweep with spatial hash") that lets the
    exact intersection test skip the overwhelming majority of far-apart pairs
    on faces with hundreds of boundary segments."""
    return (
        (boxes_i[:, None, 0] <= boxes_j[None, :, 2])
        & (boxes_j[None, :, 0] <= boxes_i[:, None, 2])
        & (boxes_i[:, None, 1] <= boxes_j[None, :, 3])
        & (boxes_j[None, :, 1] <= boxes_i[:, None, 3])
    )


def _self_intersects(ring: np.ndarray) -> bool:
    """Any proper self-intersection among a ring's own edges (non-adjacent),
    bounding-box filtered before the exact orientation test."""
    seg_a, seg_b = segments_of_ring(ring)
    n = len(seg_a)
    if n < 4:
        return False
    boxes = _bboxes(seg_a, seg_b)
    overlap = np.triu(_bbox_overlap(boxes, boxes), k=2)
    overlap[0, n - 1] = False  # the wrap-around adjacent pair (n-1, 0)
    pairs_i, pairs_j = np.nonzero(overlap)
    if pairs_i.size == 0:
        return False
    hits = _segments_intersect(seg_a[pairs_i], seg_b[pairs_i], seg_a[pairs_j], seg_b[pairs_j])
    return bool(hits.any())


def validate_topology(ctx: PipelineContext) -> ValidationReport:
    """Run the QM-01/QM-02 re-proof against the bound ``curve_set``.

    The watertightness ground truth is ``work_scale² · H · W`` (the exact
    painted area of the aspect-preserving Φ letterbox, MATH_SPEC §1.3) --
    *not* the raw content-box rectangle, which is generally larger than the
    painted region whenever the source image's aspect ratio differs from
    the page's.
    """
    curve_set = ctx.get("curve_set")
    arc_graph = ctx.get("arc_graph")
    region_graph = ctx.get("region_graph")
    assert isinstance(curve_set, CurveSet)
    assert isinstance(arc_graph, ArcGraph)
    assert isinstance(region_graph, RegionGraph)
    findings: list[Finding] = []
    tolerance_pt = _FLATTEN_MM * _MM_TO_PT

    # Pair-constancy: every arc borders exactly 2 face-walk sides, except
    # arcs on the page exterior (the exterior face is never stored, so a
    # boundary arc legitimately appears once; DATA_MODEL_SPEC's
    # ``_check_arc_references`` treats 1 or 2 as valid for the same reason).
    counts = _arc_side_counts(curve_set)
    n_badarc = 0
    for arc_id in range(len(curve_set.curves)):
        n = counts.get(arc_id, 0)
        if n not in (1, 2):
            n_badarc += 1
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I3",
                    message=f"arc borders {n} face-walk sides (expected 1 or 2)",
                    location=f"arc {arc_id}",
                )
            )

    # Watertightness: sum of independently-flattened face areas vs the exact
    # painted area (work_scale² · H · W — the letterboxed content, not the
    # full content-box rectangle).
    area_sum = sum(face_area_pt2(face, curve_set, tolerance_pt) for face in curve_set.faces)
    h, w = region_graph.component_map.shape
    painted_area = (arc_graph.work_scale**2) * h * w
    w_res = abs(area_sum - painted_area) / painted_area if painted_area > 0 else float("inf")
    if w_res > _WATERTIGHT_MAX_REL:
        findings.append(
            Finding(
                severity=Severity.FATAL,
                invariant="I3",
                message=f"watertightness residual {w_res:.2e} exceeds {_WATERTIGHT_MAX_REL:.0e}",
                location="page",
            )
        )

    # Self-intersection + cross-arc intersection sweep (segment-level, per face).
    n_selfx = 0
    n_pairx = 0
    for face in curve_set.faces:
        rings = flatten_face_rings(face, curve_set, tolerance_pt)
        for ring in rings:
            if _self_intersects(ring):
                n_selfx += 1
                findings.append(
                    Finding(
                        severity=Severity.FATAL,
                        invariant="I3",
                        message="arc self-intersection detected",
                        location=f"face {face.face_id}",
                    )
                )
        for i in range(len(rings)):
            seg_a_i, seg_b_i = segments_of_ring(rings[i])
            for j in range(i + 1, len(rings)):
                seg_a_j, seg_b_j = segments_of_ring(rings[j])
                hit_any = False
                for a, b in zip(seg_a_i, seg_b_i, strict=True):
                    if bool(_segments_intersect(a, b, seg_a_j, seg_b_j).any()):
                        hit_any = True
                        break
                if hit_any:
                    n_pairx += 1
                    findings.append(
                        Finding(
                            severity=Severity.FATAL,
                            invariant="I3",
                            message="two rings intersect away from shared junctions",
                            location=f"face {face.face_id} rings {i},{j}",
                        )
                    )

    t_err = n_badarc + n_selfx + n_pairx
    metrics = {
        "topology_errors": float(t_err),
        "watertightness_residual": w_res,
    }
    return ValidationReport(
        validator_name=VALIDATOR_NAME, findings=tuple(findings), metrics=metrics
    )
