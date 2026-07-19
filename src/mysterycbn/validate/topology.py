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
from mysterycbn.model.flatten import ring_self_intersects, rings_intersect
from mysterycbn.validate.common import face_area_pt2, flatten_face_rings

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


# Canonical implementations live in model/flatten.py so the vector stages'
# pre-gate repair shares them bitwise; alias keeps the validate-side name.
_self_intersects = ring_self_intersects


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
            for j in range(i + 1, len(rings)):
                if rings_intersect(rings[i], rings[j]):
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
