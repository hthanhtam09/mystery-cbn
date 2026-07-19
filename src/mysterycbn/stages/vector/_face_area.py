"""Shared helper: per-arc "smallest adjacent face area" for tolerance scaling.

``simplify`` and ``bezier`` each apply one fixed error tolerance
(``tolerance_mm`` / ``fit_error_mm``) to every arc, independent of how
small the face(s) that arc bounds actually are. For a face well above the
printability area floor, a sub-tolerance boundary deviation is a negligible
fraction of its pixel count; for a face ``merge_tiny`` legitimately left
just above that floor (``ENGINE_SPEC.md`` §11: ``A_min`` is the exact
contract, not a soft target), the same absolute deviation can flip enough
boundary pixels to fail the ``fidelity`` validator's 99% face/label
agreement (I1) purely from ordinary curve-fit/simplification residual, not
from any construction defect.

This module computes, for every arc in an ``ArcGraph``, the area (pt²) of
the smaller of its one or two adjacent faces -- the two curve-generation
stages use it to shrink their own tolerance on arcs that border small
faces, so approximation error scales down exactly where it can no longer
be absorbed by a face's own size.
"""

from __future__ import annotations

import math

import numpy as np

from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.vector import Arc, Face

_EPS = 1e-9


def area_floor_pt2(d_min_mm: float) -> float:
    """Printability area floor ``A_min = π (d_min/2)²`` in pt² (arcs are
    already post-Φ, so unlike ``stages/graph/merge.py::area_floor_px`` no
    ``work_scale`` conversion is needed -- ``d_min_mm`` converts to pt
    directly)."""
    d_min_pt = d_min_mm * PT_PER_INCH / MM_PER_INCH
    return math.pi * (d_min_pt / 2.0) ** 2


def _ring_area_2x(ring: np.ndarray) -> float:
    """2 × signed area of a closed polyline ring (implicit closing edge).

    Same convention as ``validate/common.py::ring_area_2x`` (outer rings
    positive, y-down page frame) -- duplicated rather than imported to keep
    ``stages/`` and ``validate/`` independent per the codebase's
    "independent re-proof" convention (``validate`` must never import
    construction-time helpers a stage also uses, or a shared bug would be
    invisible to both).
    """
    x, y = ring[:, 0], ring[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    return -float(np.sum(x * yn - xn * y))


def _walk_ring(walk: tuple[tuple[int, bool], ...], arcs: tuple[Arc, ...]) -> np.ndarray:
    parts = []
    for arc_id, rev in walk:
        pts = arcs[arc_id].points
        parts.append(pts[::-1] if rev else pts)
    return np.concatenate(parts)


def face_area_pt2(face: Face, arcs: tuple[Arc, ...]) -> float:
    """Area of one face (outer ring minus holes), in pt², from raw
    (post-Φ, pre-simplification/fit) ``Arc.points`` polylines."""
    rings = [_walk_ring(walk, arcs) for walk in face.all_walks()]
    return sum(_ring_area_2x(r) for r in rings) / 2.0


def min_adjacent_face_area_pt2_by_arc(
    arcs: tuple[Arc, ...], faces: tuple[Face, ...]
) -> dict[int, float]:
    """For every arc, the area (pt²) of the smaller face among the one or
    two faces whose walk references it. An arc referenced by only one face
    (the other side is the page exterior, never stored as a ``Face``) uses
    that single face's area."""
    face_areas = [face_area_pt2(face, arcs) for face in faces]
    best: dict[int, float] = {}
    for face, area in zip(faces, face_areas, strict=True):
        for walk in face.all_walks():
            for arc_id, _ in walk:
                current = best.get(arc_id)
                best[arc_id] = area if current is None else min(current, area)
    return best


_HEADROOM_FACTOR = 9.0  # area ratio -> 3x the floor diameter; see docstring below


def tolerance_scale_for_area(area_pt2: float, *, reference_area_pt2: float) -> float:
    """Linear-dimension scale factor in ``(0, 1]`` for an arc whose smallest
    adjacent face has area ``area_pt2``, relative to ``reference_area_pt2``
    (the printability area floor ``A_min``, in the same units).

    Gating the scale-down at exactly ``A_min`` (scale 1.0 right at/above the
    floor) does not fix the observed failure: ``merge_tiny`` only guarantees
    a face is *at or above* ``A_min``, not meaningfully above it -- a face
    at 105% of the floor is just as small in absolute terms as one at 95%,
    but a hard cutoff at 100% would leave it at full (unshrunk) tolerance.
    Instead this scales smoothly against ``_HEADROOM_FACTOR * A_min`` (area
    ratio 9 == diameter ratio 3, i.e. full tolerance is only reached once a
    face's diameter is at least triple the printability floor's), so faces
    anywhere near the floor -- above or below it -- get a proportionally
    tightened fit, while faces comfortably larger are unaffected.
    """
    reference = reference_area_pt2 * _HEADROOM_FACTOR
    if reference <= _EPS or area_pt2 >= reference:
        return 1.0
    if area_pt2 <= _EPS:
        return _EPS
    return float(np.sqrt(area_pt2 / reference))


def same_label_seam_arc_ids(
    faces: tuple[Face, ...], filler_ids: frozenset[int]
) -> frozenset[int]:
    """Arc ids whose every touching face is a filler cell AND carries the
    same palette label — the decorative seams between two cells of one
    subdivided (organic/split) same-color pattern.

    Boundary displacement across such a seam cannot change any pixel's
    label agreement (both sides own the same label), so these arcs are safe
    to simplify/fit at a *looser* tolerance for flowing, rounded curves —
    unlike filler arcs on a color boundary (a rim tracing the subject's
    silhouette), which must stay tight for fidelity."""
    sides: dict[int, list[tuple[int, int]]] = {}
    for face in faces:
        for walk in face.all_walks():
            for arc_id, _rev in walk:
                sides.setdefault(arc_id, []).append((face.face_id, face.label))
    seams: set[int] = set()
    for arc_id, touching in sides.items():
        if all(fid in filler_ids for fid, _ in touching) and (
            len({label for _, label in touching}) == 1
        ):
            seams.add(arc_id)
    return frozenset(seams)


def points_self_intersect(pts: np.ndarray) -> bool:
    """Whether a flattened ring/chain properly crosses itself — the vector
    stages' pre-gate mirror of the topology validator's I3 predicate. Tests
    every non-adjacent edge pair for a proper (interior) crossing. O(n²) in
    edge count; callers only use it for rings containing loose-fit seam
    arcs, which are short."""
    a, b = pts[:-1], pts[1:]
    n = len(a)
    d = b - a
    closed = bool(np.allclose(pts[0], pts[-1]))
    for i in range(n - 2):
        j0 = i + 2
        j1 = n - 1 if i == 0 and closed else n
        if j0 >= j1:
            continue
        cross_d = d[i, 0] * d[j0:j1, 1] - d[i, 1] * d[j0:j1, 0]
        rel = a[j0:j1] - a[i]
        t_num = rel[:, 0] * d[j0:j1, 1] - rel[:, 1] * d[j0:j1, 0]
        s_num = rel[:, 0] * d[i, 1] - rel[:, 1] * d[i, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = t_num / cross_d
            ss = s_num / cross_d
        hit = (cross_d != 0) & (tt > 1e-9) & (tt < 1 - 1e-9) & (ss > 1e-9) & (ss < 1 - 1e-9)
        if bool(hit.any()):
            return True
    return False
