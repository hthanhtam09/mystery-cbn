"""Canonical face-ring flattening + self-intersection predicate.

Single source of truth shared by the topology validator (validate/topology)
and the bezier stage's pre-gate repair (stages/vector/curves): the repair
must see EXACTLY the geometry the validator will re-derive, or a borderline
crossing slips through the pre-gate and FATALs at the gate (observed twice
with near-identical-but-not-bitwise reimplementations). Lives in ``model``
because both consumers sit on sibling layers that may not import each other
(pyproject.toml import-linter "v2 layer graph").
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mysterycbn.model.vector import Curve, Face


def flatten_bezier(control: np.ndarray, tolerance_pt: float) -> np.ndarray:
    """Sample one cubic segment at chord-proportional density; last point dropped."""
    chord = float(
        np.linalg.norm(control[3] - control[0])
        + np.linalg.norm(control[1] - control[0])
        + np.linalg.norm(control[2] - control[1])
        + np.linalg.norm(control[3] - control[2])
    )
    n = int(np.clip(math.ceil(chord / (4.0 * tolerance_pt)), 2, 24))
    u = np.linspace(0.0, 1.0, n + 1)
    b = np.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u**2 * (1 - u), u**3], axis=1)
    return np.asarray(b @ control)[:-1]


def flatten_face_rings(
    face: Face, curves: Sequence[Curve], tolerance_pt: float
) -> list[np.ndarray]:
    """Every ring (outer + holes) of ``face`` flattened to a closed polyline.

    ``curves`` must be indexable by ``arc_id`` (``curves[arc_id]``), exactly
    like ``CurveSet.curves``."""
    rings = []
    for walk in face.all_walks():
        parts = []
        for arc_id, rev in walk:
            segments = curves[arc_id].segments
            for segment in reversed(segments) if rev else segments:
                pts = flatten_bezier(segment.control, tolerance_pt)
                parts.append(pts[::-1] if rev else pts)
        rings.append(np.concatenate(parts))
    return rings


def segments_of_ring(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) arrays of the ring's edges, including the wrap-around edge."""
    return ring, np.roll(ring, -1, axis=0)


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
    # Near-parallel guard: |denom| = |d|·|e|·sin(angle). For numerically
    # collinear pairs — a ring legitimately doubling back along its own edge
    # (zero-width slit), whose exact-collinear form the ``denom != 0`` branch
    # already excludes — Bézier flattening's fp wobble (~1e-13) makes denom
    # nonzero and the t/s division ill-conditioned, reporting a phantom
    # proper crossing. Scale the exclusion by the edge lengths so the same
    # pair is excluded whether sampled exactly or with fp dirt; genuine
    # crossings have sin(angle) orders of magnitude above 1e-9.
    scale = np.sqrt((d0 * d0 + d1 * d1) * (e0 * e0 + e1 * e1))
    non_parallel = np.abs(denom) > 1e-9 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w0 * e1 - w1 * e0) / denom
        s = (w0 * d1 - w1 * d0) / denom
    return np.asarray(non_parallel & (t > 1e-9) & (t < 1 - 1e-9) & (s > 1e-9) & (s < 1 - 1e-9))


def _bboxes(seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    lo = np.minimum(seg_a, seg_b)
    hi = np.maximum(seg_a, seg_b)
    return np.concatenate([lo, hi], axis=1)


def _bbox_overlap(boxes_i: np.ndarray, boxes_j: np.ndarray) -> np.ndarray:
    return (
        (boxes_i[:, None, 0] <= boxes_j[None, :, 2])
        & (boxes_j[None, :, 0] <= boxes_i[:, None, 2])
        & (boxes_i[:, None, 1] <= boxes_j[None, :, 3])
        & (boxes_j[None, :, 1] <= boxes_i[:, None, 3])
    )


def rings_intersect(ring_a: np.ndarray, ring_b: np.ndarray) -> bool:
    """Any proper intersection between two rings' edges (open interiors —
    touching at shared junction endpoints does not count), bounding-box
    filtered before the exact orientation test. Canonical predicate for the
    topology validator's ring-pair check and the vector stages' pre-gate
    repair, so both agree bitwise."""
    seg_a_i, seg_b_i = segments_of_ring(ring_a)
    seg_a_j, seg_b_j = segments_of_ring(ring_b)
    overlap = _bbox_overlap(_bboxes(seg_a_i, seg_b_i), _bboxes(seg_a_j, seg_b_j))
    pairs_i, pairs_j = np.nonzero(overlap)
    if pairs_i.size == 0:
        return False
    hits = _segments_intersect(
        seg_a_i[pairs_i], seg_b_i[pairs_i], seg_a_j[pairs_j], seg_b_j[pairs_j]
    )
    return bool(hits.any())


def ring_self_intersects(ring: np.ndarray) -> bool:
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
