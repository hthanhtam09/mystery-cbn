"""Shared geometry helpers for the validation subsystem (ENGINE_SPEC.md §25).

Every validator re-derives its invariant from raw artifacts by a method
*different* from the constructing stage (independent double-entry
bookkeeping, ARCHITECTURE.md §0) — these helpers exist only to avoid
duplicating numerically-identical code, not to share the construction path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mysterycbn.model.vector import CurveSet, Face

_FLATTEN_MM_DEFAULT = 0.1


def _flatten_bezier(control: np.ndarray, tolerance_pt: float) -> np.ndarray:
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


def flatten_face_rings(face: Face, curve_set: CurveSet, tolerance_pt: float) -> list[np.ndarray]:
    """Every ring (outer + holes) of ``face`` flattened to a closed polyline."""
    rings = []
    for walk in face.all_walks():
        parts = []
        for arc_id, rev in walk:
            segments = curve_set.curves[arc_id].segments
            for segment in reversed(segments) if rev else segments:
                pts = _flatten_bezier(segment.control, tolerance_pt)
                parts.append(pts[::-1] if rev else pts)
        rings.append(np.concatenate(parts))
    return rings


def flatten_arc_polyline(control_chain: Sequence[np.ndarray], tolerance_pt: float) -> np.ndarray:
    """Flatten a full arc (all its Bézier segments in order) to one polyline,
    keeping the final endpoint (an arc is not implicitly closed)."""
    parts = [_flatten_bezier(ctrl, tolerance_pt) for ctrl in control_chain]
    parts.append(control_chain[-1][3][None, :])
    return np.concatenate(parts)


def ring_area_2x(ring: np.ndarray) -> float:
    """2 × signed area of a closed polyline ring (implicit closing edge).

    ``ring`` columns are ``(x, y)`` pt, matching post-Φ ``Arc.points`` /
    ``BezierSegment.control`` (e.g. the SVG renderer treats column 0 as x).
    Outer rings are positive: negated shoelace, since the page frame is
    y-down (a clockwise-in-math ring is CCW-in-screen-space / outer).
    """
    x, y = ring[:, 0], ring[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    return -float(np.sum(x * yn - xn * y))


def face_area_pt2(face: Face, curve_set: CurveSet, tolerance_pt: float) -> float:
    """Signed area of a face (outer ring minus holes) in pt², independently
    re-derived from flattened Bézier geometry (not the construction-time
    integer shoelace of the ArcGraph)."""
    rings = flatten_face_rings(face, curve_set, tolerance_pt)
    return sum(ring_area_2x(r) for r in rings) / 2.0


def segments_of_ring(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(seg_a, seg_b) endpoint arrays for a closed ring's edges."""
    return ring, np.roll(ring, -1, axis=0)


def point_in_rings(point: np.ndarray, rings: list[np.ndarray]) -> bool:
    """Even-odd containment test of ``point`` against a face's rings."""
    inside = False
    x, y = point[0], point[1]
    for ring in rings:
        a, b = segments_of_ring(ring)
        cond = (a[:, 1] > y) != (b[:, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
        crossings = int(np.count_nonzero(cond & (x < xi)))
        if crossings % 2:
            inside = not inside
    return inside
