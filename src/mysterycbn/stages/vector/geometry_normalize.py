"""Geometry Normalize stage: config validation (Sprint 36A.3), stage wiring
(Sprint 36A.4), Duplicate Point Cleanup (Sprint 36B.1), Spike Removal
(Sprint 36B.2), and Minimum Gap Enforcement / Gap Repair (Sprint 36B.3).

See ``docs/modules/geometry_normalize.md`` and ``docs/modules/
GAP_REPAIR_DESIGN.md`` for the frozen design that Pass 3 implements exactly.

Per the frozen ``geometry_normalize`` architecture (accepted architecture
review), every pass threshold is bounded by a shared ceiling: it must be
positive, and it must not exceed ``simplify.tolerance_mm`` -- so that no
pass's correction can exceed the geometric error budget ``simplify`` already
spends, keeping the combined worst case boundable in terms the engine
already reasons about for ``simplify`` alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.flatten import ring_self_intersects, rings_intersect
from mysterycbn.model.records import Provenance
from mysterycbn.model.vector import Arc, ArcGraph

STAGE_NAME = "geometry_normalize"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64
_MM_TO_PT = 72.0 / 25.4

ENABLED_DEFAULT = True
DUPLICATE_EPS_MM_DEFAULT = 0.01
SPIKE_LENGTH_MM_DEFAULT = 0.05
MIN_GAP_MM_DEFAULT = 0.1

_THRESHOLD_KEYS: tuple[str, ...] = ("duplicate_eps_mm", "spike_length_mm", "min_gap_mm")
_THRESHOLD_DEFAULTS: Mapping[str, float] = {
    "duplicate_eps_mm": DUPLICATE_EPS_MM_DEFAULT,
    "spike_length_mm": SPIKE_LENGTH_MM_DEFAULT,
    "min_gap_mm": MIN_GAP_MM_DEFAULT,
}


class GeometryNormalizeConfig:
    """Validated ``geometry_normalize`` config: ``enabled`` + three mm thresholds.

    Each threshold must be > 0 and <= ``simplify.tolerance_mm`` (the shared
    ceiling every pass's error budget is bound by); violations raise
    ``ConfigError`` at construction, before any geometry is touched.
    """

    def __init__(
        self,
        section: Mapping[str, object],
        *,
        simplify_tolerance_mm: float,
    ) -> None:
        enabled = section.get("enabled", ENABLED_DEFAULT)
        if not isinstance(enabled, bool):
            raise ConfigError(f"geometry_normalize config: enabled must be a bool, got {enabled!r}")

        thresholds: dict[str, float] = {}
        for key in _THRESHOLD_KEYS:
            value = section.get(key, _THRESHOLD_DEFAULTS[key])
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(
                    f"geometry_normalize config: {key} must be a number, got {value!r}"
                )
            value = float(value)
            if value <= 0.0:
                raise ConfigError(f"geometry_normalize config: {key} must be > 0, got {value}")
            if value > simplify_tolerance_mm:
                raise ConfigError(
                    f"geometry_normalize config: {key} must be <= simplify.tolerance_mm "
                    f"({simplify_tolerance_mm}), got {value}"
                )
            thresholds[key] = value

        self.enabled = enabled
        self.duplicate_eps_mm = thresholds["duplicate_eps_mm"]
        self.spike_length_mm = thresholds["spike_length_mm"]
        self.min_gap_mm = thresholds["min_gap_mm"]
        # Stored (not just validated against) so Pass 3 can derive its own
        # displacement ceiling (GAP_REPAIR_DESIGN.md §4.1:
        # max_displacement_pt = min(Delta/2, simplify.tolerance_mm)).
        self.simplify_tolerance_mm = simplify_tolerance_mm


# --------------------------------------------------- normalization passes ---
#
# Sprint 36B.1 implements Pass 1 (duplicate point cleanup); Pass 2 (spike
# removal) and Pass 3 (minimum gap enforcement, see GAP_REPAIR_DESIGN.md)
# remain identity placeholders pending their own frozen per-pass designs
# (docs/modules/geometry_normalize.md §16). Each pass takes the config and
# returns a per-pass count so the metrics surface is exercised uniformly.


def _clean_duplicate_points(points: np.ndarray, *, eps_pt: float, closed: bool) -> np.ndarray:
    """Drop interior points whose distance from the last *kept* point is
    < ``eps_pt``. Endpoints (index 0 and the last index) are never removed
    -- they are junctions (open arcs) or the closed-arc anchor, both
    immovable per the frozen ``geometry_normalize`` design (module doc §5:
    junction endpoints are categorically excluded from every pass).

    For a closed arc, the wraparound edge (last kept point -> point 0) must
    also stay non-degenerate; if it would end up under ``eps_pt``, the last
    interior kept point is dropped instead (never point 0, the anchor).

    Never drops below the ``Arc`` minimum point-count floor (4 for closed,
    2 for open): if unconstrained dedup would go below the floor, the
    highest-index dropped points are restored (in original order) until
    the floor is met -- preserving the invariant takes priority over
    reaching the ideal spacing.
    """
    minimum = 4 if closed else 2
    n = points.shape[0]
    if n <= minimum:
        return points

    kept = [0]
    for i in range(1, n - 1):
        if np.linalg.norm(points[i] - points[kept[-1]]) >= eps_pt:
            kept.append(i)
    kept.append(n - 1)

    if closed and len(kept) > minimum:
        if np.linalg.norm(points[kept[0]] - points[kept[-1]]) < eps_pt:
            kept.pop(-2)  # drop the last interior point, never the anchor (index 0)

    if len(kept) < minimum:
        dropped = sorted(set(range(n)) - set(kept))
        # Restore the lowest-index dropped points first (closest to
        # existing structure) until the floor is met; order restored,
        # then re-sorted with the rest of ``kept``.
        kept = sorted(set(kept) | set(dropped[: minimum - len(kept)]))

    if len(kept) == n:
        return points
    return points[np.asarray(kept, dtype=np.int64)]


def _duplicate_cleanup(
    arcs: tuple[Arc, ...], *, config: GeometryNormalizeConfig
) -> tuple[tuple[Arc, ...], int]:
    """Pass 1: remove consecutive near-duplicate points per arc (§8.1 of the
    module doc). ``arc_id``/``left_region``/``right_region``/``closed`` are
    always preserved; an unmodified arc is returned as the identical
    object, a modified arc as a new ``Arc`` with only ``points`` replaced.
    """
    eps_pt = config.duplicate_eps_mm * _MM_TO_PT
    removed = 0
    out: list[Arc] = []
    for arc in arcs:
        cleaned = _clean_duplicate_points(arc.points, eps_pt=eps_pt, closed=arc.closed)
        if cleaned.shape[0] == arc.points.shape[0]:
            out.append(arc)
            continue
        removed += arc.points.shape[0] - cleaned.shape[0]
        out.append(
            Arc(
                arc_id=arc.arc_id,
                points=cleaned,
                left_region=arc.left_region,
                right_region=arc.right_region,
                closed=arc.closed,
            )
        )
    return tuple(out), removed


_SPIKE_TURN_ANGLE_DEG = 150.0  # near-total direction reversal (MATH_SPEC §7's turn-angle idiom)


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """MATH_SPEC §7.2: sign of (b-a) x (c-a)."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_properly_intersect(
    p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray
) -> bool:
    """MATH_SPEC §7.3: proper intersection via opposite-sign orient tests."""
    o1 = _orient(p, q, a)
    o2 = _orient(p, q, b)
    o3 = _orient(a, b, p)
    o4 = _orient(a, b, q)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _turn_angles_deg(points: np.ndarray) -> np.ndarray:
    """Interior turn angle (degrees) at every index 1..P-2; MATH_SPEC-style
    dot-product formula, matching ``curves.py::_corner_split``'s convention
    (reused here for consistency, not imported -- that helper is
    ``bezier``-internal and this stage must not depend on ``bezier``)."""
    if points.shape[0] < 3:
        return np.zeros(0)
    v_in = points[1:-1] - points[:-2]
    v_out = points[2:] - points[1:-1]
    dot = (v_in * v_out).sum(axis=1)
    norms = np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1)
    cos = np.divide(dot, norms, out=np.ones_like(dot), where=norms != 0)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _removal_would_self_intersect(points: np.ndarray, remove_index: int) -> bool:
    """Would collapsing ``points[remove_index]`` (replacing the two edges
    around it with one direct edge) cross any *other*, non-adjacent
    segment of the same arc's current polyline? Local, O(P) check -- the
    only two edges that change are (i-1, i) and (i, i+1), collapsed into
    (i-1, i+1); every other existing edge of the arc is unchanged, so only
    the new edge needs testing against them (MATH_SPEC §7.3's predicate,
    same "sidedness, not distance" guard idiom used elsewhere in this
    stage and by ``simplify``'s VW guard, MATH_SPEC §8.2).
    """
    n = points.shape[0]
    new_a, new_b = points[remove_index - 1], points[remove_index + 1]
    for j in range(n - 1):
        # Skip the two collapsing edges themselves and edges sharing an
        # endpoint with the new edge (adjacency, not a crossing).
        if j in (remove_index - 1, remove_index):
            continue
        if j + 1 == remove_index - 1 or j == remove_index + 1:
            continue
        if _segments_properly_intersect(new_a, new_b, points[j], points[j + 1]):
            return True
    return False


def _is_spike_vertex(pts: np.ndarray, i: int, *, spike_length_pt: float) -> bool:
    """Does interior vertex ``i`` qualify as a spike: near-total direction
    reversal, both adjacent edges short."""
    turn = _turn_angles_deg(pts)
    if turn.shape[0] == 0:
        return False
    angle = turn[i - 1]
    edge_in = float(np.linalg.norm(pts[i] - pts[i - 1]))
    edge_out = float(np.linalg.norm(pts[i + 1] - pts[i]))
    return (
        angle > _SPIKE_TURN_ANGLE_DEG
        and edge_in < spike_length_pt
        and edge_out < spike_length_pt
    )


def _collapse_spike(pts: np.ndarray, i: int, *, minimum: int) -> np.ndarray | None:
    """Collapse the spike at index ``i``, or return ``None`` if doing so is
    not legal right now (would drop below the floor, would touch an
    endpoint, or would self-intersect) -- the caller treats ``None`` as
    "skip this vertex," never as a partial or retried removal.

    An "out-and-back" spike (vertex ``i-1`` coincides exactly with vertex
    ``i+1``) would leave a duplicate consecutive point after a plain
    collapse, violating ``Arc``'s distinctness invariant; in that case the
    redundant neighbor ``i+1`` is dropped in the same atomic step, unless
    ``i+1`` is itself an endpoint (never removed, even indirectly).
    """
    if np.array_equal(pts[i - 1], pts[i + 1]):
        if i + 1 == pts.shape[0] - 1 or pts.shape[0] - 2 < minimum:
            return None
        if _removal_would_self_intersect(pts, i):
            return None
        return np.concatenate([pts[:i], pts[i + 2 :]], axis=0)

    if _removal_would_self_intersect(pts, i):
        return None
    return np.concatenate([pts[:i], pts[i + 1 :]], axis=0)


def _remove_spikes_once(points: np.ndarray, *, spike_length_pt: float, minimum: int) -> np.ndarray:
    """One forward scan: collapse every qualifying spike found, left to
    right, backtracking one step after each collapse (removing vertex
    ``i`` changes the edge into ``i-1`` and the edge out of the vertex now
    at ``i``, formerly ``i+1``, so both are re-examined before advancing).
    A vertex whose collapse is blocked (floor, endpoint, self-intersection)
    is left in place *for this scan*; a later collapse elsewhere in the
    same polyline can change the geometry enough to unblock it, which is
    why the caller repeats this scan to a fixed point rather than treating
    one forward pass as final.
    """
    pts = points
    i = 1
    while pts.shape[0] > minimum and i < pts.shape[0] - 1:
        if not _is_spike_vertex(pts, i, spike_length_pt=spike_length_pt):
            i += 1
            continue
        collapsed = _collapse_spike(pts, i, minimum=minimum)
        if collapsed is None:
            i += 1
            continue
        pts = collapsed
        i = max(1, i - 1)
    return pts


def _remove_spikes(points: np.ndarray, *, spike_length_pt: float, closed: bool) -> np.ndarray:
    """Remove interior single-vertex protrusions that immediately reverse
    direction: a vertex qualifies iff its turn angle exceeds
    ``_SPIKE_TURN_ANGLE_DEG`` (a near-total reversal, not merely a sharp
    corner) *and* both edges meeting it are shorter than
    ``spike_length_pt`` (bounding this to genuinely short protrusions, not
    long sharp features). Endpoints (junctions/anchor) are never scanned
    or removed. Never drops below the ``Arc`` floor (2 open / 4 closed). A
    removal that would introduce a self-intersection against the arc's own
    other segments is skipped, never partially applied (this stage's
    established skip-never-retry failure policy, module doc §12) -- "skip"
    here means the specific removal is never attempted at reduced
    strength, not that the vertex is abandoned forever: a later collapse
    elsewhere can remove the third-party geometry that was blocking it, so
    the scan is repeated to a fixed point (§9's idempotence proof depends
    on this: a single forward pass alone is not always sufficient, as a
    blocked-then-later-unblocked vertex demonstrates).

    Deterministic: each repetition is a pure function of its input
    polyline and threshold; the point count strictly decreases or is
    unchanged between repetitions, so the fixed point (no scan makes any
    change) is reached in at most ``P`` repetitions and the result is
    independent of anything but the input array and configuration.
    """
    minimum = 4 if closed else 2
    pts = points
    while True:
        next_pts = _remove_spikes_once(pts, spike_length_pt=spike_length_pt, minimum=minimum)
        if next_pts.shape[0] == pts.shape[0]:
            return next_pts
        pts = next_pts


def _spike_removal(
    arcs: tuple[Arc, ...], *, config: GeometryNormalizeConfig
) -> tuple[tuple[Arc, ...], int]:
    """Pass 2: remove degenerate single-vertex protrusions per arc (§8.2 of
    the module doc). ``arc_id``/``left_region``/``right_region``/``closed``
    are always preserved; an unmodified arc is returned as the identical
    object, a modified arc as a new ``Arc`` with only ``points`` replaced.
    """
    spike_length_pt = config.spike_length_mm * _MM_TO_PT
    removed = 0
    out: list[Arc] = []
    for arc in arcs:
        cleaned = _remove_spikes(arc.points, spike_length_pt=spike_length_pt, closed=arc.closed)
        if cleaned.shape[0] == arc.points.shape[0]:
            out.append(arc)
            continue
        removed += arc.points.shape[0] - cleaned.shape[0]
        out.append(
            Arc(
                arc_id=arc.arc_id,
                points=cleaned,
                left_region=arc.left_region,
                right_region=arc.right_region,
                closed=arc.closed,
            )
        )
    return tuple(out), removed


# ------------------------------------------------- minimum gap enforcement ---
#
# Pass 3, GAP_REPAIR_DESIGN.md. Operates purely on Arc.points: no Face, no
# CurveSet, no rasterization, no bezier. See that document's numbered
# sections (referenced below by number) for the full mathematical
# definition, proofs, and rationale this implementation follows exactly.
#
# Relative tolerance for the post-displacement threshold comparison,
# matching MATH_SPEC's own post-scaling re-check idiom (1e-4 relative,
# MATH_SPEC §6.1) -- a witness point moved by exactly the computed offset
# can land a floating-point hair short of the threshold once the true
# post-move minimum is realized on a nearby segment-interior projection
# rather than the original two witness points; this tolerance absorbs that
# numerical slack without weakening the threshold itself
# (GAP_REPAIR_DESIGN.md §1.4: ``==`` at the threshold is legal already).
_GAP_RELATIVE_TOLERANCE = 1e-4


def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on segment ``ab`` to point ``p``, and the parameter
    ``t`` in [0, 1] along ``ab`` (0 = a, 1 = b) -- standard projection
    clamped to the segment.

    Returns the *exact original array* ``a``/``b`` (not a recomputed
    ``a + t * ab``) when the clamped projection lands on an endpoint --
    ``a + 1.0 * (b - a)`` is not guaranteed bit-identical to ``b`` under
    floating point, which would otherwise make ``_vertex_index_for_point``
    fail to recognize a witness that IS an existing vertex (found during
    Sprint 36B.3 implementation: this silently forced an unnecessary
    vertex insertion, destabilizing the taper computation for gap
    fractions that should have been well within the algorithm's reach).
    """
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0.0:
        return a, 0.0
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    if t <= 0.0:
        return a, 0.0
    if t >= 1.0:
        return b, 1.0
    return a + t * ab, t


def _segment_segment_distance(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimum distance between segments ``ab`` and ``cd`` (GAP_REPAIR_
    DESIGN.md §1.2); returns ``(distance, point_on_ab, point_on_cd)``.

    If the segments properly intersect or touch, the minimum is exactly 0
    at the crossing point (checked first via MATH_SPEC §7.3's predicate,
    plus an on-segment containment check for the touching/collinear
    boundary cases the *proper*-intersection test alone excludes).
    Otherwise, the minimum over two non-crossing segments always occurs at
    an endpoint-to-segment projection, so checking all four endpoint
    projections is exact.
    """
    if _segments_properly_intersect(a, b, c, d):
        # Exact intersection point via the standard parametric solve;
        # denominator is nonzero here because a proper intersection rules
        # out parallel/collinear segments.
        r, s = b - a, d - c
        denom = float(r[0] * s[1] - r[1] * s[0])
        t = float(((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom)
        point = a + t * r
        return 0.0, point, point

    candidates = [
        (_closest_point_on_segment(a, c, d)[0], a),
        (_closest_point_on_segment(b, c, d)[0], b),
        (c, _closest_point_on_segment(c, a, b)[0]),
        (d, _closest_point_on_segment(d, a, b)[0]),
    ]
    best_dist = float("inf")
    best_pair = (a, c)
    for on_cd, on_ab in candidates:
        dist = float(np.linalg.norm(on_ab - on_cd))
        if dist < best_dist:
            best_dist = dist
            best_pair = (on_ab, on_cd)
    return best_dist, best_pair[0], best_pair[1]


def _arc_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])


def _grid_cells_for_bbox(
    bbox: tuple[float, float, float, float], *, cell_size: float
) -> set[tuple[int, int]]:
    """All grid cells (§2.1) an expanded bbox overlaps."""
    x0, y0, x1, y1 = bbox
    cx0, cy0 = int(np.floor(x0 / cell_size)), int(np.floor(y0 / cell_size))
    cx1, cy1 = int(np.floor(x1 / cell_size)), int(np.floor(y1 / cell_size))
    return {(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)}


def _shares_endpoint(a: Arc, b: Arc) -> bool:
    """§1.1 exclusion rule: arcs meeting at a common junction are not
    eligible for gap measurement -- compared exactly, not by proximity."""
    a_ends = (a.points[0], a.points[-1])
    b_ends = (b.points[0], b.points[-1])
    return any(np.array_equal(x, y) for x in a_ends for y in b_ends)


def _candidate_pairs(
    arcs: tuple[Arc, ...], *, min_gap_pt: float
) -> list[tuple[int, int]]:
    """§2.1 broad phase: uniform grid, cell size 2*min_gap_pt, boxes
    expanded by min_gap_pt. Returns canonicalized (i, j) index pairs, i < j
    (into ``arcs``), deduplicated, sorted -- a fixed, deterministic order
    independent of grid bucket iteration (§2.3)."""
    cell_size = 2.0 * min_gap_pt
    cells: dict[tuple[int, int], list[int]] = {}
    for idx, arc in enumerate(arcs):
        x0, y0, x1, y1 = _arc_bbox(arc.points)
        expanded = (x0 - min_gap_pt, y0 - min_gap_pt, x1 + min_gap_pt, y1 + min_gap_pt)
        for cell in _grid_cells_for_bbox(expanded, cell_size=cell_size):
            cells.setdefault(cell, []).append(idx)

    pairs: set[tuple[int, int]] = set()
    for members in cells.values():
        for m in range(len(members)):
            for n in range(m + 1, len(members)):
                i, j = members[m], members[n]
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _segment_bbox_min_distance(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> float:
    """Lower bound on the true segment-segment distance: distance between
    the segments' axis-aligned bounding boxes (0 if they overlap). Cheap and
    exact as a lower bound (a segment never leaves its own bbox), so any
    pair whose bbox distance already exceeds the current best true distance
    cannot improve it and can be skipped without ever computing the exact
    (and much costlier) segment-segment distance -- a standard
    branch-and-bound prune that changes nothing about which pair is
    eventually selected as the minimum (GAP_REPAIR_DESIGN.md §2.2's exact
    result is unchanged; only unproductive exact evaluations are skipped)."""
    ax0, ax1 = (a0[0], a1[0]) if a0[0] <= a1[0] else (a1[0], a0[0])
    ay0, ay1 = (a0[1], a1[1]) if a0[1] <= a1[1] else (a1[1], a0[1])
    bx0, bx1 = (b0[0], b1[0]) if b0[0] <= b1[0] else (b1[0], b0[0])
    by0, by1 = (b0[1], b1[1]) if b0[1] <= b1[1] else (b1[1], b0[1])
    dx = max(0.0, bx0 - ax1, ax0 - bx1)
    dy = max(0.0, by0 - ay1, ay0 - by1)
    return float(math.hypot(dx, dy))


def _min_arc_pair_distance(
    a: Arc, b: Arc
) -> tuple[float, int, int, np.ndarray, np.ndarray]:
    """§2.2 narrow phase: exact minimum distance between two arc polylines,
    with the witness segment indices and closest points (§1.3, ties broken
    by lexicographically smallest segment-index pair).

    A cheap per-segment-pair bounding-box lower bound (see
    ``_segment_bbox_min_distance``) prunes pairs that cannot possibly beat
    the current best before paying for the exact (intersection-aware)
    distance computation -- this is load-bearing for real photos split into
    many small arcs (``stages/graph/split_large.py``'s filler/rim cells),
    where the naive O(segments_a * segments_b) sweep over every pair became
    the pipeline's dominant cost. The exact result is bit-identical to the
    unpruned sweep; only which pairs get the exact check changes."""
    best_dist = float("inf")
    best = (0, 0, a.points[0], b.points[0])
    for si in range(a.points.shape[0] - 1):
        a0, a1 = a.points[si], a.points[si + 1]
        for sj in range(b.points.shape[0] - 1):
            b0, b1 = b.points[sj], b.points[sj + 1]
            if _segment_bbox_min_distance(a0, a1, b0, b1) >= best_dist:
                continue
            dist, pa, pb = _segment_segment_distance(a0, a1, b0, b1)
            if dist < best_dist:
                best_dist = dist
                best = (si, sj, pa, pb)
    return best_dist, best[0], best[1], best[2], best[3]


def _vertex_index_for_point(points: np.ndarray, point: np.ndarray, seg_index: int) -> int | None:
    """If ``point`` coincides exactly with an existing vertex of segment
    ``seg_index`` (``points[seg_index]`` or ``points[seg_index + 1]``),
    return that vertex's index; otherwise ``None`` (the point is strictly
    interior to the segment, §3.4)."""
    if np.array_equal(point, points[seg_index]):
        return seg_index
    if np.array_equal(point, points[seg_index + 1]):
        return seg_index + 1
    return None


def _insert_witness_vertex(points: np.ndarray, point: np.ndarray, seg_index: int) -> tuple[np.ndarray, int]:
    """§3.4: insert ``point`` as a new vertex strictly inside segment
    ``seg_index``, returning the updated array and the new vertex's index.
    """
    new_points = np.concatenate(
        [points[: seg_index + 1], point[None, :], points[seg_index + 1 :]], axis=0
    )
    return new_points, seg_index + 1


def _is_endpoint_index(points: np.ndarray, index: int) -> bool:
    return index == 0 or index == points.shape[0] - 1


def _moved_edge_indices(points: np.ndarray, moved_indices: set[int]) -> list[tuple[int, int]]:
    """Every edge ``(j, j+1)`` with at least one endpoint in
    ``moved_indices`` -- these are the only edges whose geometry changes
    when exactly the vertices in ``moved_indices`` move; every other edge
    is untouched and does not need re-checking (locality argument, as in
    ``_removal_would_self_intersect``, Pass 2). Two tapered vertices can be
    adjacent to each other (e.g. indices differing by 1 under
    ``_TAPER_FRACTIONS``' 2-step reach), in which case the edge *between*
    them has *both* endpoints moved -- this function includes that edge
    exactly once, so its final (both-endpoints-moved) geometry is checked,
    not silently skipped.
    """
    n = points.shape[0]
    edges: set[tuple[int, int]] = set()
    for idx in moved_indices:
        if idx - 1 >= 0:
            edges.add((idx - 1, idx))
        if idx + 1 < n:
            edges.add((idx, idx + 1))
    return sorted(edges)


_TAPER_FRACTIONS: tuple[float, ...] = (1.0, 0.5, 0.25)  # witness vertex, then 1-2 neighbors each side


def _tapered_moves(points: np.ndarray, center_idx: int, offset: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """§3.2 point 5: the witness vertex moves by the full ``offset``; the
    immediately adjacent 1-2 vertices on each side move by a linearly
    decaying fraction of the same offset (never touching an endpoint),
    so the arc does not develop a sharp kink at the repair site. Returns
    ``(index, new_position)`` pairs, closest vertex first.
    """
    n = points.shape[0]
    moves = [(center_idx, points[center_idx] + offset * _TAPER_FRACTIONS[0])]
    for step, fraction in enumerate(_TAPER_FRACTIONS[1:], start=1):
        for direction in (-1, 1):
            idx = center_idx + direction * step
            if _is_endpoint_index(points, idx) or not (0 <= idx < n):
                continue
            moves.append((idx, points[idx] + offset * fraction))
    return moves


def _tapered_moves_would_self_intersect(
    points: np.ndarray, moves: list[tuple[int, np.ndarray]], other_points: np.ndarray
) -> bool:
    """§5.1 sidedness guard for a *batch* of simultaneous tapered moves.

    Builds the fully-moved array (every move in ``moves`` applied at
    once) up front, then checks every edge touching at least one moved
    vertex -- including an edge *between two moved vertices*, whose final
    geometry (both endpoints at their new positions) is only correctly
    represented once all moves are already applied. Checking each move's
    guard independently against the stale, unmoved array (as an earlier
    version of this function did) misses exactly this case: two adjacent
    tapered vertices whose *final* shared edge crosses foreign geometry,
    even though each vertex's move looked safe in isolation against the
    old position of its neighbor.
    """
    moved_indices = {idx for idx, _ in moves}
    moved_points = points.copy()
    for idx, new_pos in moves:
        moved_points[idx] = new_pos
    changed_edges = _moved_edge_indices(points, moved_indices)

    if _changed_edges_cross_own_arc(moved_points, changed_edges):
        return True
    return _changed_edges_cross_other_arc(moved_points, changed_edges, other_points)


def _changed_edges_cross_own_arc(
    moved_points: np.ndarray, changed_edges: list[tuple[int, int]]
) -> bool:
    """Own-arc half of the batch guard: every changed edge against every
    other, non-adjacent edge of the same (fully-moved) arc."""
    n = moved_points.shape[0]
    for lo, hi in changed_edges:
        pa, pb = moved_points[lo], moved_points[hi]
        for j in range(n - 1):
            if (j, j + 1) in changed_edges or j in (lo, hi) or j + 1 in (lo, hi):
                continue
            if _segments_properly_intersect(pa, pb, moved_points[j], moved_points[j + 1]):
                return True
    return False


def _changed_edges_cross_other_arc(
    moved_points: np.ndarray, changed_edges: list[tuple[int, int]], other_points: np.ndarray
) -> bool:
    """Cross-arc half of the batch guard: every changed edge against the
    other arc's (unmoved by this call) geometry."""
    for lo, hi in changed_edges:
        pa, pb = moved_points[lo], moved_points[hi]
        for k in range(other_points.shape[0] - 1):
            if _segments_properly_intersect(pa, pb, other_points[k], other_points[k + 1]):
                return True
    return False


def _apply_moves(points: np.ndarray, moves: list[tuple[int, np.ndarray]]) -> np.ndarray:
    out = points.copy()
    for idx, new_pos in moves:
        out[idx] = new_pos
    return out


def _repair_gap(
    arcs: list[Arc],
    i: int,
    j: int,
    *,
    min_gap_pt: float,
    max_displacement_pt: float,
) -> bool:
    """§3.2-3.4: attempt to repair one confirmed narrow gap between
    ``arcs[i]`` and ``arcs[j]`` in place (mutating the list); returns
    whether a repair was applied. Symmetric displacement, bounded by
    ``max_displacement_pt`` (§4.1), gated by the sidedness guard (§5.1) on
    both sides independently -- if either side's guard fails, no
    displacement is applied to either side (an asymmetric partial repair
    is never legal, §3.2's "half the fix, computed once from the pair").

    The witness vertex on each side moves the full computed offset; 1-2
    immediately adjacent vertices taper by a decaying fraction of the same
    offset (§3.2 point 5) so an unmoved original vertex right next to the
    witness cannot remain the new closest-approach point after the
    witness itself moves away.
    """
    a, b = arcs[i], arcs[j]
    dist, si, sj, pa, pb = _min_arc_pair_distance(a, b)
    # Same relative tolerance as the exit check below: a distance already
    # within it of the threshold is treated as satisfying the floor
    # (GAP_REPAIR_DESIGN.md §1.4, "== is legal"), so a pair a prior call
    # already brought within tolerance is never re-attempted -- required
    # for idempotence (§9 property 3): without matching tolerances here,
    # a post-repair distance accepted by the exit check as "clear enough"
    # could still fail this entry check on the next call and trigger a
    # second (wasted, and potentially different) repair attempt.
    if dist >= min_gap_pt * (1.0 - _GAP_RELATIVE_TOLERANCE):
        return False

    delta = min_gap_pt - dist
    half = delta / 2.0
    if half > max_displacement_pt:
        return False  # §4.1: refused outright, never partially applied

    sep = pb - pa
    norm = float(np.linalg.norm(sep))
    if norm <= 0.0:
        return False  # coincident witness points: not a repairable "gap"
    u = sep / norm

    a_points = a.points
    a_idx = _vertex_index_for_point(a_points, pa, si)
    if a_idx is None:
        a_points, a_idx = _insert_witness_vertex(a_points, pa, si)
    b_points = b.points
    b_idx = _vertex_index_for_point(b_points, pb, sj)
    if b_idx is None:
        b_points, b_idx = _insert_witness_vertex(b_points, pb, sj)

    if _is_endpoint_index(a_points, a_idx) or _is_endpoint_index(b_points, b_idx):
        return False  # §5.2: junction endpoints are never displaced

    a_moves = _tapered_moves(a_points, a_idx, -u * half)
    b_moves = _tapered_moves(b_points, b_idx, u * half)

    if _tapered_moves_would_self_intersect(a_points, a_moves, b_points):
        return False
    if _tapered_moves_would_self_intersect(b_points, b_moves, a_points):
        return False

    moved_a_points = _apply_moves(a_points, a_moves)
    moved_b_points = _apply_moves(b_points, b_moves)

    # §8/§9 "monotone improvement": the repair either fully clears the
    # threshold or is not attempted -- verified here, before committing,
    # not left to the caller's independent post-pass alone. A local
    # taper is not guaranteed to clear a pinch that spans more of the
    # polyline than the taper's fixed neighborhood reaches (e.g. two
    # densely-sampled, near-parallel arcs uniformly narrow apart over many
    # vertices); in that case this specific pair is skipped, never
    # committed as a partial fix.
    candidate_a = Arc(
        arc_id=a.arc_id, points=moved_a_points, left_region=a.left_region,
        right_region=a.right_region, closed=a.closed,
    )
    candidate_b = Arc(
        arc_id=b.arc_id, points=moved_b_points, left_region=b.left_region,
        right_region=b.right_region, closed=b.closed,
    )
    post_dist = _min_arc_pair_distance(candidate_a, candidate_b)[0]
    if post_dist < min_gap_pt * (1.0 - _GAP_RELATIVE_TOLERANCE):
        return False

    arcs[i] = candidate_a
    arcs[j] = candidate_b
    return True


def _minimum_gap_enforcement(
    arcs: tuple[Arc, ...], *, config: GeometryNormalizeConfig
) -> tuple[tuple[Arc, ...], int]:
    """Pass 3: symmetric constrained-displacement gap repair
    (GAP_REPAIR_DESIGN.md), operating purely on ``Arc.points`` -- no
    ``Face`` traversal, no ``CurveSet``, no rasterization, no bezier.

    Confirmed gaps are processed in the fixed total order
    ``(min(arc_id), max(arc_id))`` (§3.3), each exactly once; a repair
    updates both arcs' geometry before later gaps involving either arc are
    evaluated. After all repairs, every pair actually repaired is
    independently re-measured (never repaired silently, matching
    ``topology.py``'s convention); if any repaired pair's clearance is
    still below the threshold, a ``StageError`` is raised.
    """
    min_gap_pt = config.min_gap_mm * _MM_TO_PT
    # §4.1: max_displacement_pt bounds Delta/2 per side, capped by the
    # simplify.tolerance_mm ceiling every pass shares.
    max_displacement_pt = config.simplify_tolerance_mm * _MM_TO_PT

    arc_list = list(arcs)
    id_to_index = {arc.arc_id: idx for idx, arc in enumerate(arc_list)}
    candidates = _candidate_pairs(arc_list, min_gap_pt=min_gap_pt)

    # Canonicalize by arc_id (not list index) so ordering matches §3.3
    # exactly even if ``arcs`` is not already sorted by arc_id, and dedupe
    # any pair sharing a junction endpoint (§1.1 exclusion).
    ordered_pairs = sorted(
        {
            (min(arc_list[i].arc_id, arc_list[j].arc_id), max(arc_list[i].arc_id, arc_list[j].arc_id))
            for i, j in candidates
            if not _shares_endpoint(arc_list[i], arc_list[j])
        }
    )

    repaired_ids: list[tuple[int, int]] = []
    repaired_count = 0
    for id_a, id_b in ordered_pairs:
        i, j = id_to_index[id_a], id_to_index[id_b]
        applied = _repair_gap(
            arc_list, i, j, min_gap_pt=min_gap_pt, max_displacement_pt=max_displacement_pt
        )
        if applied:
            repaired_count += 1
            repaired_ids.append((id_a, id_b))

    for id_a, id_b in repaired_ids:
        i, j = id_to_index[id_a], id_to_index[id_b]
        dist = _min_arc_pair_distance(arc_list[i], arc_list[j])[0]
        if dist < min_gap_pt * (1.0 - _GAP_RELATIVE_TOLERANCE):
            raise StageError(
                f"geometry_normalize: gap between arcs {id_a} and {id_b} still "
                f"{dist:.4f}pt < {min_gap_pt:.4f}pt after repair",
                stage_name=STAGE_NAME,
                config_hash=_UNSET_HASH,
            )

    return tuple(arc_list), repaired_count


def _revert_crossing_walks(
    faces: tuple,  # type: ignore[type-arg]
    original_by_id: dict[int, Arc],
    arcs: tuple[Arc, ...],
) -> tuple[tuple[Arc, ...], int]:
    """Planarity guard: each of the three passes applies only local per-arc
    checks, so a correction can still make one face's ring cross itself or a
    sibling ring (a dedup/spike chord or a gap-repair displacement sweeping
    across a neighboring edge) — a topology I3 FATAL no downstream stage can
    undo, and one that also breaks the bezier stage's polyline-fallback
    repair (its "exact polylines are crossing-free" fixpoint is THIS stage's
    output). Any walk involved in a crossing has its modified arcs reverted
    to their exact pre-normalization points; the scan repeats until a full
    pass reverts nothing (the all-original state is the arc graph's planar
    partition — crossing-free by construction, so the loop terminates).
    Detection uses the canonical model/flatten predicates, matching the
    validator bitwise. Returns ``(arcs, n_arcs_reverted)``.
    """
    by_id = {a.arc_id: a for a in arcs}

    def _ring(walk: tuple[tuple[int, bool], ...]) -> np.ndarray:
        parts = []
        for i, (aid, rev) in enumerate(walk):
            pts = by_id[aid].points
            if rev:
                pts = pts[::-1]
            parts.append(pts if i == 0 else pts[1:])
        return np.concatenate(parts)

    def _crossing_arc_ids(face) -> list[int]:  # type: ignore[no-untyped-def]
        walks = list(face.all_walks())
        rings = [_ring(walk) for walk in walks]
        bad = [ring_self_intersects(ring) for ring in rings]
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                if rings_intersect(rings[i], rings[j]):
                    bad[i] = bad[j] = True
        return [aid for walk, is_bad in zip(walks, bad, strict=True) if is_bad for aid, _ in walk]

    reverted: set[int] = set()
    for _ in range(4):
        changed = False
        for face in faces:
            for aid in _crossing_arc_ids(face):
                if by_id[aid] is not original_by_id[aid]:
                    by_id[aid] = original_by_id[aid]
                    reverted.add(aid)
                    changed = True
        if not changed:
            break
    return tuple(by_id[a.arc_id] for a in arcs), len(reverted)


def normalize_geometry(
    arc_graph: ArcGraph,
    *,
    config: GeometryNormalizeConfig,
    config_hash: str = _UNSET_HASH,
) -> tuple[ArcGraph, dict[str, int]]:
    """Run the fixed three-pass sequence (§7 of the module doc) over
    ``arc_graph.arcs``; ``faces``/``work_scale`` are always carried over
    unchanged. Returns the new graph and the per-pass correction counts.

    If ``config.enabled`` is false, returns ``arc_graph`` unchanged (still
    re-stamped with this stage's provenance, per the module doc's
    ``enabled: False`` contract) and all-zero metrics.
    """
    if not config.enabled:
        return (
            ArcGraph(
                arcs=arc_graph.arcs,
                faces=arc_graph.faces,
                work_scale=arc_graph.work_scale,
                provenance=Provenance(
                    stage_name=STAGE_NAME,
                    stage_version=STAGE_VERSION,
                    config_hash=config_hash,
                    source_hash=arc_graph.provenance.source_hash,
                ),
            ),
            {
                "duplicates_removed": 0,
                "spikes_removed": 0,
                "gaps_repaired": 0,
                "crossings_reverted": 0,
            },
        )

    arcs, duplicates_removed = _duplicate_cleanup(arc_graph.arcs, config=config)
    arcs, spikes_removed = _spike_removal(arcs, config=config)
    arcs, gaps_repaired = _minimum_gap_enforcement(arcs, config=config)
    original_by_id = {a.arc_id: a for a in arc_graph.arcs}
    arcs, crossings_reverted = _revert_crossing_walks(arc_graph.faces, original_by_id, arcs)

    new_graph = ArcGraph(
        arcs=arcs,
        faces=arc_graph.faces,
        work_scale=arc_graph.work_scale,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=arc_graph.provenance.source_hash,
        ),
    )
    metrics = {
        "duplicates_removed": duplicates_removed,
        "spikes_removed": spikes_removed,
        "gaps_repaired": gaps_repaired,
        "crossings_reverted": crossings_reverted,
    }
    return new_graph, metrics


class GeometryNormalizeStage:
    """Stage wrapper: ``arc_graph`` -> ``arc_graph`` (replaced).

    Disabled: passes ``arc_graph`` through unchanged (re-stamped
    provenance only). Enabled: runs the fixed three-pass sequence, each
    currently a no-op placeholder (Sprint 36A.4; algorithms land later).
    Per-pass correction counts are bound to ``ctx`` as the
    ``geometry_normalize_metrics`` artifact, following the same
    ``PipelineContext.put`` mechanism every other stage uses to publish its
    own outputs -- no new reporting channel is introduced.
    """

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        simplify_tolerance_mm: float,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        self._config = GeometryNormalizeConfig(
            section or {}, simplify_tolerance_mm=simplify_tolerance_mm
        )
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("arc_graph",)

    @property
    def provides(self) -> tuple[str, ...]:
        return ("arc_graph", "geometry_normalize_metrics")

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        arc_graph = ctx.get("arc_graph")
        if not isinstance(arc_graph, ArcGraph):
            raise ConfigError("geometry_normalize requires an ArcGraph artifact")
        new_graph, metrics = normalize_geometry(
            arc_graph, config=self._config, config_hash=self._config_hash
        )
        ctx.put("arc_graph", new_graph)
        ctx.put("geometry_normalize_metrics", metrics)
