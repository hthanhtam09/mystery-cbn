"""Curve Fitting stage: arc polylines → cubic Bézier chains
(ENGINE_SPEC.md §18; Curve/CurveSet DATA_MODEL_SPEC §13–§14).

Four fitters are implemented behind one interface and compared in
``benchmarks/quality/test_curves_quality.py``; **Schneider is the default**
(Graphics Gems "An Algorithm for Automatically Fitting Digitized Curves"):

- ``schneider`` — least-squares cubic per corner-free run with Newton–Raphson
  reparameterization (≤ 4 iterations) and adaptive splitting at the max-error
  point with a centripetal tangent. Bounded error by construction, exact
  endpoint interpolation, G1 inside runs. *Default.*
- ``bezier`` — plain least-squares cubics over uniform chunks (no
  reparameterization, no adaptive split). Cheap, no error guarantee.
- ``chaikin`` — the Chaikin corner-cutting limit curve (= clamped uniform
  quadratic B-spline of the run polygon), degree-elevated to cubics. Very
  smooth, but does not interpolate interior vertices (systematic deviation
  up to half the local feature size) and emits one segment per vertex.
- ``catmull`` — Catmull–Rom spline through every vertex, one cubic per edge.
  Exact interpolation, but chases input noise (worst smoothness) and emits
  the most segments.

All fitters share the topology/corner frame:

- **Corners** — interior vertices with turn angle > ``corner_angle_deg`` are
  corners; arcs are split into corner-free runs and refitted independently,
  so corner positions survive bitwise and joints there are intentional C0
  (recorded in ``corner_indices``). A closed arc's anchor is a corner by
  definition.
- **Topology** — chain endpoints interpolate the arc's junction coordinates
  *bitwise* (watertightness at junctions is positional identity, not
  tolerance); faces are carried over from the ArcGraph unchanged.

``max_fit_error_pt`` is the max residual at the input vertices' parameters
(0 for interpolating fitters at vertices; the QM-04 sampled-deviation gate
runs in the quality benchmarks).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import pairwise

import numpy as np

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Provenance
from mysterycbn.model.vector import ArcGraph, BezierSegment, Curve, CurveSet
from mysterycbn.model.flatten import flatten_face_rings, ring_self_intersects, rings_intersect
from mysterycbn.stages.vector._face_area import (
    min_adjacent_face_area_pt2_by_arc,
    same_label_seam_arc_ids,
    tolerance_reference_area_pt2,
    tolerance_scale_for_area,
)

STAGE_NAME = "bezier"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

FIT_ERROR_MM_DEFAULT = 0.15  # core CurveConfig defaults
CORNER_ANGLE_DEG_DEFAULT = 65.0
# Fixed, area-independent tolerance scale for split_large filler/rim arcs
# (see fit_curves' filler_ids docstring) -- a rim cell's area can be large
# while its shape is a thin winding strip, so area-based scaling alone is
# not enough to keep its fit tight.
_FILLER_TOLERANCE_SCALE = 0.3
# Same-color seams only (see same_label_seam_arc_ids): loose on purpose —
# displacement across a same-label boundary cannot affect fidelity, and the
# larger budget lets the fitter replace the pixel staircase with long,
# rounded sweeps (the commercial hand-drawn look).
_SEAM_TOLERANCE_SCALE = 4.0
# Seam arcs also suppress most corner detection (a corner is an intentional
# sharp C0 joint — the opposite of the rounded hand-drawn seam look): only
# near-reversals survive as corners.
_SEAM_CORNER_ANGLE_DEG = 120.0
_MAX_REPARAM = 4
_MAX_DEPTH = 32
_CHUNK = 8  # uniform chunk size of the plain least-squares fitter

# Sprint 41: homogeneity_split() thresholds (Sprint 40 forensic finding --
# bbox aspect ratio and edge-length ratio both correlate with ill-conditioned
# Schneider normal-equations systems across 588 sampled production segments;
# bbox aspect ratio was the strongest single predictor, r=0.4581 vs.
# log10(condition number)). A run is split when EITHER ratio exceeds its
# threshold. Thresholds are set above the population's typical range
# (median bbox aspect ~0.26 in log10, i.e. ~1.8 raw; median edge ratio
# ~2.24) so ordinary runs are never split, while the traced production
# failure (bbox aspect ~91, edge ratio ~63.6, both >99th percentile) is
# reliably caught.
SEGMENT_MAX_BBOX_ASPECT = 15.0
SEGMENT_MAX_EDGE_RATIO = 15.0

_Run = np.ndarray  # (P, 2) float64 vertices of a corner-free run
_Fitter = Callable[[_Run, float], tuple[list[np.ndarray], float]]


# The pre-gate repair reuses the topology validator's OWN flatten +
# self-intersection predicate (model/flatten.py) so it sees bitwise the same
# geometry the validator re-derives — two near-identical reimplementations
# each let a borderline crossing slip through the pre-gate and FATAL at the
# gate before this was unified.
_SELF_X_FLATTEN_PT = 0.1 * PT_PER_INCH / MM_PER_INCH  # 0.1mm, same as validator


def fit_error_pt(fit_error_mm: float) -> float:
    """Configured fit tolerance converted to pt."""
    if fit_error_mm <= 0:
        raise ConfigError(f"fit_error_mm must be > 0, got {fit_error_mm}")
    return fit_error_mm * PT_PER_INCH / MM_PER_INCH


# ---------------------------------------------------------------- shared ---


def _chord_params(pts: _Run) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    return np.asarray(u / u[-1])


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _end_tangent(pts: _Run) -> np.ndarray:
    """Unit tangent into the run: average of the first ≤ 3 chords (§18.2).
    Pass the reversed polyline to get the tangent into the run at its end."""
    chords = [pts[i + 1] - pts[i] for i in range(min(3, len(pts) - 1))]
    return _normalize(np.asarray(sum(_normalize(c) for c in chords), dtype=np.float64))


def _line_segment(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Exact line as a cubic: control points at ⅓ and ⅔ chord (§18.5)."""
    return np.stack([p0, p0 + (p1 - p0) / 3.0, p0 + 2.0 * (p1 - p0) / 3.0, p1])


def _evaluate(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    b0 = (1 - u) ** 3
    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u**2 * (1 - u)
    b3 = u**3
    return np.asarray(
        ctrl[0] * b0[:, None]
        + ctrl[1] * b1[:, None]
        + ctrl[2] * b2[:, None]
        + ctrl[3] * b3[:, None]
    )


def _max_error(pts: _Run, ctrl: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    dev = np.linalg.norm(_evaluate(ctrl, u) - pts, axis=1)
    idx = int(np.argmax(dev))
    return float(dev[idx]), idx


def _evaluate_derivative(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    """`B'(t) = 3 * sum_{i=0}^{2} (b_{i+1} - b_i) * B_i^2(t)` (MATH_SPEC.md §9.1)."""
    d = 3.0 * np.diff(ctrl, axis=0)
    b0 = (1 - u) ** 2
    b1 = 2 * u * (1 - u)
    b2 = u**2
    return np.asarray(d[0] * b0[:, None] + d[1] * b1[:, None] + d[2] * b2[:, None])


_LOOP_CHECK_SAMPLES = 33  # Sprint 43: fixed, deterministic sample count over the FULL parameter
# domain [0, 1] (uniform, endpoints included). MATH_SPEC.md §9.2 names the predicate
# `B'(t)*chord_dir > 0` but only specifies it as "checked at samples" -- the original
# implementation reused the fit's own `u` (the input points' chord-length parameters), which
# left the interior of the curve between those points unchecked (confirmed in production: a
# curve whose true derivative reverses over ~42% of [0,1] can still pass with `u` having only
# 3 points, none of which happen to land in the reversing region). 33 uniform samples is a
# fixed, cheap, O(1)-per-segment cost (independent of input point count) that is dense enough
# to catch the loops observed in production (single contiguous reversal spanning tens of
# percent of the domain) while remaining far cheaper than the O(n) per-arc fitting work it
# sits inside.
_LOOP_CHECK_T = np.linspace(0.0, 1.0, _LOOP_CHECK_SAMPLES)


def _has_loop(pts: _Run, ctrl: np.ndarray, u: np.ndarray) -> bool:
    """MATH_SPEC.md §9.2 "Failure cases": oscillating data can produce
    loops, detected by checking `B'(t)*chord_dir > 0` -- a violation (the
    curve's tangent turning against its own overall chord direction)
    forces a split.

    Sprint 43: evaluated over a fixed, dense, uniform sampling of the
    FULL parameter domain [0, 1] (not just the fit's own `u` values,
    which only cover the input points' own chord-length parameters and
    can miss a reversal occurring strictly between them). ``u`` is
    accepted for signature compatibility but no longer used -- the
    predicate now checks the curve's interior directly, per Sprint 43.
    """
    del u
    chord_dir = pts[-1] - pts[0]
    chord_norm = float(np.linalg.norm(chord_dir))
    if chord_norm <= 0.0:
        return False
    chord_dir = chord_dir / chord_norm
    deriv = _evaluate_derivative(ctrl, _LOOP_CHECK_T)
    return bool(np.any((deriv @ chord_dir) <= 0.0))


# ------------------------------------------------------------- schneider ---


def _generate_bezier(
    pts: _Run, u: np.ndarray, t_left: np.ndarray, t_right: np.ndarray
) -> np.ndarray:
    """Least-squares cubic with fixed end tangent directions (Graphics Gems).

    MATH_SPEC.md §9.2 "Numerical stability": the 2×2 normal-equations
    system is near-singular when `t̂_L ≈ ±t̂_R` and points are nearly
    collinear (det → 0). Guard: if `det < 1e-12·‖A‖²` or any `α ≤ 0` or
    `α > 3·chord`, fall back to the heuristic `α_L = α_R = chord/3`
    (Wu/Schneider fallback) -- ``‖A‖²`` is the Gram-matrix trace `c00+c11`
    (the squared norm of the two-column design matrix `A = [A_L | A_R]`).
    """
    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u**2 * (1 - u)
    a1 = t_left[None, :] * b1[:, None]
    a2 = t_right[None, :] * b2[:, None]
    rhs = pts - (pts[0] * ((1 - u) ** 3 + b1)[:, None] + pts[-1] * (u**3 + b2)[:, None])
    c00 = float((a1 * a1).sum())
    c01 = float((a1 * a2).sum())
    c11 = float((a2 * a2).sum())
    x0 = float((a1 * rhs).sum())
    x1 = float((a2 * rhs).sum())
    det = c00 * c11 - c01 * c01
    chord = float(np.linalg.norm(pts[-1] - pts[0]))

    a_norm_sq = c00 + c11
    # <= so a fully degenerate run (det == 0 AND a_norm_sq == 0, e.g. zero
    # tangents) also takes the fallback instead of dividing by zero.
    if det <= 1e-12 * a_norm_sq:
        alpha_l = alpha_r = chord / 3.0
    else:
        alpha_l = (x0 * c11 - x1 * c01) / det
        alpha_r = (c00 * x1 - c01 * x0) / det
        if alpha_l <= 0.0 or alpha_r <= 0.0 or alpha_l > 3.0 * chord or alpha_r > 3.0 * chord:
            alpha_l = alpha_r = chord / 3.0
    return np.stack([pts[0], pts[0] + t_left * alpha_l, pts[-1] + t_right * alpha_r, pts[-1]])


def _reparameterize(pts: _Run, ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Newton–Raphson step of the foot-point parameters."""
    d1 = 3.0 * np.diff(ctrl, axis=0)  # derivative control points
    d2 = 2.0 * np.diff(d1, axis=0)
    q = _evaluate(ctrl, u) - pts
    b0 = (1 - u) ** 2
    b1 = 2 * u * (1 - u)
    b2 = u**2
    qp = d1[0] * b0[:, None] + d1[1] * b1[:, None] + d1[2] * b2[:, None]
    qpp = d2[0] * (1 - u)[:, None] + d2[1] * u[:, None]
    num = (q * qp).sum(axis=1)
    den = (qp * qp).sum(axis=1) + (q * qpp).sum(axis=1)
    step = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return np.asarray(np.clip(u - step, 0.0, 1.0))


def _fit_single(
    pts: _Run,
    t_left: np.ndarray,
    t_right: np.ndarray,
    tolerance: float,
    max_reparam: int,
) -> tuple[np.ndarray, float, int]:
    """One least-squares cubic with ≤ ``max_reparam`` Newton–Raphson refits
    (§18.2); returns (control points, max error, max-error index)."""
    u = _chord_params(pts)
    ctrl = _generate_bezier(pts, u, t_left, t_right)
    err, split = _max_error(pts, ctrl, u)
    if err <= tolerance or err > 16.0 * tolerance:  # done, or Newton hopeless
        return ctrl, err, split
    for _ in range(max_reparam):
        prev = err
        u = _reparameterize(pts, ctrl, u)
        ctrl = _generate_bezier(pts, u, t_left, t_right)
        err, split = _max_error(pts, ctrl, u)
        if err <= tolerance or err > 0.95 * prev:  # fitted, or stalled: split
            break
    return ctrl, err, split


def _fit_schneider_run(
    pts: _Run,
    tolerance: float,
    *,
    max_reparam: int = _MAX_REPARAM,
) -> tuple[list[np.ndarray], float]:
    """Schneider fitting of one corner-free run; returns (segments, max err)."""

    def recurse(
        pts: _Run, t_left: np.ndarray, t_right: np.ndarray, depth: int
    ) -> tuple[list[np.ndarray], float]:
        if len(pts) == 2:
            return [_line_segment(pts[0], pts[1])], 0.0
        ctrl, err, split = _fit_single(pts, t_left, t_right, tolerance, max_reparam)
        # MATH_SPEC.md §9.2 "Failure cases": a fit within tolerance but
        # whose tangent reverses against the run's own chord direction
        # (B'(t)*chord_dir <= 0 at a t_k) is a loop -- accepting it would
        # ship a curve that doubles back on itself; the violation forces a
        # split, exactly as an out-of-tolerance fit would.
        if err <= tolerance and not _has_loop(pts, ctrl, _chord_params(pts)):
            return [ctrl], err
        if depth >= _MAX_DEPTH:  # floor: per-edge line segments (§18 failure mode)
            segs = [_line_segment(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            return segs, 0.0
        split = min(max(split, 1), len(pts) - 2)
        t_center = _normalize(pts[split - 1] - pts[split + 1])  # centripetal estimate
        left, err_l = recurse(pts[: split + 1], t_left, t_center, depth + 1)
        right, err_r = recurse(pts[split:], -t_center, t_right, depth + 1)
        return left + right, max(err_l, err_r)

    return recurse(pts, _end_tangent(pts), _end_tangent(pts[::-1]), 0)


# ----------------------------------------------------------- comparators ---


def _fit_leastsq_run(pts: _Run, tolerance: float) -> tuple[list[np.ndarray], float]:
    """Plain least-squares cubics over uniform chunks (no reparam, no split)."""
    del tolerance  # no error control by design — that is the comparison
    segments: list[np.ndarray] = []
    worst = 0.0
    for start in range(0, len(pts) - 1, _CHUNK):
        chunk = pts[start : start + _CHUNK + 1]
        if len(chunk) == 2:
            segments.append(_line_segment(chunk[0], chunk[1]))
            continue
        u = _chord_params(chunk)
        ctrl = _generate_bezier(chunk, u, _end_tangent(chunk), _end_tangent(chunk[::-1]))
        err, _ = _max_error(chunk, ctrl, u)
        worst = max(worst, err)
        segments.append(ctrl)
    return segments, worst


def _fit_chaikin_run(pts: _Run, tolerance: float) -> tuple[list[np.ndarray], float]:
    """Chaikin limit curve: clamped uniform quadratic B-spline of the run
    polygon, degree-elevated to cubics. One segment per interior vertex."""
    del tolerance
    if len(pts) == 2:
        return [_line_segment(pts[0], pts[1])], 0.0
    quads = []
    for i in range(1, len(pts) - 1):
        q0 = pts[0] if i == 1 else (pts[i - 1] + pts[i]) / 2.0
        q2 = pts[-1] if i == len(pts) - 2 else (pts[i] + pts[i + 1]) / 2.0
        quads.append((q0, pts[i], q2))
    segments = [
        np.stack([q0, q0 + 2.0 / 3.0 * (q1 - q0), q2 + 2.0 / 3.0 * (q1 - q2), q2])
        for q0, q1, q2 in quads
    ]
    # Deviation of the limit curve at interior vertices: |mid(q0,q2)/2+q1/2 − p|.
    worst = 0.0
    for (q0, q1, q2), p in zip(quads, pts[1:-1], strict=True):
        worst = max(worst, float(np.linalg.norm((q0 + 2 * q1 + q2) / 4.0 - p)))
    return segments, worst


def _fit_catmull_run(pts: _Run, tolerance: float) -> tuple[list[np.ndarray], float]:
    """Catmull–Rom through every vertex, one cubic per edge (exact at data)."""
    del tolerance
    if len(pts) == 2:
        return [_line_segment(pts[0], pts[1])], 0.0
    tangents = np.empty_like(pts)
    tangents[1:-1] = (pts[2:] - pts[:-2]) / 2.0
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    segments = [
        np.stack(
            [pts[i], pts[i] + tangents[i] / 3.0, pts[i + 1] - tangents[i + 1] / 3.0, pts[i + 1]]
        )
        for i in range(len(pts) - 1)
    ]
    return segments, 0.0


_FITTERS: dict[str, _Fitter] = {
    "schneider": _fit_schneider_run,
    "bezier": _fit_leastsq_run,
    "chaikin": _fit_chaikin_run,
    "catmull": _fit_catmull_run,
}


# ------------------------------------------------------------- assembly ---


def _corner_split(pts: _Run, corner_angle_deg: float) -> list[int]:
    """Indices of corner vertices (interior turn angle > threshold)."""
    if len(pts) < 3:
        return []
    v_in = pts[1:-1] - pts[:-2]
    v_out = pts[2:] - pts[1:-1]
    dot = (v_in * v_out).sum(axis=1)
    norms = np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1)
    cos = np.divide(dot, norms, out=np.ones_like(dot), where=norms != 0)
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return [int(i) + 1 for i in np.flatnonzero(turn > corner_angle_deg)]


def _run_bbox_aspect(pts: _Run) -> float:
    """max(width, height) / max(min(width, height), eps) of the run's bbox.

    An axis-aligned (or near-axis-aligned) straight run has a degenerate,
    near-zero-width bbox on one axis by construction -- that is a normal,
    healthy shape (a plain straight line), not evidence of ill-conditioning,
    so it must not be reported as an extreme aspect ratio. Returns 1.0
    (the "square," non-suspicious value) whenever the run's points are
    collinear within a small tolerance, matching this function's job of
    flagging genuinely 2-D-oddly-shaped runs only.
    """
    span = pts.max(axis=0) - pts.min(axis=0)
    w, h = float(span[0]), float(span[1])
    diag = float(np.hypot(w, h))
    if diag <= 1e-9:
        return 1.0
    # perpendicular distance of every point from the p0->p_last chord line;
    # near-zero everywhere means the run is straight, regardless of how thin
    # its bbox is on one axis.
    chord = pts[-1] - pts[0]
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm > 1e-9:
        chord_dir = chord / chord_norm
        rel = pts - pts[0]
        perp = np.abs(rel[:, 0] * chord_dir[1] - rel[:, 1] * chord_dir[0])
        if float(perp.max()) <= 1e-6 * max(chord_norm, 1.0):
            return 1.0
    return max(w, h) / max(min(w, h), 1e-9)


def _run_edge_ratio(pts: _Run) -> float:
    """max(edge length) / max(min(edge length), eps) along the run's polyline."""
    edges = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(edges.max() / max(float(edges.min()), 1e-9))


def _homogeneity_split_point(pts: _Run) -> int:
    """Deterministic interior split index for a run that fails the
    homogeneity check: the vertex adjacent to the largest edge-length
    transition (the point where consecutive edge lengths differ most),
    falling back to the midpoint if the run is too short to have an
    interior transition (never a first/last index, so the two resulting
    runs are always non-empty and have at least 2 points each)."""
    n = len(pts)
    edges = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if len(edges) >= 2:
        transitions = np.abs(np.diff(edges))  # |edge[i+1] - edge[i]|, indexed 0..len(edges)-2
        # transitions[k] is the jump between edge k and edge k+1, straddling vertex k+1
        split = int(np.argmax(transitions)) + 1
        split = min(max(split, 1), n - 2)
        return split
    return n // 2  # midpoint fallback (n==2 has no interior point; guarded by caller)


def homogeneity_split(pts: _Run) -> list[int]:
    """Sprint 41: second segmentation pass, applied to a single corner-free
    run (the output of ``_corner_split``). Returns extra interior split
    indices (local to ``pts``) needed to keep every sub-run within the
    ``SEGMENT_MAX_BBOX_ASPECT`` / ``SEGMENT_MAX_EDGE_RATIO`` bounds --
    Sprint 40's forensic finding that both ratios predict ill-conditioned
    Schneider normal-equations systems, with bbox aspect the strongest
    single predictor (r=0.4581 across 588 sampled production segments).

    Deterministic, non-recursive-to-infinity: each violating run is split
    exactly once at ``_homogeneity_split_point`` (the largest edge-length
    transition, i.e. the point most likely to separate two differently
    shaped sub-runs), and each resulting half is re-checked exactly once
    more -- bounded by a small fixed depth (4), not open recursion, so a
    pathological run cannot loop. Runs with fewer than 3 points (no
    interior split point exists) are never split, matching ``_corner_split``
    and ``_fit_schneider_run``'s own 2-point base case.
    """
    if len(pts) < 3:
        return []

    extra_splits: list[int] = []

    def check(start: int, end: int, depth: int) -> None:
        sub = pts[start : end + 1]
        if len(sub) < 3:
            return
        aspect = _run_bbox_aspect(sub)
        edge_ratio = _run_edge_ratio(sub)
        if aspect <= SEGMENT_MAX_BBOX_ASPECT and edge_ratio <= SEGMENT_MAX_EDGE_RATIO:
            return
        if depth >= 4:  # bounded, deterministic -- never open-ended recursion
            return
        local_split = _homogeneity_split_point(sub)
        global_split = start + local_split
        if global_split <= start or global_split >= end:
            return  # would create an empty or one-point run; refuse the split
        extra_splits.append(global_split)
        check(start, global_split, depth + 1)
        check(global_split, end, depth + 1)

    check(0, len(pts) - 1, 0)
    return sorted(set(extra_splits))


def fit_arc(
    points: np.ndarray,
    *,
    tolerance_pt: float,
    corner_angle_deg: float = CORNER_ANGLE_DEG_DEFAULT,
    impl: str = "schneider",
) -> tuple[tuple[BezierSegment, ...], tuple[int, ...], float]:
    """Fit one arc polyline: (segments, corner joint indices, max error)."""
    fitter = _FITTERS.get(impl)
    if fitter is None:
        raise ConfigError(f"unknown curve fitter {impl!r}; choose from {sorted(_FITTERS)}")
    pts = np.asarray(points, dtype=np.float64)
    corner_cuts = [0, *_corner_split(pts, corner_angle_deg), len(pts) - 1]
    # Sprint 41: second segmentation pass -- for each corner-free run, insert
    # extra split points where the run itself is geometrically unsuitable
    # for Schneider fitting (bbox aspect ratio / edge-length ratio outliers,
    # Sprint 40), so `fitter` never sees a run that predicts ill-conditioning.
    # `fitter` (Schneider or otherwise) is called exactly as before; only the
    # set of run boundaries passed to it changes.
    cuts: list[int] = [corner_cuts[0]]
    for a, b in pairwise(corner_cuts):
        run = pts[a : b + 1]
        extra = homogeneity_split(run)
        cuts.extend(a + s for s in extra)
        cuts.append(b)
    segments: list[np.ndarray] = []
    corner_joints: list[int] = []
    worst = 0.0
    for a, b in pairwise(cuts):
        run_segments, err = fitter(pts[a : b + 1], tolerance_pt)
        # Junction/corner interpolation is bitwise: endpoints are the input
        # vertices themselves, never fitted values.
        run_segments[0][0] = pts[a]
        run_segments[-1][3] = pts[b]
        if segments:
            corner_joints.append(len(segments))
            run_segments[0][0] = segments[-1][3]  # share the corner bitwise
        segments.extend(run_segments)
        worst = max(worst, err)
    return (
        tuple(BezierSegment(control=s) for s in segments),
        tuple(corner_joints),
        worst,
    )


def fit_curves(
    arc_graph: ArcGraph,
    *,
    fit_error_mm: float = FIT_ERROR_MM_DEFAULT,
    corner_angle_deg: float = CORNER_ANGLE_DEG_DEFAULT,
    impl: str = "schneider",
    d_min_mm: float | None = None,
    filler_ids: frozenset[int] = frozenset(),
    config_hash: str = _UNSET_HASH,
) -> CurveSet:
    """Full §18: fit every arc independently, carry faces over unchanged.

    Pure per-arc transform: each arc's ``Curve`` depends only on that
    arc's own ``points``, the three config knobs (``fit_error_mm``,
    ``corner_angle_deg``, ``impl``), and (if ``d_min_mm`` is given) the
    read-only face-area lookup computed once up front from ``arc_graph``
    -- never on any other arc's *fitted output*, preserving the
    independence ``fidelity``/``printability`` (I1/I4) rely on for their
    re-proof to be meaningful (see ``docs/adr/002-sprint36-print-aware-
    geometry-simplification.md``'s accepted architecture review):
    narrow-feature self-intersection risk is handled upstream, before this
    stage runs, by the dedicated ``geometry_normalize`` stage's Minimum Gap
    Enforcement pass (``stages/vector/geometry_normalize.py``;
    GAP_REPAIR_DESIGN.md).

    ``d_min_mm``, when given, shrinks an arc's own fit tolerance when the
    smaller of its adjacent faces is below the printability area floor
    (``stages/vector/_face_area.py``): a fixed absolute residual is a
    negligible fraction of a large face's boundary pixel count but can be a
    large fraction of a tiny (``merge_tiny``-surviving, legally printable)
    face's, which was observed causing spurious ``fidelity`` FATALs on real
    photos with many small regions right at the floor.

    ``filler_ids`` (split_large's filler/rim cells) always get the smallest
    tolerance scale regardless of their own area: a rim cell in particular
    can be large in area yet a thin, winding strip tracing a subject's own
    detail edges (ears, fur) rather than a normal region's smoother boundary
    -- area-based scaling alone under-shrinks its tolerance, which was
    observed causing the same spurious fidelity FATALs on split_large output.
    """
    tolerance = fit_error_pt(fit_error_mm)

    scale_by_arc: dict[int, float] = {}
    if d_min_mm is not None:
        reference_area = tolerance_reference_area_pt2(d_min_mm)
        min_area_by_arc = min_adjacent_face_area_pt2_by_arc(arc_graph.arcs, arc_graph.faces)
        scale_by_arc = {
            arc_id: tolerance_scale_for_area(area, reference_area_pt2=reference_area)
            for arc_id, area in min_area_by_arc.items()
        }
    if filler_ids:
        # Same-color seams (both sides are filler cells of one label) can
        # deviate freely without touching fidelity — fit them LOOSE so the
        # bezier sweeps smoothly over the pixel staircase instead of tracing
        # it gear-tooth by gear-tooth (the "cog" look on mystery pages).
        # Filler arcs on a color boundary keep the tight scale as before.
        seam_arcs = same_label_seam_arc_ids(arc_graph.faces, frozenset(filler_ids))
        filler_arc_ids: set[int] = set()
        for face in arc_graph.faces:
            if face.face_id not in filler_ids:
                continue
            for walk in face.all_walks():
                for arc_id, _ in walk:
                    filler_arc_ids.add(arc_id)
        for arc_id in filler_arc_ids:
            if arc_id in seam_arcs:
                scale_by_arc[arc_id] = _SEAM_TOLERANCE_SCALE
            else:
                scale_by_arc[arc_id] = min(scale_by_arc.get(arc_id, 1.0), _FILLER_TOLERANCE_SCALE)

    curves = []
    for arc in arc_graph.arcs:
        is_seam = scale_by_arc.get(arc.arc_id, 1.0) > 1.0
        arc_tolerance = tolerance * scale_by_arc.get(arc.arc_id, 1.0)
        arc_corner_deg = _SEAM_CORNER_ANGLE_DEG if is_seam else corner_angle_deg
        segments, corners, err = fit_arc(
            arc.points,
            tolerance_pt=arc_tolerance,
            corner_angle_deg=arc_corner_deg,
            impl=impl,
        )
        curves.append(
            Curve(
                arc_id=arc.arc_id,
                segments=segments,
                corner_indices=corners,
                max_fit_error_pt=err,
            )
        )

    # Ring-level self-intersection repair: a fit can sweep wide enough that
    # a face's ring crosses itself — one arc looping, two arcs crossing, or
    # a loose seam brushing across a neighboring tight arc (the topology
    # validator's I3 check is per-ring, not per-arc). The deliberately-loose
    # seam fit (scale > 1) is the usual culprit, but at loose global
    # tolerances (dense preset) an ordinary arc can cross too, so EVERY face
    # is checked — not just seam-touching ones. Detection uses the
    # validator's OWN flatten + predicate (model/flatten.py) so nothing
    # borderline slips through. Escalation per affected face, re-checking
    # between passes:
    #   pass 1 — replace the face's loose seam arcs with their exact
    #            simplified polylines (degree-1 chains);
    #   pass 2 — if a ring still crosses, polyline-ify EVERY arc of that
    #            face: the resulting rings are geometrically the exact
    #            polyline rings, which the simplify stage's identical guard
    #            already verified crossing-free — a guaranteed fixpoint.
    seam_arc_ids_loose = {a.arc_id for a in arc_graph.arcs if scale_by_arc.get(a.arc_id, 1.0) > 1.0}
    if curves:
        max_id = max(a.arc_id for a in arc_graph.arcs)
        indexed: list[Curve | None] = [None] * (max_id + 1)
        for c in curves:
            indexed[c.arc_id] = c
        arcs_by_id = {a.arc_id: a for a in arc_graph.arcs}

        def _polylineify(arc_id: int) -> None:
            pts = np.asarray(arcs_by_id[arc_id].points, dtype=np.float64)
            segments = tuple(
                BezierSegment(control=_line_segment(pts[i], pts[i + 1]))
                for i in range(len(pts) - 1)
            )
            indexed[arc_id] = Curve(
                arc_id=arc_id, segments=segments, corner_indices=(), max_fit_error_pt=0.0
            )

        def _face_crosses(face) -> bool:
            rings = flatten_face_rings(face, indexed, _SELF_X_FLATTEN_PT)
            if any(ring_self_intersects(ring) for ring in rings):
                return True
            return any(
                rings_intersect(rings[i], rings[j])
                for i in range(len(rings))
                for j in range(i + 1, len(rings))
            )

        # Polyline-ifying an arc shared with an already-checked face can (in
        # principle) introduce a new crossing there, so the scan repeats
        # until a full pass makes no repair. Convergence is guaranteed: each
        # repair only ever converts arcs to their exact polylines (monotone),
        # and the all-polyline state is crossing-free by the simplify guard.
        # Passes are still capped: a dense page can have thousands of faces,
        # and re-scanning all of them every pass is O(faces) per pass, so an
        # unbounded loop trades a silent FATAL for an impractically slow one.
        # If it hasn't converged within the cap, the residual check below
        # raises a clear, early, diagnosable error instead.
        _MAX_REPAIR_PASSES = 16
        repaired = False
        for _ in range(_MAX_REPAIR_PASSES):
            changed = False
            for face in arc_graph.faces:
                if not _face_crosses(face):
                    continue
                changed = repaired = True
                walk_arc_ids = [aid for walk in face.all_walks() for aid, _ in walk]
                loose_in_walk = [aid for aid in walk_arc_ids if aid in seam_arc_ids_loose]
                for aid in loose_in_walk:
                    _polylineify(aid)
                if loose_in_walk and not _face_crosses(face):
                    continue
                for aid in walk_arc_ids:
                    _polylineify(aid)
            if not changed:
                break

        if repaired:
            curves = [indexed[c.arc_id] for c in curves]
            residual = [face for face in arc_graph.faces if _face_crosses(face)]
            if residual:
                raise AssertionError(
                    "curve fitting self-intersection repair did not converge "
                    f"for face id(s): {[face.face_id for face in residual]}"
                )

    return CurveSet(
        curves=tuple(curves),
        faces=arc_graph.faces,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=arc_graph.provenance.source_hash,
        ),
    )


class CurveFitStage:
    """Stage wrapper: ``arc_graph`` → ``curve_set``."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        d_min_mm: float | None = None,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        error = section.get("fit_error_mm", FIT_ERROR_MM_DEFAULT)
        angle = section.get("corner_angle_deg", CORNER_ANGLE_DEG_DEFAULT)
        impl = section.get("impl", "schneider")
        if not isinstance(error, (int, float)) or not 0.02 <= float(error) <= 2.0:
            raise ConfigError(f"bezier config: fit_error_mm must be in [0.02, 2.0], got {error!r}")
        if not isinstance(angle, (int, float)) or not 15.0 <= float(angle) <= 120.0:
            raise ConfigError(
                f"bezier config: corner_angle_deg must be in [15, 120], got {angle!r}"
            )
        if not isinstance(impl, str) or impl not in _FITTERS:
            raise ConfigError(f"bezier config: impl must be one of {sorted(_FITTERS)}")
        self._fit_error_mm = float(error)
        self._corner_angle_deg = float(angle)
        self._impl = impl
        self._d_min_mm = d_min_mm
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
        return ("curve_set",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        arc_graph = ctx.get("arc_graph")
        if not isinstance(arc_graph, ArcGraph):
            raise ConfigError("bezier requires an ArcGraph artifact")
        filler_ids = ctx.get("filler_region_ids") if ctx.has("filler_region_ids") else frozenset()
        if not isinstance(filler_ids, (set, frozenset)):
            filler_ids = frozenset()
        ctx.put(
            "curve_set",
            fit_curves(
                arc_graph,
                fit_error_mm=self._fit_error_mm,
                corner_angle_deg=self._corner_angle_deg,
                impl=self._impl,
                d_min_mm=self._d_min_mm,
                filler_ids=frozenset(filler_ids),
                config_hash=self._config_hash,
            ),
        )
