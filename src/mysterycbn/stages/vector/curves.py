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

STAGE_NAME = "bezier"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

FIT_ERROR_MM_DEFAULT = 0.15  # core CurveConfig defaults
CORNER_ANGLE_DEG_DEFAULT = 65.0
_MAX_REPARAM = 4
_MAX_DEPTH = 32
_CHUNK = 8  # uniform chunk size of the plain least-squares fitter

_Run = np.ndarray  # (P, 2) float64 vertices of a corner-free run
_Fitter = Callable[[_Run, float], tuple[list[np.ndarray], float]]


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


# ------------------------------------------------------------- schneider ---


def _generate_bezier(
    pts: _Run, u: np.ndarray, t_left: np.ndarray, t_right: np.ndarray
) -> np.ndarray:
    """Least-squares cubic with fixed end tangent directions (Graphics Gems)."""
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
    alpha_l = (x0 * c11 - x1 * c01) / det if det else 0.0
    alpha_r = (c00 * x1 - c01 * x0) / det if det else 0.0
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    if alpha_l < 1e-6 * chord or alpha_r < 1e-6 * chord:  # Wu/Barsky fallback
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
        if err <= tolerance:
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
    cuts = [0, *_corner_split(pts, corner_angle_deg), len(pts) - 1]
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
    config_hash: str = _UNSET_HASH,
) -> CurveSet:
    """Full §18: fit every arc, carry faces over unchanged."""
    tolerance = fit_error_pt(fit_error_mm)
    curves = []
    for arc in arc_graph.arcs:
        segments, corners, err = fit_arc(
            arc.points,
            tolerance_pt=tolerance,
            corner_angle_deg=corner_angle_deg,
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
        ctx.put(
            "curve_set",
            fit_curves(
                arc_graph,
                fit_error_mm=self._fit_error_mm,
                corner_angle_deg=self._corner_angle_deg,
                impl=self._impl,
                config_hash=self._config_hash,
            ),
        )
