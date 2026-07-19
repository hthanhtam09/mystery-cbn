"""Label Placement stage: one readable number per face
(ENGINE_SPEC.md §19; math MATH_SPEC §13–§14; Label/LabelPlan DATA_MODEL §15).

Pipeline per face, all in pt on flattened Bézier boundaries (0.1 mm
tolerance):

1. **Pole of inaccessibility / largest empty circle** — quadtree
   branch-and-bound maximization of the face's signed distance function
   (holes respected via even-odd containment over all rings), precision
   ``polylabel_precision_pt``. Yields anchor ``c*`` and clearance ``r*``
   (the largest empty circle's center and radius; 2r* is the printability
   diameter).
2. **Font scaling** — a string of ``n`` digits at size S occupies
   ``(n·ω_f·S) × (κ_f·S)`` (DejaVu Sans metrics ω_f = 0.636, κ_f = 0.729);
   inscribing that box in the clearance circle gives
   ``S_fit = 2r*/√((n·ω_f)² + κ_f²)`` (the 1.35·r* closed form for n = 2),
   clipped to ``[font_min, font_max]``. ``S_fit < font_min`` → leader.
3. **Leader lines** — 16 candidates on the ring ``r* + ρ`` (ρ = 4 mm)
   around the pole at fixed π/8 steps; feasible iff the bbox at
   ``font_min`` has clearance > its own half-diagonal from *all* page
   geometry and the segment candidate→pole crosses < 3 arcs. Choose
   minimal (crossings, angle index) — deterministic. No feasible
   candidate → FATAL Finding (invariant I4; the validator decides).
4. **Collision avoidance** — greedy by descending clearance: a
   conflicting label is displaced along ∇d (central differences on the
   face distance) by up to r*/2 in fixed fractions, accepting only
   positions where its bbox still fits inside the face; still conflicting
   → demoted to leader.

Printed numbers are ``label + 1`` (1-based); the §20 palette permutation
rewrites them downstream.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from mysterycbn.foundation.codes import code_for_number
from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.layout import Label, LabelMode, LabelPlan
from mysterycbn.model.records import Provenance, RegionGraph
from mysterycbn.model.reports import Finding, Severity
from mysterycbn.model.vector import CurveSet

STAGE_NAME = "labels"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64


@dataclass(frozen=True)
class LabelFindings:
    """Context-transportable wrapper for the stage's FATAL findings."""

    findings: tuple[Finding, ...]
    provenance: Provenance


FONT_MIN_PT_DEFAULT = 6.0
FONT_MAX_PT_DEFAULT = 14.0
_MICRO_FONT_MIN_PT = 1.0  # smallest stamped number for a filler cell (dense mode)
# Absolute legibility floor for any printed micro-label: below this a number
# is unreadable ink, so a sliver's label is allowed to overrun its hairline
# boundary stroke rather than shrink past it (a slightly-overlapping number
# can still be colored by; an invisible one cannot).
_MICRO_FONT_READABLE_PT = 5.0
_STROKE_HALF_PT = 0.16  # half the 0.3 pt region stroke, rounded up
# Safety margin subtracted from the clearance radius before sizing a label:
# half the 0.3 pt region stroke plus visible daylight, so a glyph box is
# never tangent to (or across) the printed line.
_CLEARANCE_PAD_PT = 0.5
POLYLABEL_PRECISION_PT_DEFAULT = 0.5
LEADER_RING_MM_DEFAULT = 4.0
_OMEGA_F = 0.636  # DejaVu Sans digit advance / em (MATH_SPEC §14.1, pinned)
_KAPPA_F = 0.729  # DejaVu Sans cap height / em
_FLATTEN_MM = 0.1
_LEADER_ANGLES = 16
_MAX_LEADER_CROSSINGS = 2  # "< 3 arcs" — normative constant, not a knob
_DISPLACE_STEPS = (0.25, 0.5, 0.75, 1.0)  # fractions of the r*/2 budget


def _mm_to_pt(mm: float) -> float:
    return mm * PT_PER_INCH / MM_PER_INCH


# ------------------------------------------------------------ flattening ---


def _flatten_segment(ctrl: np.ndarray, tolerance: float) -> np.ndarray:
    """Sample one cubic at a chord-proportional density (≤ tolerance sag)."""
    chord = float(
        np.linalg.norm(ctrl[3] - ctrl[0])
        + np.linalg.norm(ctrl[1] - ctrl[0])
        + np.linalg.norm(ctrl[2] - ctrl[1])
        + np.linalg.norm(ctrl[3] - ctrl[2])
    )
    n = int(np.clip(math.ceil(chord / (4.0 * tolerance)), 2, 24))
    u = np.linspace(0.0, 1.0, n + 1)
    b = np.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u**2 * (1 - u), u**3], axis=1)
    return np.asarray(b @ ctrl)


def _flatten_walk(
    walk: tuple[tuple[int, bool], ...], curve_set: CurveSet, tolerance: float
) -> np.ndarray:
    """One closed ring (last point dropped) for a face walk."""
    parts = []
    for arc_id, rev in walk:
        for segment in (
            curve_set.curves[arc_id].segments
            if not rev
            else reversed(curve_set.curves[arc_id].segments)
        ):
            pts = _flatten_segment(segment.control, tolerance)
            parts.append(pts[:-1] if not rev else pts[::-1][:-1])
    return np.concatenate(parts)


def _segments_of(rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(seg_a, seg_b) arrays over all rings, closed."""
    a = np.concatenate([r for r in rings])
    b = np.concatenate([np.roll(r, -1, axis=0) for r in rings])
    return a, b


# --------------------------------------- pole / largest empty circle -------


def _signed_distances(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Distances of (N, 2) probes to the face boundary, positive inside
    (even-odd over rings); batched over probes."""
    ab = seg_b - seg_a  # (S, 2)
    denom = np.einsum("ij,ij->i", ab, ab)  # (S,)
    ap = points[:, None, :] - seg_a[None, :, :]  # (N, S, 2)
    tt = np.clip(
        np.divide(
            np.einsum("nsj,sj->ns", ap, ab),
            denom[None, :],
            out=np.zeros((points.shape[0], seg_a.shape[0])),
            where=denom[None, :] > 0,
        ),
        0.0,
        1.0,
    )
    d = np.linalg.norm(ap - tt[..., None] * ab[None, :, :], axis=2).min(axis=1)  # (N,)
    y = points[:, 1][:, None]
    x = points[:, 0][:, None]
    cond = (seg_a[None, :, 1] > y) != (seg_b[None, :, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = seg_a[None, :, 0] + (y - seg_a[None, :, 1]) * (ab[None, :, 0] / ab[None, :, 1])
    inside = (np.count_nonzero(cond & (x < xi), axis=1) % 2).astype(bool)
    return np.asarray(np.where(inside, d, -d))


def _signed_distance(x: float, y: float, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """Single-probe form of :func:`_signed_distances`."""
    return float(_signed_distances(np.array([[x, y]]), seg_a, seg_b)[0])


def largest_empty_circle(
    rings: list[np.ndarray], precision_pt: float
) -> tuple[tuple[float, float], float]:
    """Pole of inaccessibility + clearance radius of a face with holes
    (quadtree branch-and-bound, MATH_SPEC §13.3). Returns ((x, y), r*)."""
    seg_a, seg_b = _segments_of(rings)
    outer = rings[0]
    lo, hi = outer.min(axis=0), outer.max(axis=0)
    half = float(max(hi[0] - lo[0], hi[1] - lo[1])) / 2.0
    cx0, cy0 = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
    best_d = _signed_distance(cx0, cy0, seg_a, seg_b)
    best = (cx0, cy0)
    root = math.sqrt(2.0)
    heap: list[tuple[float, float, float, float, float]] = [
        (-(best_d + half * root), cx0, cy0, best_d, half)
    ]
    while heap:
        neg_u, cx, cy, d, hh = heapq.heappop(heap)
        if d > best_d:
            best_d, best = d, (cx, cy)
        if -neg_u - best_d <= precision_pt:
            continue  # prune: cannot beat best by more than the precision
        q = hh / 2.0
        # Batch the 4 children; heap key (−U, cx, cy) — lex tie-break (§13.3).
        children = np.array(
            [[cx - q, cy - q], [cx - q, cy + q], [cx + q, cy - q], [cx + q, cy + q]]
        )
        for (ccx, ccy), cd in zip(children, _signed_distances(children, seg_a, seg_b), strict=True):
            heapq.heappush(heap, (-(float(cd) + q * root), float(ccx), float(ccy), float(cd), q))
    return best, max(best_d, 0.0)


# ------------------------------------------------------------ font model ---


def text_bbox_pt(number: int, size_pt: float) -> tuple[float, float]:
    """(width, height) of the printed one-char code at ``size_pt`` (MATH_SPEC §14.1)."""
    digits = len(code_for_number(number))
    return digits * _OMEGA_F * size_pt, _KAPPA_F * size_pt


def fitted_font_size(number: int, clearance_pt: float) -> float:
    """Largest size whose bbox inscribes in the clearance circle shrunk by
    the safety pad: ``S_fit = 2(r* − pad)/√((n·ω_f)² + κ_f²)``. Without the
    pad the inscribed box is tangent to the circle — and the circle tangent
    to the region boundary — so glyph corners land exactly on the printed
    line."""
    digits = len(code_for_number(number))
    usable = max(clearance_pt - _CLEARANCE_PAD_PT, 0.0)
    return 2.0 * usable / math.hypot(digits * _OMEGA_F, _KAPPA_F)


def _micro_font_size(number: int, clearance_pt: float, fit: float) -> float:
    """Size for a stamped micro-label: raised toward the readability floor,
    but never past the hard cap where the glyph box would enter the printed
    stroke's ink (clearance minus half the 0.3 pt line weight) — except that
    no label is ever printed below ``_MICRO_FONT_READABLE_PT``. A degenerate
    sliver thus gets a readable number that may overrun its hairline
    boundary, never an invisible one."""
    digits = len(code_for_number(number))
    ink_free = max(clearance_pt - _STROKE_HALF_PT, 0.0)
    hard_cap = 2.0 * ink_free / math.hypot(digits * _OMEGA_F, _KAPPA_F)
    return max(min(max(fit, _MICRO_FONT_MIN_PT), hard_cap), _MICRO_FONT_READABLE_PT)


def _bbox_rect(
    anchor: tuple[float, float], number: int, size_pt: float
) -> tuple[float, float, float, float]:
    w, h = text_bbox_pt(number, size_pt)
    return (anchor[0] - w / 2, anchor[1] - h / 2, anchor[0] + w / 2, anchor[1] + h / 2)


def _rects_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------- leader lines ---


def _segment_crossings(p: np.ndarray, q: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> int:
    """Number of boundary segments properly crossed by segment p→q."""
    d = q - p
    e = seg_b - seg_a
    w = seg_a - p
    denom = d[0] * e[:, 1] - d[1] * e[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / denom
        s = (w[:, 0] * d[1] - w[:, 1] * d[0]) / denom
    hits = (denom != 0) & (t > 0.0) & (t < 1.0) & (s > 0.0) & (s < 1.0)
    return int(np.count_nonzero(hits))


def _min_distance(p: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    ab = seg_b - seg_a
    ap = p - seg_a
    denom = np.einsum("ij,ij->i", ab, ab)
    tt = np.clip(
        np.divide(np.einsum("ij,ij->i", ap, ab), denom, out=np.zeros_like(denom), where=denom > 0),
        0.0,
        1.0,
    )
    return float(np.min(np.linalg.norm(ap - tt[:, None] * ab, axis=1)))


def _place_leader(
    pole: tuple[float, float],
    clearance: float,
    number: int,
    font_min: float,
    ring_pt: float,
    all_a: np.ndarray,
    all_b: np.ndarray,
    kept_rects: list[tuple[float, float, float, float]] = (),
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]] | None:
    """Best leader anchor on the ring, or None (MATH_SPEC §14.2).

    Candidates whose bbox overlaps an already-kept label are skipped, so a
    leader label that collides during collision avoidance can fall back to
    its next-best ring position instead of only being retried unchanged.
    """
    w, h = text_bbox_pt(number, font_min)
    half_diag = math.hypot(w, h) / 2.0
    radius = clearance + ring_pt
    c = np.asarray(pole)
    candidates = []
    for k in range(_LEADER_ANGLES):
        angle = k * math.pi / (_LEADER_ANGLES / 2.0)
        q = c + radius * np.array([math.cos(angle), math.sin(angle)])
        if _min_distance(q, all_a, all_b) <= half_diag + _CLEARANCE_PAD_PT:
            continue  # not in whitespace
        crossings = _segment_crossings(q, c, all_a, all_b)
        if crossings > _MAX_LEADER_CROSSINGS:
            continue
        anchor = (float(q[0]), float(q[1]))
        rect = _bbox_rect(anchor, number, font_min)
        if any(_rects_overlap(rect, other) for other in kept_rects):
            continue
        candidates.append((crossings, k, q))
    if not candidates:
        return None
    _, _, q = min(candidates, key=lambda item: (item[0], item[1]))
    anchor = (float(q[0]), float(q[1]))
    return anchor, (anchor, pole)


# ---------------------------------------------------------------- stage ----


class _FaceGeometry:
    """Flattened rings + boundary segment arrays of one face."""

    def __init__(self, rings: list[np.ndarray]) -> None:
        self.rings = rings
        self.seg_a, self.seg_b = _segments_of(rings)

    def distance(self, x: float, y: float) -> float:
        return _signed_distance(x, y, self.seg_a, self.seg_b)


def _displace(
    label: Label,
    geometry: _FaceGeometry,
    kept_rects: list[tuple[float, float, float, float]],
) -> Label | None:
    """Slide a conflicting label uphill along ∇d by ≤ r*/2 (MATH_SPEC §14.3)."""
    x, y = label.anchor
    eps = 0.5
    gx = geometry.distance(x + eps, y) - geometry.distance(x - eps, y)
    gy = geometry.distance(x, y + eps) - geometry.distance(x, y - eps)
    norm = math.hypot(gx, gy)
    if norm == 0.0:
        return None
    w, h = text_bbox_pt(label.printed_number, label.font_size_pt)
    half_diag = math.hypot(w, h) / 2.0
    for frac in _DISPLACE_STEPS:
        step = frac * label.clearance_pt / 2.0
        anchor = (x + gx / norm * step, y + gy / norm * step)
        rect = _bbox_rect(anchor, label.printed_number, label.font_size_pt)
        if any(_rects_overlap(rect, kept) for kept in kept_rects):
            continue
        if geometry.distance(*anchor) < half_diag + _CLEARANCE_PAD_PT:
            continue  # bbox would cross (or touch) the region boundary
        return Label(
            region_id=label.region_id,
            printed_number=label.printed_number,
            anchor=anchor,
            font_size_pt=label.font_size_pt,
            mode=LabelMode.IN_REGION,
            clearance_pt=label.clearance_pt,
        )
    return None


def place_labels(
    curve_set: CurveSet,
    region_graph: RegionGraph,
    *,
    font_min_pt: float = FONT_MIN_PT_DEFAULT,
    font_max_pt: float = FONT_MAX_PT_DEFAULT,
    polylabel_precision_pt: float = POLYLABEL_PRECISION_PT_DEFAULT,
    leader_ring_mm: float = LEADER_RING_MM_DEFAULT,
    filler_ids: frozenset[int] = frozenset(),
    micro_fallback: bool = False,
    config_hash: str = _UNSET_HASH,
) -> tuple[LabelPlan, tuple[Finding, ...]]:
    """Full §19 placement. Returns (plan, findings); FATAL findings mark
    faces with no feasible leader anchor (the validator decides abort).

    ``filler_ids`` marks faces produced by ``split_large`` in dense mode:
    these carry a tiny in-region number stamped at the pole (clamped to a
    small floor, never a leader, never dropped), matching the commercial
    background-tiling look; they are exempt from the readable-font gate both
    here and in the printability validator."""
    if len(curve_set.faces) != len(region_graph.regions):
        raise ConfigError("curve_set faces and region_graph regions are different generations")
    tolerance = _mm_to_pt(_FLATTEN_MM)
    ring_pt = _mm_to_pt(leader_ring_mm)
    faces = curve_set.faces
    geometries: list[_FaceGeometry] = []
    for face in faces:
        rings = [_flatten_walk(walk, curve_set, tolerance) for walk in face.all_walks()]
        geometries.append(_FaceGeometry(rings))
    all_a = np.concatenate([g.seg_a for g in geometries])
    all_b = np.concatenate([g.seg_b for g in geometries])

    # Anchor + font size per face (leader when the fit misses the floor).
    proposed: list[Label] = []
    findings: list[Finding] = []
    for face, geometry in zip(faces, geometries, strict=True):
        number = face.label + 1  # 1-based; §20 permutation rewrites downstream
        pole, clearance = largest_empty_circle(geometry.rings, polylabel_precision_pt)
        fit = fitted_font_size(number, clearance)
        if fit >= font_min_pt:
            proposed.append(
                Label(
                    region_id=face.face_id,
                    printed_number=number,
                    anchor=pole,
                    font_size_pt=min(fit, font_max_pt),
                    mode=LabelMode.IN_REGION,
                    clearance_pt=clearance,
                )
            )
            continue
        if face.face_id in filler_ids:
            # Micro-label: stamp the number in-cell at whatever size fits (down
            # to a small floor), no leader, never dropped. The printability
            # validator exempts filler faces from the readable-font gate.
            proposed.append(
                Label(
                    region_id=face.face_id,
                    printed_number=number,
                    anchor=pole,
                    font_size_pt=_micro_font_size(number, clearance, fit),
                    mode=LabelMode.IN_REGION,
                    clearance_pt=clearance,
                )
            )
            continue
        if micro_fallback:
            # Dense mode: never place leaders. A leader anchor lands in the
            # interior of a *neighboring* region (the whitespace test only
            # keeps it clear of boundary lines), so that region would show
            # two printed numbers. Stamp a micro-label at the pole inside
            # the region itself instead -- its sub-font size is what the
            # printability validator keys on to exempt it (a micro-label is
            # any IN_REGION label below the readable font floor).
            proposed.append(
                Label(
                    region_id=face.face_id,
                    printed_number=number,
                    anchor=pole,
                    font_size_pt=_micro_font_size(number, clearance, fit),
                    mode=LabelMode.IN_REGION,
                    clearance_pt=clearance,
                )
            )
            continue
        placed = _place_leader(pole, clearance, number, font_min_pt, ring_pt, all_a, all_b)
        if placed is None:
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I4",
                    message="no feasible leader anchor",
                    location=f"region {face.face_id}",
                )
            )
            continue
        anchor, leader = placed
        proposed.append(
            Label(
                region_id=face.face_id,
                printed_number=number,
                anchor=anchor,
                font_size_pt=font_min_pt,
                mode=LabelMode.LEADER,
                clearance_pt=clearance,
                leader=leader,
            )
        )

    # Collision avoidance: greedy by descending clearance (MATH_SPEC §14.3).
    order = sorted(proposed, key=lambda lb: (-lb.clearance_pt, lb.region_id))
    kept: list[Label] = []
    kept_rects: list[tuple[float, float, float, float]] = []
    for label in order:
        rect = _bbox_rect(label.anchor, label.printed_number, label.font_size_pt)
        is_micro = label.mode is LabelMode.IN_REGION and label.font_size_pt < font_min_pt
        if label.region_id in filler_ids or is_micro:
            # Micro-labels (filler cells, plus dense-mode sub-font stamps) are
            # placed one per small cell at its own pole; keep them
            # unconditionally and exclude from collision resolution -- their
            # tiny boxes are expected to sit close together like a commercial
            # background fill, and dropping/displacing them leaves cells
            # unnumbered.
            kept.append(label)
            continue
        if not any(_rects_overlap(rect, other) for other in kept_rects):
            kept.append(label)
            kept_rects.append(rect)
            continue
        moved = None
        if label.mode is LabelMode.IN_REGION:
            moved = _displace(label, geometries[label.region_id], kept_rects)
        if moved is None and micro_fallback:
            # Dense mode: no leader demotion (a leader anchor sits inside a
            # neighboring region, giving that region two printed numbers).
            # Keep the label at its original in-region anchor; overlap of two
            # small stamped numbers is a lesser defect than a number printed
            # in the wrong region or a cell left unnumbered.
            kept.append(label)
            continue
        if moved is None:  # demote to (or re-place) leader
            pole = label.leader[1] if label.mode is LabelMode.LEADER else label.anchor
            clearance = label.clearance_pt
            placed = _place_leader(
                pole,
                clearance,
                label.printed_number,
                font_min_pt,
                ring_pt,
                all_a,
                all_b,
                kept_rects,
            )
            if placed is not None:
                anchor, leader = placed
                moved = Label(
                    region_id=label.region_id,
                    printed_number=label.printed_number,
                    anchor=anchor,
                    font_size_pt=font_min_pt,
                    mode=LabelMode.LEADER,
                    clearance_pt=clearance,
                    leader=leader,
                )
        if moved is None:
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I4",
                    message="label conflicts with kept labels and has no leader fallback",
                    location=f"region {label.region_id}",
                )
            )
            continue
        rect = _bbox_rect(moved.anchor, moved.printed_number, moved.font_size_pt)
        if any(_rects_overlap(rect, other) for other in kept_rects):
            if micro_fallback:
                kept.append(moved)
                continue
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I4",
                    message="displaced label still overlaps",
                    location=f"region {label.region_id}",
                )
            )
            continue
        kept.append(moved)
        kept_rects.append(rect)

    plan = LabelPlan(
        labels=tuple(sorted(kept, key=lambda lb: lb.region_id)),
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=curve_set.provenance.source_hash,
        ),
    )
    return plan, tuple(findings)


class LabelPlacementStage:
    """Stage wrapper: (``curve_set``, ``region_graph``) → ``label_plan``."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        font_min_pt: float = FONT_MIN_PT_DEFAULT,
        font_max_pt: float = FONT_MAX_PT_DEFAULT,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        precision = section.get("polylabel_precision_pt", POLYLABEL_PRECISION_PT_DEFAULT)
        ring = section.get("leader_ring_mm", LEADER_RING_MM_DEFAULT)
        if not isinstance(precision, (int, float)) or not 0.05 <= float(precision) <= 2.0:
            raise ConfigError(
                f"labels config: polylabel_precision_pt must be in [0.05, 2], got {precision!r}"
            )
        if not isinstance(ring, (int, float)) or not 1.0 <= float(ring) <= 20.0:
            raise ConfigError(f"labels config: leader_ring_mm must be in [1, 20], got {ring!r}")
        if font_min_pt <= 0 or font_max_pt < font_min_pt:
            raise ConfigError("labels config: need 0 < font_min_pt ≤ font_max_pt")
        self._precision = float(precision)
        self._ring_mm = float(ring)
        self._font_min = font_min_pt
        self._font_max = font_max_pt
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("curve_set", "region_graph")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("label_plan", "label_findings")

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        curve_set = ctx.get("curve_set")
        region_graph = ctx.get("region_graph")
        if not isinstance(curve_set, CurveSet) or not isinstance(region_graph, RegionGraph):
            raise ConfigError("labels requires CurveSet + RegionGraph artifacts")
        filler_ids = ctx.get("filler_region_ids") if ctx.has("filler_region_ids") else frozenset()
        if not isinstance(filler_ids, (set, frozenset)):
            filler_ids = frozenset()
        plan, findings = place_labels(
            curve_set,
            region_graph,
            font_min_pt=self._font_min,
            font_max_pt=self._font_max,
            polylabel_precision_pt=self._precision,
            leader_ring_mm=self._ring_mm,
            filler_ids=frozenset(filler_ids),
            # Dense mode (any filler cells present) also micro-labels small
            # non-split regions with no feasible leader instead of dropping
            # them, so every cell on a dense sheet carries a number.
            micro_fallback=bool(filler_ids),
            config_hash=self._config_hash,
        )
        ctx.put("label_plan", plan)
        ctx.put("label_findings", LabelFindings(findings=findings, provenance=plan.provenance))
        # In dense mode, republish the exempt set as filler_ids ∪ (faces that
        # received a micro-label via the leader-failure fallback), so the
        # printability validator exempts exactly the faces we deliberately
        # micro-labelled -- never inferring the exemption from font size alone.
        if filler_ids:
            micro = {
                lb.region_id
                for lb in plan.labels
                if lb.mode is LabelMode.IN_REGION and lb.font_size_pt < self._font_min
            }
            ctx.put("filler_region_ids", frozenset(filler_ids) | micro)
