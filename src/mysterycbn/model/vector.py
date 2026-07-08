"""Concrete vector-domain artifacts (DATA_MODEL_SPEC.md §9–§14)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from mysterycbn.model._utils import readonly, require
from mysterycbn.model.records import Provenance

EXTERIOR_ID = -1


@dataclass(frozen=True)
class Arc:
    """One maximal boundary piece separating exactly two regions (DATA_MODEL_SPEC §10).

    ``points`` is (P, 2) float64 — doubled-crack px inside a TopologyGraph,
    points inside an ArcGraph (post-Φ). Closed arcs implicitly join last→first.
    """

    arc_id: int
    points: np.ndarray
    left_region: int
    right_region: int
    closed: bool = False

    def __post_init__(self) -> None:
        pts = readonly(self.points, np.float64)
        require(pts.ndim == 2 and pts.shape[1] == 2, f"points must be (P, 2), got {pts.shape}")
        minimum = 4 if self.closed else 2
        require(pts.shape[0] >= minimum, f"arc needs ≥ {minimum} points, got {pts.shape[0]}")
        require(self.arc_id >= 0, "arc_id must be ≥ 0")
        require(self.left_region != self.right_region, "left and right regions must differ")
        require(
            bool((np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.0).all()),
            "consecutive points must be distinct",
        )
        object.__setattr__(self, "points", pts)

    def to_dict(self) -> dict[str, object]:
        return {
            "arc_id": self.arc_id,
            "points": self.points.tolist(),
            "left_region": self.left_region,
            "right_region": self.right_region,
            "closed": self.closed,
        }


@dataclass(frozen=True)
class TopologyGraph:
    """Junction/arc decomposition of the crack boundary (DATA_MODEL_SPEC §9)."""

    junctions: np.ndarray
    arcs: tuple[Arc, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        junctions = readonly(self.junctions, np.int64)
        require(
            junctions.ndim == 2 and junctions.shape[1] == 2,
            f"junctions must be (V, 2), got {junctions.shape}",
        )
        require(
            tuple(a.arc_id for a in self.arcs) == tuple(range(len(self.arcs))),
            "arcs must be sorted by dense arc_id",
        )
        junction_set = {(int(u), int(v)) for u, v in junctions}
        for arc in self.arcs:
            if arc.closed:
                continue
            for end in (arc.points[0], arc.points[-1]):
                key = (int(end[0]), int(end[1]))
                require(
                    key in junction_set,
                    f"arc {arc.arc_id} endpoint {key} is not a junction",
                )
        object.__setattr__(self, "junctions", junctions)

    def to_dict(self) -> dict[str, object]:
        return {
            "junctions": self.junctions.tolist(),
            "arcs": [a.to_dict() for a in self.arcs],
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class Face:
    """A region as ordered arc walks: outer ring plus holes (DATA_MODEL_SPEC §11).

    Walk entries are ``(arc_id, reversed)``.
    """

    face_id: int
    label: int
    outer_walk: tuple[tuple[int, bool], ...]
    hole_walks: tuple[tuple[tuple[int, bool], ...], ...] = ()

    def __post_init__(self) -> None:
        require(self.face_id >= 0 and self.label >= 0, "ids must be ≥ 0")
        require(len(self.outer_walk) >= 1, "outer_walk must be non-empty")
        for walk in (self.outer_walk, *self.hole_walks):
            require(len(walk) >= 1, "walks must be non-empty")

    def all_walks(self) -> tuple[tuple[tuple[int, bool], ...], ...]:
        """Outer walk followed by hole walks."""
        return (self.outer_walk, *self.hole_walks)

    def to_dict(self) -> dict[str, object]:
        return {
            "face_id": self.face_id,
            "label": self.label,
            "outer_walk": [list(e) for e in self.outer_walk],
            "hole_walks": [[list(e) for e in w] for w in self.hole_walks],
        }


def _check_arc_references(faces: tuple[Face, ...], arc_count: int, what: str) -> None:
    """Every referenced arc exists and is used by 1 or 2 face-walk sides.

    (Arcs bordering the exterior appear once — the exterior face is not stored.)
    """
    counts: Counter[int] = Counter()
    for face in faces:
        for walk in face.all_walks():
            for arc_id, _ in walk:
                require(0 <= arc_id < arc_count, f"{what}: walk references unknown arc {arc_id}")
                counts[arc_id] += 1
    for arc_id, n in counts.items():
        require(n <= 2, f"{what}: arc {arc_id} referenced by {n} walk sides (max 2)")


@dataclass(frozen=True)
class ArcGraph:
    """The planar map in physical units (DATA_MODEL_SPEC §11)."""

    arcs: tuple[Arc, ...]
    faces: tuple[Face, ...]
    work_scale: float
    provenance: Provenance

    def __post_init__(self) -> None:
        require(self.work_scale > 0.0, "work_scale must be positive (Φ already applied)")
        require(
            tuple(a.arc_id for a in self.arcs) == tuple(range(len(self.arcs))),
            "arcs must be sorted by dense arc_id",
        )
        require(
            tuple(f.face_id for f in self.faces) == tuple(range(len(self.faces))),
            "faces must be sorted by dense face_id",
        )
        _check_arc_references(self.faces, len(self.arcs), "ArcGraph")

    def to_dict(self) -> dict[str, object]:
        return {
            "arcs": [a.to_dict() for a in self.arcs],
            "faces": [f.to_dict() for f in self.faces],
            "work_scale": self.work_scale,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class BezierSegment:
    """One cubic Bézier piece, control points b₀…b₃ in pt (DATA_MODEL_SPEC §12)."""

    control: np.ndarray

    def __post_init__(self) -> None:
        ctrl = readonly(self.control, np.float64)
        require(ctrl.shape == (4, 2), f"control must be (4, 2), got {ctrl.shape}")
        require(bool(np.isfinite(ctrl).all()), "control points must be finite")
        require(
            not (np.array_equal(ctrl[0], ctrl[1]) and np.array_equal(ctrl[2], ctrl[3])),
            "doubly-degenerate segment (b0=b1 and b2=b3) is not allowed",
        )
        object.__setattr__(self, "control", ctrl)

    def to_dict(self) -> dict[str, object]:
        return {"control": self.control.tolist()}


@dataclass(frozen=True)
class Curve:
    """The fitted Bézier chain for one arc (DATA_MODEL_SPEC §13)."""

    arc_id: int
    segments: tuple[BezierSegment, ...]
    corner_indices: tuple[int, ...]
    max_fit_error_pt: float

    def __post_init__(self) -> None:
        require(self.arc_id >= 0, "arc_id must be ≥ 0")
        require(len(self.segments) >= 1, "curve needs ≥ 1 segment")
        require(self.max_fit_error_pt >= 0.0, "max_fit_error_pt must be ≥ 0")
        for a, b in zip(self.segments[:-1], self.segments[1:], strict=True):
            require(
                np.array_equal(a.control[3], b.control[0]),
                "consecutive segments must share endpoints exactly (bitwise)",
            )
        n_joints = len(self.segments) - 1
        require(
            all(1 <= i <= n_joints for i in self.corner_indices),
            "corner_indices must reference interior segment joints",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arc_id": self.arc_id,
            "segments": [s.to_dict() for s in self.segments],
            "corner_indices": list(self.corner_indices),
            "max_fit_error_pt": self.max_fit_error_pt,
        }


@dataclass(frozen=True)
class CurveSet:
    """Final vector geometry: all curves + carried-over faces (DATA_MODEL_SPEC §14)."""

    curves: tuple[Curve, ...]
    faces: tuple[Face, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        require(
            tuple(c.arc_id for c in self.curves) == tuple(range(len(self.curves))),
            "curves must be sorted by dense arc_id",
        )
        _check_arc_references(self.faces, len(self.curves), "CurveSet")

    def to_dict(self) -> dict[str, object]:
        return {
            "curves": [c.to_dict() for c in self.curves],
            "faces": [f.to_dict() for f in self.faces],
            "provenance": self.provenance.to_dict(),
        }
