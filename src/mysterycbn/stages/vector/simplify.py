"""Simplify stage: topology-preserving polyline simplification per shared
arc (ENGINE_SPEC.md §16-17, ARCHITECTURE.md §15 "simplify" row; Sprint 19
orchestration gap).

This is a thin Stage wrapper around ``DefaultGeometryKernel.simplify_polyline``
(Visvalingam-Whyatt, ``foundation/geometry/default.py``) -- the algorithm
already exists and is fully implemented and unit-tested; this module only
wires it into the ``ArcGraph -> ArcGraph`` pipeline slot the architecture
dossier describes, since no stage previously called it
(confirmed absent: ``grep -rn "simplify_polyline" src/mysterycbn/stages/``
returned nothing before this file).

Each arc is simplified independently as an open polyline (an arc is never
literally closed -- see ``Arc.closed`` is a *ring* concept scoped to a
single-region island, and even then the kernel's own endpoint-pinning keeps
the anchor fixed). Endpoints are always pinned by
``DefaultGeometryKernel.simplify_polyline`` itself, so two arcs sharing a
junction continue to share the identical junction point after simplification
-- the kernel performs no rounding or endpoint adjustment, only interior
vertex removal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.geometry.default import DefaultGeometryKernel
from mysterycbn.foundation.geometry.primitives import PolylineData
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Provenance
from mysterycbn.model.flatten import rings_intersect
from mysterycbn.model.vector import Arc, ArcGraph
from mysterycbn.stages.vector._face_area import (
    area_floor_pt2,
    min_adjacent_face_area_pt2_by_arc,
    points_self_intersect,
    same_label_seam_arc_ids,
    tolerance_scale_for_area,
)

STAGE_NAME = "simplify"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

TOLERANCE_MM_DEFAULT = 0.15
_MM_TO_PT = 72.0 / 25.4
# See stages/vector/curves.py's identical constants for rationale.
_FILLER_TOLERANCE_SCALE = 0.3
_SEAM_TOLERANCE_SCALE = 2.0

_KERNEL = DefaultGeometryKernel()


def _repair_crossing_walks(
    arc_graph: ArcGraph,
    simplified_by_id: dict[int, Arc],
    loose_ids: set[int],
    simplify_one: Callable[[Arc, float], Arc],
) -> None:
    """Ring-level intersection repair (in place on ``simplified_by_id``).

    Simplification can make a face's ring cross itself, or two rings of one
    face (outer boundary vs a hole) cross each other — either is a topology
    I3 FATAL that no downstream stage can undo (the bezier fit follows the
    crossed polyline). The aggressive (scale > 1) seam scale is the usual
    culprit, but at loose global tolerances (dense preset) an ordinary arc
    can cross too, so EVERY face is checked — both intersection kinds, same
    as the validator. Escalation per crossing face, re-checking between
    passes (mirrors fit_curves' identical pass):
      pass 1 — re-simplify the crossing walks' loose seam arcs at the tight
               scale;
      pass 2 — re-simplify every arc of the crossing walks at the tight
               scale;
      pass 3 — restore those arcs to their exact input polylines (identity:
               the arc graph's planar partition is crossing-free by
               construction — a guaranteed fixpoint).
    """
    arcs_by_id = {a.arc_id: a for a in arc_graph.arcs}

    def _ring(walk: tuple[tuple[int, bool], ...]) -> np.ndarray:
        parts = []
        for i, (aid, rev) in enumerate(walk):
            pts = simplified_by_id[aid].points
            if rev:
                pts = pts[::-1]
            parts.append(pts if i == 0 else pts[1:])
        return np.concatenate(parts)

    def _crossing_arc_ids(face) -> list[int]:  # type: ignore[no-untyped-def]
        """Arc ids of every walk involved in a self- or pair-intersection."""
        walks = list(face.all_walks())
        rings = [_ring(walk) for walk in walks]
        bad = [points_self_intersect(ring) for ring in rings]
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                if rings_intersect(rings[i], rings[j]):
                    bad[i] = bad[j] = True
        return [aid for walk, is_bad in zip(walks, bad, strict=True) if is_bad for aid, _ in walk]

    # Re-simplifying an arc shared with an already-checked face can (in
    # principle) introduce a new crossing there, so the scan repeats until a
    # full pass makes no repair; the raw-polyline fallback is idempotent, so
    # the loop terminates.
    for _ in range(4):
        changed = False
        for face in arc_graph.faces:
            walk_ids = _crossing_arc_ids(face)
            if not walk_ids:
                continue
            changed = True
            walk_seams = [aid for aid in walk_ids if aid in loose_ids]
            for aid in walk_seams:
                simplified_by_id[aid] = simplify_one(arcs_by_id[aid], _FILLER_TOLERANCE_SCALE)
            if walk_seams and not _crossing_arc_ids(face):
                continue
            for aid in walk_ids:
                simplified_by_id[aid] = simplify_one(arcs_by_id[aid], _FILLER_TOLERANCE_SCALE)
            if not _crossing_arc_ids(face):
                continue
            for aid in walk_ids:
                simplified_by_id[aid] = arcs_by_id[aid]
        if not changed:
            break


def simplify_arc_graph(
    arc_graph: ArcGraph,
    *,
    tolerance_mm: float = TOLERANCE_MM_DEFAULT,
    d_min_mm: float | None = None,
    filler_ids: frozenset[int] = frozenset(),
    config_hash: str = _UNSET_HASH,
) -> ArcGraph:
    """Simplify every arc's polyline independently; faces (walks) are
    unchanged since arc identity and endpoint count semantics are preserved
    -- only interior vertices are removed (MATH_SPEC §8.1).

    ``d_min_mm``, when given, scales each arc's tolerance down when its
    smallest adjacent face is below the printability area floor
    (``stages/vector/_face_area.py``) -- an arc bordering only comfortably
    large faces is unaffected (scale 1.0, identical to the ``None``
    behavior). This exists because a fixed absolute tolerance is a
    negligible fraction of a large face's boundary but can be a large
    fraction of a tiny face's, causing spurious ``fidelity`` (I1) failures
    on real photos with many small-but-legal (``merge_tiny``-surviving)
    regions.

    ``filler_ids`` (split_large's filler/rim cells) always get the smallest
    tolerance scale regardless of their own area -- see ``fit_curves``'s
    matching parameter for why area alone is not enough for a rim cell.
    """
    if tolerance_mm < 0.0:
        raise ConfigError(f"simplify: tolerance_mm must be >= 0, got {tolerance_mm}")
    tolerance_pt = tolerance_mm * _MM_TO_PT

    scale_by_arc: dict[int, float] = {}
    if d_min_mm is not None:
        reference_area = area_floor_pt2(d_min_mm)
        min_area_by_arc = min_adjacent_face_area_pt2_by_arc(arc_graph.arcs, arc_graph.faces)
        scale_by_arc = {
            arc_id: tolerance_scale_for_area(area, reference_area_pt2=reference_area)
            for arc_id, area in min_area_by_arc.items()
        }
    if filler_ids:
        # Same-color seams simplify LOOSE (cannot affect fidelity), other
        # filler arcs tight — see curves.py's identical logic.
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

    def _simplify_one(arc: Arc, scale: float) -> Arc:
        polyline = PolylineData(np.asarray(arc.points, dtype=np.float64), is_closed=arc.closed)
        out = _KERNEL.simplify_polyline(polyline, tolerance_pt * scale)
        return Arc(
            arc_id=arc.arc_id,
            points=np.asarray(out.coords, dtype=np.float64),
            left_region=arc.left_region,
            right_region=arc.right_region,
            closed=arc.closed,
        )

    simplified_by_id = {
        arc.arc_id: _simplify_one(arc, scale_by_arc.get(arc.arc_id, 1.0))
        for arc in arc_graph.arcs
    }

    loose_ids = {aid for aid, s in scale_by_arc.items() if s > 1.0}
    _repair_crossing_walks(arc_graph, simplified_by_id, loose_ids, _simplify_one)

    simplified_arcs = [simplified_by_id[arc.arc_id] for arc in arc_graph.arcs]
    return ArcGraph(
        arcs=tuple(simplified_arcs),
        faces=arc_graph.faces,
        work_scale=arc_graph.work_scale,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=arc_graph.provenance.source_hash,
        ),
    )


class SimplifyStage:
    """Stage wrapper: ``arc_graph`` -> ``arc_graph`` (replaced, simplified)."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        d_min_mm: float | None = None,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        tolerance = section.get("tolerance_mm", TOLERANCE_MM_DEFAULT)
        if not isinstance(tolerance, (int, float)) or not 0.0 <= float(tolerance) <= 2.0:
            raise ConfigError(f"simplify config: tolerance_mm must be in [0, 2], got {tolerance!r}")
        self._tolerance_mm = float(tolerance)
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
        return ("arc_graph",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        arc_graph = ctx.get("arc_graph")
        if not isinstance(arc_graph, ArcGraph):
            raise ConfigError("simplify requires an ArcGraph artifact")
        filler_ids = ctx.get("filler_region_ids") if ctx.has("filler_region_ids") else frozenset()
        if not isinstance(filler_ids, (set, frozenset)):
            filler_ids = frozenset()
        ctx.put(
            "arc_graph",
            simplify_arc_graph(
                arc_graph,
                tolerance_mm=self._tolerance_mm,
                d_min_mm=self._d_min_mm,
                filler_ids=frozenset(filler_ids),
                config_hash=self._config_hash,
            ),
        )
