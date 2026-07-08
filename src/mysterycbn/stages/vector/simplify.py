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

from collections.abc import Mapping

import numpy as np

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.geometry.default import DefaultGeometryKernel
from mysterycbn.foundation.geometry.primitives import PolylineData
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Provenance
from mysterycbn.model.vector import Arc, ArcGraph

STAGE_NAME = "simplify"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

TOLERANCE_MM_DEFAULT = 0.15
_MM_TO_PT = 72.0 / 25.4

_KERNEL = DefaultGeometryKernel()


def simplify_arc_graph(
    arc_graph: ArcGraph,
    *,
    tolerance_mm: float = TOLERANCE_MM_DEFAULT,
    config_hash: str = _UNSET_HASH,
) -> ArcGraph:
    """Simplify every arc's polyline independently; faces (walks) are
    unchanged since arc identity and endpoint count semantics are preserved
    -- only interior vertices are removed (MATH_SPEC §8.1)."""
    if tolerance_mm < 0.0:
        raise ConfigError(f"simplify: tolerance_mm must be >= 0, got {tolerance_mm}")
    tolerance_pt = tolerance_mm * _MM_TO_PT
    simplified_arcs = []
    for arc in arc_graph.arcs:
        polyline = PolylineData(np.asarray(arc.points, dtype=np.float64), is_closed=arc.closed)
        out = _KERNEL.simplify_polyline(polyline, tolerance_pt)
        simplified_arcs.append(
            Arc(
                arc_id=arc.arc_id,
                points=np.asarray(out.coords, dtype=np.float64),
                left_region=arc.left_region,
                right_region=arc.right_region,
                closed=arc.closed,
            )
        )
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
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        tolerance = section.get("tolerance_mm", TOLERANCE_MM_DEFAULT)
        if not isinstance(tolerance, (int, float)) or not 0.0 <= float(tolerance) <= 2.0:
            raise ConfigError(f"simplify config: tolerance_mm must be in [0, 2], got {tolerance!r}")
        self._tolerance_mm = float(tolerance)
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
        ctx.put(
            "arc_graph",
            simplify_arc_graph(
                arc_graph,
                tolerance_mm=self._tolerance_mm,
                config_hash=self._config_hash,
            ),
        )
