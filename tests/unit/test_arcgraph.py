"""Unit tests for the Arc Graph stage (ENGINE_SPEC §15).

Property-based tests for this stage live in
``tests/property/test_arcgraph_properties.py`` (ARCHITECTURE.md §2, §10).
"""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.model.vector import ArcGraph
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.vector.arcgraph import (
    ArcGraphStage,
    build_arc_graph,
    content_box_pt,
)
from mysterycbn.stages.vector.topology import build_topology_graph

PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)
PAL4 = Palette(
    colors=tuple(PaletteColor.from_lab(i, (10.0 + 25.0 * i, 0.0, 0.0), 100) for i in range(4)),
    provenance=PROV,
)
BOX = (10.0, 20.0, 100.0, 200.0)  # (x0, y0, w, h) pt


def _graphs(rows: list[list[int]]):
    lm = LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV)
    rg = build_region_graph(lm, PAL4)
    return build_topology_graph(rg.component_map), rg


def _build(rows: list[list[int]], box: tuple[float, float, float, float] = BOX) -> ArcGraph:
    topo, rg = _graphs(rows)
    return build_arc_graph(topo, rg, content_box=box)


def test_two_region_page_counts() -> None:
    graph = _build([[0, 1], [0, 1]])
    # V=6 junctions, E=7 arcs, F=3 (2 regions + exterior): 6−7+3 = 2.
    assert len(graph.arcs) == 7
    assert len(graph.faces) == 2
    for face in graph.faces:
        assert face.hole_walks == ()
        assert len(face.outer_walk) == 4  # two border pieces + wall + border
    # Every arc borders exactly 2 faces counting sides; the exterior face is
    # not stored, so border arcs appear once and the shared wall twice.
    refs: dict[int, int] = {}
    for face in graph.faces:
        for arc_id, _ in face.outer_walk:
            refs[arc_id] = refs.get(arc_id, 0) + 1
    assert sorted(refs.values()) == [1, 1, 1, 1, 1, 1, 2]


def test_face_region_correspondence_and_labels() -> None:
    rows = [[2, 2, 0], [1, 2, 0], [1, 1, 1]]
    topo, rg = _graphs(rows)
    graph = build_arc_graph(topo, rg, content_box=BOX)
    assert [f.face_id for f in graph.faces] == [r.region_id for r in rg.regions]
    assert [f.label for f in graph.faces] == [r.label for r in rg.regions]


def test_donut_hole_attachment() -> None:
    graph = _build(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    ring, hole = graph.faces
    assert len(ring.hole_walks) == 1  # the island ring is region 0's hole
    assert hole.hole_walks == ()
    (hole_walk,) = ring.hole_walks
    (outer_walk,) = (hole.outer_walk,)
    # Hole walk of the ring and outer walk of the island share the SAME arc,
    # opposite sides — the shared-boundary guarantee.
    assert hole_walk[0][0] == outer_walk[0][0]
    assert hole_walk[0][1] != outer_walk[0][1]


def test_nested_donut() -> None:
    rows = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    graph = _build(rows)
    outer, ring, core = graph.faces
    assert len(outer.hole_walks) == 1  # ring's outer boundary
    assert len(ring.hole_walks) == 1  # core's boundary
    assert core.hole_walks == ()


def test_scale_applied_exactly_once() -> None:
    rows = [[0, 1], [0, 1]]
    graph = _build(rows)
    # s = min(100/2, 200/2) = 50; letterboxed content spans exactly s·W × s·H.
    assert graph.work_scale == pytest.approx(50.0)
    xs = np.concatenate([a.points[:, 0] for a in graph.arcs])
    ys = np.concatenate([a.points[:, 1] for a in graph.arcs])
    assert (float(xs.min()), float(xs.max())) == (10.0, 110.0)
    # Vertical letterbox centering: (200 − 100)/2 = 50 offset inside the box.
    assert (float(ys.min()), float(ys.max())) == (70.0, 170.0)
    # Coordinate ratio equals the stored provenance scale (applied once).
    spread = (xs.max() - xs.min()) / 2  # page width = 2 px
    assert float(spread) == pytest.approx(graph.work_scale)


def test_single_region_page() -> None:
    graph = _build([[0, 0], [0, 0]])
    assert len(graph.faces) == 1
    assert len(graph.faces[0].outer_walk) == 4  # four border arcs
    assert graph.faces[0].hole_walks == ()


def test_content_box_pt_validation() -> None:
    x0, y0, w, h = content_box_pt((215.9, 279.4, 12.7))
    assert x0 == y0 == pytest.approx(12.7 * 72 / 25.4)
    assert w == pytest.approx((215.9 - 25.4) * 72 / 25.4)
    assert h == pytest.approx((279.4 - 25.4) * 72 / 25.4)
    with pytest.raises(ConfigError, match="content area"):
        content_box_pt((20.0, 100.0, 10.0))


def test_stage_wrapper_contract() -> None:
    stage = ArcGraphStage({"width_mm": 100.0, "height_mm": 100.0, "margin_mm": 10.0})
    assert stage.name == "arcgraph"
    assert stage.requires == ("topology_graph", "region_graph")
    assert stage.provides == ("arc_graph",)
    assert stage.config_section == "page"
    with pytest.raises(ConfigError, match="must be numbers"):
        ArcGraphStage({"width_mm": "wide"})

    topo, rg = _graphs([[0, 1], [0, 1]])
    ctx = InMemoryContext(seed=0)
    ctx.put("topology_graph", topo)
    ctx.put("region_graph", rg)
    stage.run(ctx)
    graph = ctx.get("arc_graph")
    assert isinstance(graph, ArcGraph)
    assert graph.provenance.stage_name == "arcgraph"

    bad = InMemoryContext(seed=0)
    bad.put("topology_graph", "nope")
    bad.put("region_graph", rg)
    with pytest.raises(ConfigError):
        stage.run(bad)
