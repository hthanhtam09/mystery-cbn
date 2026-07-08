"""Unit tests for the Topology Graph stage (ENGINE_SPEC §14).

Property-based tests for this stage live in
``tests/property/test_topology_properties.py`` (ARCHITECTURE.md §2, §10).
"""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.model.vector import Arc, TopologyGraph
from mysterycbn.stages.graph.components import build_region_graph, label_components
from mysterycbn.stages.vector.topology import (
    TopologyStage,
    build_topology_graph,
    validate_topology,
)

PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)
PAL4 = Palette(
    colors=tuple(PaletteColor.from_lab(i, (10.0 + 25.0 * i, 0.0, 0.0), 100) for i in range(4)),
    provenance=PROV,
)


def _cmap(rows: list[list[int]]) -> np.ndarray:
    return label_components(np.array(rows, dtype=np.int32))


def _corners(graph: TopologyGraph) -> set[tuple[int, int]]:
    return {((int(r) + 1) // 2, (int(c) + 1) // 2) for r, c in graph.junctions.tolist()}


def test_single_region_page() -> None:
    graph = build_topology_graph(_cmap([[0, 0], [0, 0]]))
    # 4 page-corner junctions, 4 border arcs, no interior structure.
    assert _corners(graph) == {(0, 0), (0, 2), (2, 0), (2, 2)}
    assert len(graph.arcs) == 4
    assert all({a.left_region, a.right_region} == {-1, 0} for a in graph.arcs)
    assert sum(len(a.points) - 1 for a in graph.arcs) == 8
    validate_topology(graph, _cmap([[0, 0], [0, 0]]))


def test_vertical_split() -> None:
    cmap = _cmap([[0, 1], [0, 1]])
    graph = build_topology_graph(cmap)
    # Junctions: 4 page corners + the two border points where the split meets.
    assert _corners(graph) == {(0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (2, 1)}
    assert len(graph.arcs) == 7  # 2+2 top/bottom halves, 2 sides, 1 shared wall
    walls = [a for a in graph.arcs if {a.left_region, a.right_region} == {0, 1}]
    assert len(walls) == 1 and len(walls[0].points) == 3  # ONE shared polyline
    validate_topology(graph, cmap)


def test_t_junction_three_regions() -> None:
    cmap = _cmap([[0, 0], [1, 2]])
    graph = build_topology_graph(cmap)
    interior = {c for c in _corners(graph) if 0 < c[0] < 2 and 0 < c[1] < 2}
    assert interior == {(1, 1)}  # the T-point
    assert len(graph.arcs) == 10
    interior_pairs = {
        (a.left_region, a.right_region)
        for a in graph.arcs
        if a.left_region != -1 and a.right_region != -1
    }
    assert len(interior_pairs) == 3  # 0|1, 0|2, 1|2 — one arc each
    validate_topology(graph, cmap)


def test_island_is_one_closed_arc_no_interior_junctions() -> None:
    cmap = _cmap(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    graph = build_topology_graph(cmap)
    assert _corners(graph) == {(0, 0), (0, 3), (3, 0), (3, 3)}  # page corners only
    closed = [a for a in graph.arcs if a.closed]
    assert len(closed) == 1
    island = closed[0]
    assert len(island.points) == 5  # 4 unit cracks, anchor repeated
    assert np.array_equal(island.points[0], island.points[-1])
    assert tuple(island.points[0]) == (1.0, 1.0)  # lex-smallest corner (1,1) doubled
    assert {island.left_region, island.right_region} == {0, 1}
    validate_topology(graph, cmap)


def test_degree_4_corner() -> None:
    cmap = _cmap([[0, 1], [2, 3]])  # four regions meet at the centre corner
    graph = build_topology_graph(cmap)
    assert (1, 1) in _corners(graph)
    centre = np.array([1, 1], dtype=np.float64)
    incident = [
        a
        for a in graph.arcs
        if np.array_equal(a.points[0], centre) or np.array_equal(a.points[-1], centre)
    ]
    assert len(incident) == 4  # one junction, four incident arcs
    validate_topology(graph, cmap)


def test_arc_id_stability_and_orientation_canonicality() -> None:
    cmap = _cmap([[2, 2, 0], [1, 2, 0], [1, 1, 1]])
    first = build_topology_graph(cmap)
    second = build_topology_graph(cmap)
    assert [a.to_dict() for a in first.arcs] == [a.to_dict() for a in second.arcs]
    # Ids follow (min corner, left, right); open arcs stored lex-smaller-first.
    keys = [
        (tuple(a.points.min(axis=0).tolist()), a.left_region, a.right_region) for a in first.arcs
    ]
    assert keys == sorted(keys)
    for a in first.arcs:
        if not a.closed:
            fwd = tuple(map(tuple, a.points.tolist()))
            assert fwd <= tuple(reversed(fwd))


def test_pair_constancy_validation_catches_corruption() -> None:
    cmap = _cmap([[0, 1], [0, 1]])
    graph = build_topology_graph(cmap)
    wall = next(a for a in graph.arcs if {a.left_region, a.right_region} == {0, 1})
    tampered_arcs = tuple(
        a
        if a.arc_id != wall.arc_id
        else Arc(a.arc_id, a.points, a.right_region, a.left_region, a.closed)
        for a in graph.arcs
    )
    tampered = TopologyGraph(graph.junctions, tampered_arcs, graph.provenance)
    with pytest.raises(StageError, match="pair inconsistent"):
        validate_topology(tampered, cmap)


def test_missing_arc_is_a_gap() -> None:
    cmap = _cmap([[0, 1], [0, 1]])
    graph = build_topology_graph(cmap)
    kept = tuple(
        Arc(i, a.points, a.left_region, a.right_region, a.closed)
        for i, a in enumerate(graph.arcs[:-1])
    )
    with pytest.raises(StageError, match=r"gap|Euler"):
        validate_topology(TopologyGraph(graph.junctions, kept, graph.provenance), cmap)


def test_stage_wrapper_contract() -> None:
    stage = TopologyStage()
    assert stage.name == "topology"
    assert stage.requires == ("region_graph",)
    assert stage.provides == ("topology_graph",)
    lm = LabelMap(labels=np.array([[0, 1], [0, 1]], dtype=np.int32), provenance=PROV)
    ctx = InMemoryContext(seed=0)
    ctx.put("region_graph", build_region_graph(lm, PAL4))
    stage.run(ctx)
    topo = ctx.get("topology_graph")
    assert isinstance(topo, TopologyGraph)
    assert topo.provenance.stage_name == "topology"

    bad = InMemoryContext(seed=0)
    bad.put("region_graph", "nope")
    with pytest.raises(ConfigError):
        stage.run(bad)
