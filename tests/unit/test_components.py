"""Unit tests for the Connected Components stage (ENGINE_SPEC §9–§10).

Property-based tests for this stage live in
``tests/property/test_components_properties.py`` (ARCHITECTURE.md §2, §10).
"""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import ConnectedComponentsStage, build_region_graph

PROV = Provenance("denoise", "1.0.0", "0" * 64, "1" * 64)


def _palette(k: int) -> Palette:
    colors = tuple(
        PaletteColor.from_lab(i, (5.0 + 90.0 * i / max(k - 1, 1), 0.0, 0.0), 100) for i in range(k)
    )
    return Palette(colors=colors, provenance=PROV)


PAL4 = _palette(4)


def _label_map(rows: list[list[int]]) -> LabelMap:
    return LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV)


def test_diagonal_pixels_are_two_regions() -> None:
    graph = build_region_graph(_label_map([[1, 0], [0, 1]]), PAL4)
    assert len(graph.regions) == 4  # the two 1-pixels must NOT connect diagonally
    assert graph.component_map.tolist() == [[0, 1], [2, 3]]


def test_donut_hole_is_a_distinct_region() -> None:
    ring = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    graph = build_region_graph(_label_map(ring), PAL4)
    assert len(graph.regions) == 2
    hole = graph.regions[1]
    assert hole.label == 0 and hole.area_px == 1
    assert graph.neighbors(1) == (0,)
    assert graph.edge_weight(0, 1) == (4.0, PAL4.delta_e_table[1, 0])


def test_region_statistics() -> None:
    graph = build_region_graph(_label_map([[0, 0, 1], [0, 0, 1]]), PAL4)
    big, thin = graph.regions
    assert (big.area_px, thin.area_px) == (4, 2)
    assert big.bbox == (0, 0, 1, 1) and thin.bbox == (0, 2, 1, 2)
    assert big.seed_px == (0, 0) and thin.seed_px == (0, 2)
    assert big.centroid == (0.5, 0.5) and thin.centroid == (0.5, 2.0)
    assert big.perimeter_px == 8 and thin.perimeter_px == 6
    assert graph.edges == ((0, 1, float(PAL4.delta_e_table[0, 1]), 2),)


def test_2x2_four_regions_form_a_4_cycle() -> None:
    graph = build_region_graph(_label_map([[0, 1], [2, 3]]), PAL4)
    assert [(a, b, w) for a, b, _, w in graph.edges] == [
        (0, 1, 1),
        (0, 2, 1),
        (1, 3, 1),
        (2, 3, 1),
    ]  # no diagonal edges (0,3) or (1,2)
    for region in graph.regions:
        assert graph.neighbors(region.region_id) == tuple(
            sorted({0, 1, 2, 3} - {region.region_id, 3 - region.region_id})
        )
        assert region.perimeter_px == 4  # 2 shared cracks + 2 border cracks


def test_id_order_is_raster_scan_first_occurrence() -> None:
    labels = [[2, 2, 0], [1, 2, 0], [1, 1, 1]]
    graph = build_region_graph(_label_map(labels), PAL4)
    assert [r.label for r in graph.regions] == [2, 0, 1]
    assert [r.seed_px for r in graph.regions] == [(0, 0), (0, 2), (1, 0)]
    again = build_region_graph(_label_map(labels), PAL4)
    assert again.component_map.tolist() == graph.component_map.tolist()
    assert again.edges == graph.edges


def test_single_region_page() -> None:
    graph = build_region_graph(_label_map([[1] * 4] * 3), PAL4)
    assert len(graph.regions) == 1 and graph.edges == ()
    assert graph.regions[0].perimeter_px == 2 * (3 + 4)
    assert graph.regions[0].centroid == (1.0, 1.5)


def test_same_label_regions_can_be_adjacent_only_diagonally() -> None:
    # 4-connectivity: equal labels touching orthogonally are ONE region, so
    # any edge between same-label regions has ΔE00 = 0 and diagonal contact.
    graph = build_region_graph(_label_map([[0, 1], [1, 0]]), PAL4)
    zero_ids = [r.region_id for r in graph.regions if r.label == 0]
    assert len(zero_ids) == 2
    with pytest.raises(KeyError):
        graph.edge_weight(*zero_ids)


def test_stage_wrapper_contract() -> None:
    stage = ConnectedComponentsStage({})
    assert stage.name == "regions"
    assert stage.requires == ("label_map", "palette")
    assert stage.provides == ("region_graph",)
    ctx = InMemoryContext(seed=0)
    ctx.put("label_map", _label_map([[0, 1], [2, 3]]))
    ctx.put("palette", PAL4)
    stage.run(ctx)
    graph = ctx.get("region_graph")
    assert len(graph.regions) == 4
    assert graph.provenance.stage_name == "regions"

    bad = InMemoryContext(seed=0)
    bad.put("label_map", "nope")
    bad.put("palette", PAL4)
    with pytest.raises(ConfigError):
        stage.run(bad)
