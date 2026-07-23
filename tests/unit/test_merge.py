"""Unit tests for the Tiny Region Merge stage (ENGINE_SPEC §11, MATH_SPEC §11).

Property-based tests for this stage live in
``tests/property/test_merge_properties.py`` (ARCHITECTURE.md §2, §10).
"""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.merge import (
    MergeTinyStage,
    area_floor_px,
    merge_cost,
    merge_tiny_regions,
)

PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)


def _palette(labs: list[tuple[float, float, float]]) -> Palette:
    colors = tuple(PaletteColor.from_lab(i, lab, 100) for i, lab in enumerate(labs))
    return Palette(colors=colors, provenance=PROV)


# 0 dark, 1 mid, 2 light, 3 very light: ΔE(1,2) < ΔE(1,0) etc.
PAL4 = _palette([(10.0, 0.0, 0.0), (40.0, 0.0, 0.0), (70.0, 0.0, 0.0), (95.0, 0.0, 0.0)])


def _graph(rows: list[list[int]], palette: Palette = PAL4):
    lm = LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV)
    return build_region_graph(lm, palette)


def test_area_floor() -> None:
    # work_scale = 1 pt/px → ppmm = 72/25.4 px/mm; d = 2 mm → A = π·ppmm².
    assert area_floor_px(2.0, 1.0) == pytest.approx(np.pi * (72.0 / 25.4) ** 2)
    with pytest.raises(ConfigError):
        area_floor_px(0.0, 1.0)
    with pytest.raises(ConfigError):
        area_floor_px(2.0, 0.0)


def test_merge_cost_formula() -> None:
    # One full-perimeter hug is worth λ ΔE00 of color mismatch.
    assert merge_cost(20.0, 10, 10, 15.0) == pytest.approx(5.0)
    assert merge_cost(20.0, 6, 10, 15.0) == pytest.approx(11.0)
    assert merge_cost(0.0, 1, 4, 0.0) == 0.0


def test_cost_picks_hugging_neighbor_over_closer_color() -> None:
    # Sliver region (label 2, 1×2 px) sits between label 1 (hugs 5 of its
    # 6-crack perimeter... construct: sliver's boundary mostly shared with
    # region A (label 0, far color), tiny contact with region B (label 1,
    # near color). λ = 15 must fold it into the hugging far-color neighbor
    # when ΔE advantage is small, and into the close color when λ = 0.
    rows = [
        [0, 0, 0, 0, 1],
        [0, 2, 2, 1, 1],
        [0, 0, 0, 0, 1],
    ]
    graph = _graph(rows)
    # region ids: 0 = label-0 field (9 px), 1 = label-1 field (4 px),
    # 2 = sliver (label 2, 2 px; hugs region 0 on 5 of 6 perimeter cracks).
    assert [(r.label, r.area_px) for r in graph.regions] == [(0, 9), (1, 4), (2, 2)]
    merged_geo, _, _ = merge_tiny_regions(graph, PAL4, a_min=3.0, lambda_boundary=50.0)
    assert sorted(r.area_px for r in merged_geo.regions) == [4, 11]  # sliver → hug
    merged_col, _, _ = merge_tiny_regions(graph, PAL4, a_min=3.0, lambda_boundary=0.0)
    # Pure color: sliver (label 2) joins label 1 (ΔE(2,1) < ΔE(2,0)).
    assert sorted(r.area_px for r in merged_col.regions) == [6, 9]


def test_single_neighbor_merges_regardless_of_cost() -> None:
    ring = [
        [1, 1, 1],
        [1, 3, 1],  # hole color maximally far from ring
        [1, 1, 1],
    ]
    merged, _, _ = merge_tiny_regions(_graph(ring), PAL4, a_min=2.0)
    assert len(merged.regions) == 1
    assert merged.regions[0].area_px == 9 and merged.regions[0].label == 1


def test_chain_merge_determinism_and_heap_update() -> None:
    # Two adjacent sub-floor regions: smallest merges first; the grown result
    # re-enters the heap (still sub-floor) and merges again — chain resolves
    # into the big field deterministically.
    rows = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 1],
        [0, 0, 0, 2, 2, 1],
    ]
    graph = _graph(rows)
    runs = [merge_tiny_regions(graph, PAL4, a_min=6.0) for _ in range(3)]
    for merged, _, _ in runs:
        assert len(merged.regions) == 1 and merged.regions[0].area_px == 18
        assert merged.component_map.tolist() == runs[0][0].component_map.tolist()


def test_survivor_ids_compact_order_preserving() -> None:
    rows = [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [2, 3, 3, 3, 1, 1],
    ]
    graph = _graph(rows)
    merged, palette, renumber = merge_tiny_regions(graph, PAL4, a_min=2.0)
    # Only the 1-px '2' merges; label 2 drops from the palette → labels 0/1/3
    # renumber to 0/1/2, survivors keep their relative id order.
    assert renumber == (0, 1, -1, 2)
    assert [c.lab for c in palette.colors] == [PAL4.colors[i].lab for i in (0, 1, 3)]
    assert [r.label for r in merged.regions] == [0, 1, 2]
    assert sorted(np.unique(merged.component_map).tolist()) == [0, 1, 2]


def test_palette_compaction_and_renumber_map() -> None:
    rows = [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 3],
    ]
    graph = _graph(rows)
    merged, palette, renumber = merge_tiny_regions(graph, PAL4, a_min=2.0)
    # Labels 2 and 3 lose their only regions; palette compacts 4 → 2? No:
    # label 1 had no region at all pre-merge, so survivors use labels {0}∪...
    assert len(merged.regions) == 1
    # Only label 0 survives → compaction would leave K = 1 < 2: skipped.
    assert palette is PAL4
    assert renumber == (0, 1, 2, 3)

    rows2 = [
        [0, 0, 0, 2, 2, 2],
        [0, 0, 0, 2, 2, 2],
        [0, 0, 3, 2, 2, 2],
    ]
    merged2, palette2, renumber2 = merge_tiny_regions(_graph(rows2), PAL4, a_min=2.0)
    assert renumber2 == (0, -1, 1, -1)  # labels 1 (absent) and 3 (merged away) drop
    assert palette2.size == 2
    assert [c.lab for c in palette2.colors] == [PAL4.colors[0].lab, PAL4.colors[2].lab]
    # The '3' pixel folds into the color-closer label-2 field: coverage 8 + 10.
    assert [c.coverage_px for c in palette2.colors] == [8, 10]
    assert {r.label for r in merged2.regions} == {0, 1}
    assert merged2.provenance.stage_name == "merge_tiny"


def test_floor_exceeding_content_area_is_config_error() -> None:
    graph = _graph([[0, 1], [0, 1]])
    with pytest.raises(ConfigError, match="content area"):
        merge_tiny_regions(graph, PAL4, a_min=4.0)


def test_stage_wrapper_contract() -> None:
    stage = MergeTinyStage({"lambda_boundary": 15.0}, d_min_mm=1.0)
    assert stage.name == "merge_tiny"
    assert stage.requires == ("region_graph", "palette", "raster_working")
    assert stage.provides == ("region_graph", "palette")
    assert stage.config_section == "merge"
    with pytest.raises(ConfigError, match="lambda_boundary"):
        MergeTinyStage({"lambda_boundary": 99.0})

    class _Raster:
        work_scale = 1.0

    ctx = InMemoryContext(seed=0)
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[:16] = 1
    labels[5, 5] = 2  # 1-px speck, far below the d=1mm floor (≈6.3 px²)
    ctx.put("region_graph", build_region_graph(LabelMap(labels=labels, provenance=PROV), PAL4))
    ctx.put("palette", PAL4)
    ctx.put("raster_working", _Raster())
    stage.run(ctx)
    graph = ctx.get("region_graph")
    assert len(graph.regions) == 2
    assert ctx.get("palette").size == 2

    bad = InMemoryContext(seed=0)
    bad.put("region_graph", "nope")
    bad.put("palette", PAL4)
    bad.put("raster_working", _Raster())
    with pytest.raises(ConfigError):
        stage.run(bad)


def test_protect_dark_dot_keeps_subfloor_dark_region() -> None:
    # A single dark pixel (label 0, L*=10) in a light field (label 3, L*=95):
    # sub-floor, so it merges away by default, but the dark-dot protection
    # keeps it (a pupil/nostril on a light surround).
    rows = [[3, 3, 3], [3, 0, 3], [3, 3, 3]]
    graph = _graph(rows)
    without, _, _ = merge_tiny_regions(graph, PAL4, a_min=2.0)
    assert len(without.regions) == 1  # dark dot absorbed into the light field
    with_protect, _, _ = merge_tiny_regions(
        graph, PAL4, a_min=2.0, protect_dark_l=20.0, protect_dark_delta_l=15.0
    )
    labels = {r.label for r in with_protect.regions}
    assert 0 in labels  # the dark dot survived
    assert len(with_protect.regions) == 2


def test_protect_dark_skips_when_neighbor_not_light_enough() -> None:
    # Dark dot (L*=10) beside a mid region (label 1, L*=40): the light gap
    # (30) is below the delta, so it is NOT protected and still merges.
    rows = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    graph = _graph(rows)
    merged, _, _ = merge_tiny_regions(
        graph, PAL4, a_min=2.0, protect_dark_l=20.0, protect_dark_delta_l=40.0
    )
    assert len(merged.regions) == 1  # delta 30 < 40 -> not protected
