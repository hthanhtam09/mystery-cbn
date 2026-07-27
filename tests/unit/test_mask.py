"""Unit tests for the No-Color Mask stage's near-white rule."""

from __future__ import annotations

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.mask import select_white_region_ids

PROV = Provenance("mask", "1.0.0", "0" * 64, "1" * 64)
# 0 = mid grey subject, 1 = near-white (the L* >= 95 rule's target).
PAL = Palette(
    colors=(
        PaletteColor.from_lab(0, (55.0, 0.0, 0.0), 100),
        PaletteColor.from_lab(1, (98.0, 0.0, 0.0), 100),
    ),
    provenance=PROV,
)


def _graph(rows: list[list[int]]):
    return build_region_graph(LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV), PAL)


def test_enclosed_white_feature_keeps_its_number() -> None:
    # A near-white 4x4 patch (teeth / sclera) fully enclosed by the subject.
    rows = [[0] * 20 for _ in range(20)]
    for r in range(8, 12):
        for c in range(8, 12):
            rows[r][c] = 1
    assert select_white_region_ids(_graph(rows), PAL, l_threshold=95.0) == frozenset()


def test_white_ground_touching_the_raster_edge_is_folded() -> None:
    # Near-white page ground with the subject sitting in the middle.
    rows = [[1] * 20 for _ in range(20)]
    for r in range(6, 14):
        for c in range(6, 14):
            rows[r][c] = 0
    graph = _graph(rows)
    white = select_white_region_ids(graph, PAL, l_threshold=95.0)
    assert white == frozenset({r.region_id for r in graph.regions if r.label == 1})


def test_large_matted_white_backdrop_is_folded() -> None:
    # Near-white backdrop over a ground-plane-sized share of the page, ringed
    # by the subject so it never reaches the raster edge.
    rows = [[0] * 20 for _ in range(20)]
    for r in range(1, 19):
        for c in range(1, 19):
            rows[r][c] = 1
    graph = _graph(rows)
    white = select_white_region_ids(graph, PAL, l_threshold=95.0)
    assert white == frozenset({r.region_id for r in graph.regions if r.label == 1})


def test_a_dark_region_is_never_folded_however_placed() -> None:
    rows = [[0] * 20 for _ in range(20)]
    assert select_white_region_ids(_graph(rows), PAL, l_threshold=95.0) == frozenset()
