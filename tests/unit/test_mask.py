"""Unit tests for the No-Color Mask stage's near-white and hand-drawn rules."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.mask import (
    NoColorMaskStage,
    decode_bitmap_mask,
    numbered_blank_region_ids,
    select_no_color_region_ids_from_bitmap,
    select_white_region_ids,
)

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


# --- hand-drawn bitmap mask ------------------------------------------------


def _two_region_graph():
    """20x20 split down the middle: label 0 left, label 1 right."""
    rows = [[0] * 10 + [1] * 10 for _ in range(20)]
    graph = _graph(rows)
    left = next(r.region_id for r in graph.regions if r.label == 0)
    right = next(r.region_id for r in graph.regions if r.label == 1)
    return graph, left, right


def test_stroke_grazing_a_neighbour_does_not_claim_it() -> None:
    # The whole left region plus a one-column graze of the right one -- what a
    # brush aimed at the left half unavoidably does along the shared boundary.
    graph, left, right = _two_region_graph()
    mask = np.zeros((20, 20), dtype=bool)
    mask[:, :10] = True
    mask[:, 10] = True
    selected = select_no_color_region_ids_from_bitmap(graph, mask)
    assert selected == frozenset({left})
    # coverage_fraction=0 restores the old any-pixel behaviour.
    assert select_no_color_region_ids_from_bitmap(
        graph, mask, coverage_fraction=0.0
    ) == frozenset({left, right})


def test_majority_coverage_claims_the_region() -> None:
    graph, left, right = _two_region_graph()
    mask = np.zeros((20, 20), dtype=bool)
    mask[:, 4:16] = True  # 60% of each half
    assert select_no_color_region_ids_from_bitmap(graph, mask) == frozenset({left, right})


def test_empty_mask_selects_nothing() -> None:
    graph, _, _ = _two_region_graph()
    mask = np.zeros((20, 20), dtype=bool)
    assert select_no_color_region_ids_from_bitmap(graph, mask) == frozenset()


def _stage_ctx(graph, palette, section):
    ctx = InMemoryContext(seed=0)
    ctx.put("region_graph", graph)
    ctx.put("palette", palette)
    NoColorMaskStage(section).run(ctx)
    return ctx


def test_painted_blanks_stay_numbered_while_white_ground_does_not() -> None:
    # Label 1 is near-white page ground touching the raster edge; label 0 is the
    # subject. Paint the subject: it must be left unpainted on the colored
    # preview (in no_color_region_ids) but still numbered (in the numbered set),
    # while the page ground is blanked outright (in no_color, not numbered).
    rows = [[1] * 20 for _ in range(20)]
    for r in range(6, 14):
        for c in range(6, 14):
            rows[r][c] = 0
    graph = _graph(rows)
    subject = next(r.region_id for r in graph.regions if r.label == 0)

    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[6:14, 6:14] = 255
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    ctx = _stage_ctx(
        graph,
        PAL,
        {
            "enabled": True,
            "bitmap": base64.b64encode(buf.getvalue()).decode(),
            "white_l_threshold": 95.0,
        },
    )

    no_color = ctx.get("no_color_region_ids")
    numbered = numbered_blank_region_ids(ctx.get("region_graph"), ctx.get("numbered_blank_pixels"))
    ground = {r.region_id for r in graph.regions if r.label == 1}
    assert subject in no_color  # unpainted on the colored preview
    assert numbered == frozenset({subject})  # ...but keeps its number
    assert ground <= no_color
    assert not (ground & numbered)  # page ground stays numberless


def test_numbered_blank_region_ids_tolerates_a_missing_raster() -> None:
    graph, _, _ = _two_region_graph()
    assert numbered_blank_region_ids(graph, None) == frozenset()
    assert numbered_blank_region_ids(graph, np.zeros((3, 3), dtype=bool)) == frozenset()


def test_decode_bitmap_mask_accepts_the_web_canvas_png() -> None:
    # What MaskEditor exports: a binary black/white PNG data URL at the source
    # resolution, white = no_color.
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[:, :20] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Decoded onto a half-scale region grid: nearest-neighbour keeps it binary.
    mask = decode_bitmap_mask(data_url, (20, 20))
    assert mask.shape == (20, 20)
    assert mask[:, :10].all()
    assert not mask[:, 10:].any()
