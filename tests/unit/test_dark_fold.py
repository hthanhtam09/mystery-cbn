"""The dark fold absorbs outline strokes, never round dark features."""

from __future__ import annotations

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.organic_partition import organic_partition_regions, stage_seed

PROV = Provenance("organic", "1.0.0", "0" * 64, "1" * 64)
# 0 = pale ground, 1 = near-black ink (the dark fold's target band).
PAL = Palette(
    colors=(
        PaletteColor.from_lab(0, (94.0, 0.0, 0.0), 100),
        PaletteColor.from_lab(1, (4.0, 0.0, 0.0), 100),
        PaletteColor.from_lab(2, (70.0, 0.0, 0.0), 100),
    ),
    provenance=PROV,
)


def _run(rows: list[list[int]], *, max_inradius_px: float):
    graph = build_region_graph(
        LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV), PAL
    )
    new_graph, _, _, _ = organic_partition_regions(
        graph,
        PAL,
        mode="streamline",
        min_area_px=1e9,  # no subdivision: isolate the fold behaviour
        seed_density_px=25.0,
        rim_px=0.0,
        warp_px=0.0,
        noise_scale_px=6.0,
        fold_a_min_px=0.0,
        dark_fold_max_inradius_px=max_inradius_px,
        skip_dark_lab_l_threshold=37.0,
        warp_seed=stage_seed(0),
    )
    return {c.lab[0] for c in (PAL.colors[r.label] for r in new_graph.regions)}


def _blank(n: int = 60) -> list[list[int]]:
    return [[0] * n for _ in range(n)]


def test_round_dark_pupil_survives_the_dark_fold() -> None:
    # A filled disc of radius 8 (elongation ~1) well under the inradius gate.
    rows = _blank()
    cy = cx = 30
    for r in range(60):
        for c in range(60):
            if (r - cy) ** 2 + (c - cx) ** 2 <= 8**2:
                rows[r][c] = 1
    assert 4.0 in _run(rows, max_inradius_px=20.0), "the pupil must keep its dark fill"


def test_thin_winding_outline_stroke_is_folded_away() -> None:
    # A 3 px stroke tracing a long rectangle: same inradius band as the disc
    # above, but elongation in the tens.
    rows = _blank()
    for r in range(10, 50):
        for c in range(10, 13):
            rows[r][c] = 1
        for c in range(47, 50):
            rows[r][c] = 1
    for c in range(10, 50):
        for r in range(10, 13):
            rows[r][c] = 1
        for r in range(47, 50):
            rows[r][c] = 1
    assert 4.0 not in _run(rows, max_inradius_px=20.0), "the stroke must be absorbed"


def test_enclosed_dark_arc_a_closed_eye_survives() -> None:
    # A lidded eye is drawn as a thin curved arc: elongated like a stroke, so
    # only its being an ISLAND -- enclosed by one region -- tells them apart.
    # Folding it leaves nothing to draw the eye, and the face prints blank.
    rows = _blank()
    for c in range(18, 42):
        r = 30 + round(6 * ((c - 30) / 12.0) ** 2)
        for dr in range(3):
            rows[r + dr][c] = 1
    assert 4.0 in _run(rows, max_inradius_px=20.0), "a closed eye must survive"


def test_a_stroke_along_a_colour_border_is_still_folded() -> None:
    # Same arc thickness, but lying between two colour masses: after folding,
    # the region edge still traces the line, so absorbing it is right.
    rows = [[0] * 60 for _ in range(30)] + [[2] * 60 for _ in range(30)]
    for c in range(60):
        for dr in range(3):
            rows[29 + dr][c] = 1
    assert 4.0 not in _run(rows, max_inradius_px=20.0), "a border stroke must fold"
