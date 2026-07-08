"""Unit tests for the Label Placement stage (ENGINE_SPEC §19)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.layout import LabelMode, LabelPlan
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import (
    LabelPlacementStage,
    fitted_font_size,
    largest_empty_circle,
    place_labels,
    text_bbox_pt,
)
from mysterycbn.stages.vector.arcgraph import build_arc_graph
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph

PROV = Provenance("bezier", "1.0.0", "0" * 64, "1" * 64)
PAL8 = Palette(
    colors=tuple(PaletteColor.from_lab(i, (8.0 + 11.0 * i, 0.0, 0.0), 100) for i in range(8)),
    provenance=PROV,
)


def _ring(points: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


def _rect_ring(x0: float, y0: float, x1: float, y1: float, n: int = 8) -> np.ndarray:
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    top = np.stack([xs, np.full(n, y0)], axis=1)
    right = np.stack([np.full(n, x1), ys], axis=1)[1:]
    bottom = np.stack([xs[::-1], np.full(n, y1)], axis=1)[1:]
    left = np.stack([np.full(n, x0), ys[::-1]], axis=1)[1:-1]
    return np.concatenate([top, right, bottom, left])


def _pipeline(rows: list[list[int]], box=(0.0, 0.0, 200.0, 200.0)):
    lm = LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV)
    rg = build_region_graph(lm, PAL8)
    ag = build_arc_graph(build_topology_graph(rg.component_map), rg, content_box=box)
    return fit_curves(ag), rg


def test_pole_of_c_shape_lies_inside_the_c() -> None:
    # C-shape: 100×100 square minus a 60×80 bite from the right.
    outer = _ring(
        [
            (0, 0), (100, 0), (100, 10), (40, 10), (40, 90),
            (100, 90), (100, 100), (0, 100),
        ]
    )  # fmt: skip
    (px, py), r = largest_empty_circle([outer], 0.5)
    assert px < 40.0  # inside the C's spine, not the bite
    assert r > 15.0
    # The clearance circle lies inside the shape: probe the bite region.
    assert not (40.0 < px < 100.0 and 10.0 < py < 90.0)


def test_annulus_pole_lies_inside_the_ring() -> None:
    theta = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    outer = np.stack([50 * np.cos(theta), 50 * np.sin(theta)], axis=1)
    hole = np.stack([20 * np.cos(theta), 20 * np.sin(theta)], axis=1)
    (px, py), r = largest_empty_circle([outer, hole], 0.25)
    rho = math.hypot(px, py)
    assert 20.0 < rho < 50.0  # inside the ring band, not the hole
    assert r == pytest.approx((50.0 - 20.0) / 2.0, abs=1.0)


def test_font_size_formula_vs_brute_force_bbox_check() -> None:
    for number, clearance in ((7, 10.0), (23, 10.0), (5, 3.0)):
        size = fitted_font_size(number, clearance)
        w, h = text_bbox_pt(number, size)
        # Fit condition: the bbox half-diagonal equals the clearance radius.
        assert math.hypot(w, h) / 2.0 == pytest.approx(clearance, rel=1e-9)
        # 1% larger no longer fits — the formula is the exact optimum.
        w2, h2 = text_bbox_pt(number, size * 1.01)
        assert math.hypot(w2, h2) / 2.0 > clearance
    # Two-digit closed-form seed ≈ 1.35–1.37 · r (MATH_SPEC §14.1).
    assert fitted_font_size(42, 1.0) == pytest.approx(1.364, abs=0.01)


def test_leader_fallback_on_sliver() -> None:
    # ~1 mm × 30 mm sliver (label 1) inside a big field: too thin for 6 pt.
    rows = [[0] * 40 for _ in range(40)]
    for c in range(4, 36):
        rows[19][c] = 1
    curve_set, rg = _pipeline(rows, box=(0.0, 0.0, 120.0, 120.0))
    plan, findings = place_labels(curve_set, rg)
    assert findings == ()
    sliver = next(lb for lb in plan.labels if lb.printed_number == 2)
    assert sliver.mode is LabelMode.LEADER
    assert sliver.font_size_pt == 6.0
    assert sliver.leader is not None
    assert sliver.leader[0] == sliver.anchor  # leader runs from text to pole
    # 100% of faces labeled; big region stays in-region.
    assert len(plan.labels) == len(curve_set.faces)
    big = next(lb for lb in plan.labels if lb.printed_number == 1)
    assert big.mode is LabelMode.IN_REGION


def test_no_overlaps_and_determinism() -> None:
    rng = np.random.default_rng(3)
    rows = np.repeat(np.repeat(rng.integers(0, 8, (5, 5)), 8, axis=0), 8, axis=1)
    curve_set, rg = _pipeline(rows.tolist())
    plan_a, _ = place_labels(curve_set, rg)
    plan_b, _ = place_labels(curve_set, rg)
    assert plan_a.to_dict() == plan_b.to_dict()  # displacement determinism
    rects = []
    for lb in plan_a.labels:
        w, h = text_bbox_pt(lb.printed_number, lb.font_size_pt)
        rects.append(
            (lb.anchor[0] - w / 2, lb.anchor[1] - h / 2, lb.anchor[0] + w / 2, lb.anchor[1] + h / 2)
        )
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def test_in_region_labels_fit_inside_their_clearance() -> None:
    curve_set, rg = _pipeline([[0, 1], [0, 1]])
    plan, findings = place_labels(curve_set, rg)
    assert findings == ()
    for lb in plan.labels:
        assert lb.mode is LabelMode.IN_REGION
        assert 6.0 <= lb.font_size_pt <= 14.0
        w, h = text_bbox_pt(lb.printed_number, lb.font_size_pt)
        assert math.hypot(w, h) / 2.0 <= lb.clearance_pt + 1e-9


def test_stage_wrapper_contract() -> None:
    with pytest.raises(ConfigError, match="polylabel_precision_pt"):
        LabelPlacementStage({"polylabel_precision_pt": 99.0})
    with pytest.raises(ConfigError, match="leader_ring_mm"):
        LabelPlacementStage({"leader_ring_mm": 0.1})
    with pytest.raises(ConfigError, match="font_min_pt"):
        LabelPlacementStage({}, font_min_pt=10.0, font_max_pt=6.0)

    stage = LabelPlacementStage({})
    assert stage.name == "labels"
    assert stage.requires == ("curve_set", "region_graph")
    assert stage.provides == ("label_plan", "label_findings")
    curve_set, rg = _pipeline([[0, 1], [0, 1]])
    ctx = InMemoryContext(seed=0)
    ctx.put("curve_set", curve_set)
    ctx.put("region_graph", rg)
    stage.run(ctx)
    plan = ctx.get("label_plan")
    assert isinstance(plan, LabelPlan)
    assert plan.provenance.stage_name == "labels"
    assert ctx.get("label_findings").findings == ()

    bad = InMemoryContext(seed=0)
    bad.put("curve_set", "nope")
    bad.put("region_graph", rg)
    with pytest.raises(ConfigError):
        stage.run(bad)
