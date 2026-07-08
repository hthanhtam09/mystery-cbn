"""Unit tests for the Curve Fitting stage (ENGINE_SPEC §18)."""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.model.vector import CurveSet
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.vector.arcgraph import build_arc_graph
from mysterycbn.stages.vector.curves import (
    CurveFitStage,
    _fit_schneider_run,
    fit_arc,
    fit_curves,
    fit_error_pt,
)
from mysterycbn.stages.vector.topology import build_topology_graph

PROV = Provenance("arcgraph", "1.0.0", "0" * 64, "1" * 64)
PAL4 = Palette(
    colors=tuple(PaletteColor.from_lab(i, (10.0 + 25.0 * i, 0.0, 0.0), 100) for i in range(4)),
    provenance=PROV,
)
BOX = (0.0, 0.0, 100.0, 100.0)


def _arc_graph(rows: list[list[int]]):
    lm = LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV)
    rg = build_region_graph(lm, PAL4)
    return build_arc_graph(build_topology_graph(rg.component_map), rg, content_box=BOX)


def _quarter_circle(n: int = 100, radius: float = 100.0) -> np.ndarray:
    theta = np.linspace(0.0, np.pi / 2.0, n)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def test_quarter_circle_fits_with_at_most_two_segments() -> None:
    pts = _quarter_circle()
    segments, corners, err = fit_arc(pts, tolerance_pt=0.5)
    assert len(segments) <= 2
    assert corners == ()
    assert err <= 0.5


def test_straight_line_is_one_exact_segment() -> None:
    pts = np.stack([np.linspace(0, 90, 10), np.linspace(0, 30, 10)], axis=1)
    segments, corners, err = fit_arc(pts, tolerance_pt=0.1)
    assert len(segments) == 1 and corners == () and err <= 1e-9
    ctrl = segments[0].control
    assert np.array_equal(ctrl[0], pts[0]) and np.array_equal(ctrl[3], pts[-1])


def test_corner_produces_c0_break() -> None:
    # L-shape: 90° turn at the middle vertex — a corner at any threshold ≤ 90.
    leg1 = np.stack([np.linspace(0, 50, 6), np.zeros(6)], axis=1)
    leg2 = np.stack([np.full(5, 50.0), np.linspace(10, 50, 5)], axis=1)
    pts = np.concatenate([leg1, leg2])
    segments, corners, _ = fit_arc(pts, tolerance_pt=0.5, corner_angle_deg=65.0)
    assert len(corners) == 1
    joint = corners[0]
    # The corner vertex survives bitwise as the shared chain point...
    assert np.array_equal(segments[joint].control[0], np.array([50.0, 0.0]))
    # ...and tangents are independent (intentional C0, not mirrored G1).
    t_in = segments[joint - 1].control[3] - segments[joint - 1].control[2]
    t_out = segments[joint].control[1] - segments[joint].control[0]
    cos = np.dot(t_in, t_out) / (np.linalg.norm(t_in) * np.linalg.norm(t_out))
    assert cos < 0.5  # ~90° break preserved


def test_junction_exactness_and_topology_carryover() -> None:
    graph = _arc_graph([[0, 0, 1], [2, 2, 1], [2, 2, 1]])
    for impl in ("schneider", "bezier", "chaikin", "catmull"):
        curve_set = fit_curves(graph, impl=impl)
        assert curve_set.faces is graph.faces  # topology carried over unchanged
        for arc, curve in zip(graph.arcs, curve_set.curves, strict=True):
            # Chain endpoints equal the arc's junction coordinates BITWISE.
            assert np.array_equal(curve.segments[0].control[0], arc.points[0])
            assert np.array_equal(curve.segments[-1].control[3], arc.points[-1])


def test_reparameterization_improves_error() -> None:
    # Coarse quarter circle: chord-length parameters are suboptimal; four
    # Newton–Raphson refits pull the single-cubic error below a tolerance
    # the unreparameterized fit misses (1 segment instead of 3).
    pts = _quarter_circle(n=24)
    segs_reparam, err_reparam = _fit_schneider_run(pts, 1.2, max_reparam=4)
    segs_none, _ = _fit_schneider_run(pts, 1.2, max_reparam=0)
    assert len(segs_reparam) == 1 and err_reparam <= 1.2
    assert len(segs_none) > 1


def test_determinism() -> None:
    graph = _arc_graph([[0, 1, 2], [0, 1, 2], [3, 3, 3]])
    a = fit_curves(graph)
    b = fit_curves(graph)
    assert a.to_dict() == b.to_dict()


def test_closed_arc_anchor_is_a_corner() -> None:
    graph = _arc_graph(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    closed = [a for a in graph.arcs if a.closed]
    assert closed
    curve_set = fit_curves(graph)
    for arc in closed:
        curve = curve_set.curves[arc.arc_id]
        # Chain starts and ends at the anchor exactly (cut there, C0).
        assert np.array_equal(curve.segments[0].control[0], arc.points[0])
        assert np.array_equal(curve.segments[-1].control[3], arc.points[0])


def test_unknown_impl_and_stage_validation() -> None:
    with pytest.raises(ConfigError, match="unknown curve fitter"):
        fit_arc(np.array([[0.0, 0.0], [1.0, 1.0]]), tolerance_pt=1.0, impl="potrace")
    with pytest.raises(ConfigError, match="fit_error_mm"):
        CurveFitStage({"fit_error_mm": 99.0})
    with pytest.raises(ConfigError, match="corner_angle_deg"):
        CurveFitStage({"corner_angle_deg": 5.0})
    with pytest.raises(ConfigError, match="impl"):
        CurveFitStage({"impl": "potrace"})
    with pytest.raises(ConfigError, match="> 0"):
        fit_error_pt(0.0)
    assert fit_error_pt(25.4) == pytest.approx(72.0)


def test_stage_wrapper_contract() -> None:
    stage = CurveFitStage({"fit_error_mm": 0.5, "impl": "schneider"})
    assert stage.name == "bezier"
    assert stage.requires == ("arc_graph",)
    assert stage.provides == ("curve_set",)
    graph = _arc_graph([[0, 1], [0, 1]])
    ctx = InMemoryContext(seed=0)
    ctx.put("arc_graph", graph)
    stage.run(ctx)
    curve_set = ctx.get("curve_set")
    assert isinstance(curve_set, CurveSet)
    assert curve_set.provenance.stage_name == "bezier"
    assert len(curve_set.curves) == len(graph.arcs)

    bad = InMemoryContext(seed=0)
    bad.put("arc_graph", "nope")
    with pytest.raises(ConfigError):
        stage.run(bad)
