"""Unit tests for the default geometry kernel (MATH_SPEC §7–§9, §13)."""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.geometry.default import DefaultGeometryKernel, _shoelace
from mysterycbn.foundation.geometry.primitives import PolylineData

K = DefaultGeometryKernel()


def _area_sum(loops) -> float:  # type: ignore[no-untyped-def]
    return sum(_shoelace(np.asarray(lp.coords)) for lp in loops)


# ------------------------------------------------------------- crack tracing


def test_trace_single_pixel_region() -> None:
    loops = K.trace_cracks(np.zeros((1, 1), dtype=np.int32))
    assert len(loops) == 1
    assert loops[0].coords.shape == (4, 2)
    assert abs(_shoelace(loops[0].coords)) == pytest.approx(1.0)


def test_trace_two_regions_area_identity() -> None:
    labels = np.array([[0, 1]], dtype=np.int32)
    loops = K.trace_cracks(labels)
    assert len(loops) == 2  # one boundary loop per region
    assert abs(_area_sum(loops)) == pytest.approx(2.0)


def test_trace_donut_produces_hole_loop() -> None:
    labels = np.zeros((3, 3), dtype=np.int32)
    labels[1, 1] = 1
    loops = K.trace_cracks(labels)
    # ring outer + ring hole + island outer = 3 loops; signed sum = total area.
    assert len(loops) == 3
    assert abs(_area_sum(loops)) == pytest.approx(9.0)


def test_trace_t_junction_and_directed_coverage() -> None:
    # Three regions meeting at one corner: undirected-once tracing is impossible
    # here (odd crack degree) — the per-side model must still cover everything.
    labels = np.array([[0, 0], [1, 2]], dtype=np.int32)
    loops = K.trace_cracks(labels)
    assert abs(_area_sum(loops)) == pytest.approx(4.0)
    # Per region: one loop each.
    assert len(loops) == 3


def test_trace_random_maps_area_property() -> None:
    rng = np.random.default_rng(3)
    for _ in range(10):
        h, w = rng.integers(1, 7, 2)
        labels = rng.integers(0, 3, (h, w)).astype(np.int32)
        loops = K.trace_cracks(labels)
        assert abs(_area_sum(loops)) == pytest.approx(float(h * w))


def test_trace_determinism() -> None:
    labels = np.random.default_rng(4).integers(0, 3, (6, 6)).astype(np.int32)
    a = K.trace_cracks(labels)
    b = K.trace_cracks(labels)
    assert len(a) == len(b)
    for la, lb in zip(a, b, strict=True):
        np.testing.assert_array_equal(la.coords, lb.coords)


# ------------------------------------------------------------- simplification


def test_simplify_staircase_to_chord() -> None:
    # Unit staircase: effective areas are 0.5, far below tolerance² = 4.
    steps = [(0.0, 0.0)]
    for i in range(10):
        steps.append((i + 1.0, float(i)))
        steps.append((i + 1.0, i + 1.0))
    line = PolylineData(np.array(steps))
    # tolerance² must exceed every effective area reachable during removal
    # (≤ ~7 here) for full collapse to the chord.
    out = K.simplify_polyline(line, tolerance=4.0)
    assert out.coords.shape[0] == 2
    np.testing.assert_array_equal(out.coords[[0, -1]], line.coords[[0, -1]])
    # At a small tolerance VW stops early but still reduces the staircase.
    partial = K.simplify_polyline(line, tolerance=1.0)
    assert 2 < partial.coords.shape[0] < line.coords.shape[0]


def test_simplify_preserves_significant_vertices() -> None:
    tri = PolylineData(np.array([[0.0, 0.0], [5.0, 10.0], [10.0, 0.0]]))
    out = K.simplify_polyline(tri, tolerance=1.0)  # area 50 >> 1
    assert out.coords.shape[0] == 3


def test_simplify_closed_keeps_minimum_four() -> None:
    square = PolylineData(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), is_closed=True
    )
    out = K.simplify_polyline(square, tolerance=100.0)
    assert out.is_closed
    assert out.coords.shape[0] == 4


# ------------------------------------------------------------- Bézier fitting


def test_fit_straight_line_single_exact_segment() -> None:
    pts = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])
    chain = K.fit_bezier_chain(PolylineData(pts), max_error=0.1, corner_angle_deg=60.0)
    assert chain.control_points.shape[0] == 1
    np.testing.assert_allclose(chain.control_points[0, 0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(chain.control_points[0, 3], [10.0, 0.0], atol=1e-12)


def test_fit_quarter_circle_within_error() -> None:
    theta = np.linspace(0.0, np.pi / 2.0, 64)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    err = 0.01
    chain = K.fit_bezier_chain(PolylineData(pts), max_error=err, corner_angle_deg=60.0)
    assert chain.control_points.shape[0] <= 2  # a quarter circle is 1–2 cubics
    # Independent resampling: every input point within the error bound. The
    # curve must be sampled densely enough that point-to-sample distance is
    # dominated by fit error, not sampling gap.
    t = np.linspace(0.0, 1.0, 1024)
    curve = np.vstack(
        [
            (1 - t)[:, None] ** 3 * seg[0]
            + 3 * t[:, None] * (1 - t)[:, None] ** 2 * seg[1]
            + 3 * t[:, None] ** 2 * (1 - t)[:, None] * seg[2]
            + t[:, None] ** 3 * seg[3]
            for seg in chain.control_points
        ]
    )
    dists = np.min(np.linalg.norm(pts[:, None, :] - curve[None, :, :], axis=2), axis=1)
    assert float(dists.max()) <= err * 1.2  # small slack for residual sampling gap


def test_fit_corner_produces_c0_break() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
    chain = K.fit_bezier_chain(PolylineData(pts), max_error=0.01, corner_angle_deg=60.0)
    assert chain.control_points.shape[0] >= 2
    np.testing.assert_allclose(chain.control_points[0, 3], [2.0, 0.0], atol=1e-12)


def test_fit_endpoints_exact() -> None:
    rng = np.random.default_rng(5)
    pts = np.cumsum(rng.uniform(0.1, 1.0, (30, 2)), axis=0)
    chain = K.fit_bezier_chain(PolylineData(pts), max_error=0.5, corner_angle_deg=60.0)
    np.testing.assert_array_equal(chain.control_points[0, 0], pts[0])
    np.testing.assert_array_equal(chain.control_points[-1, 3], pts[-1])


# --------------------------------------------------------- polylabel & areas


def test_pole_of_square_is_center() -> None:
    square = PolylineData(
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]), is_closed=True
    )
    pole, r = K.pole_of_inaccessibility(square)
    assert (pole.x, pole.y) == pytest.approx((5.0, 5.0), abs=0.05)
    assert r == pytest.approx(5.0, abs=0.05)
    assert K.inscribed_circle_diameter(square) == pytest.approx(10.0, abs=0.1)


def test_pole_of_rectangle_radius_is_half_width() -> None:
    rect = PolylineData(
        np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 4.0], [0.0, 4.0]]), is_closed=True
    )
    _, r = K.pole_of_inaccessibility(rect)
    assert r == pytest.approx(2.0, abs=0.05)


def test_pole_of_concave_c_shape_lies_inside() -> None:
    c_shape = PolylineData(
        np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 3.0],
                [3.0, 3.0],
                [3.0, 7.0],
                [10.0, 7.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ]
        ),
        is_closed=True,
    )
    pole, r = K.pole_of_inaccessibility(c_shape)
    assert r > 0.0
    assert pole.x < 3.0 + r  # inside the C's spine, not in the notch


def test_watertight_accepts_partition_and_rejects_gap() -> None:
    labels = np.random.default_rng(6).integers(0, 3, (5, 5)).astype(np.int32)
    loops = K.trace_cracks(labels)
    assert K.is_watertight(loops, page_area=25.0)
    assert not K.is_watertight(loops[:-1], page_area=25.0)
    assert not K.is_watertight(loops, page_area=24.0)
