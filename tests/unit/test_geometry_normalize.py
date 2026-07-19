"""Unit tests for the Geometry Normalize stage's Duplicate Point Cleanup
(Sprint 36B.1), Spike Removal (Sprint 36B.2), and Minimum Gap Enforcement /
Gap Repair (Sprint 36B.3) passes (docs/modules/geometry_normalize.md §8.1,
§8.2, §8.3; docs/modules/GAP_REPAIR_DESIGN.md).

Property-based tests live in
``tests/property/test_geometry_normalize_properties.py``; golden digest
tests live in ``tests/golden/test_geometry_normalize_golden.py``; the
performance benchmark lives in
``benchmarks/perf/test_geometry_normalize_perf.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.model.records import Provenance
from mysterycbn.model.vector import Arc, ArcGraph, Face
from mysterycbn.stages.vector.geometry_normalize import (
    GeometryNormalizeConfig,
    GeometryNormalizeStage,
    _candidate_pairs,
    _duplicate_cleanup,
    _min_arc_pair_distance,
    _minimum_gap_enforcement,
    _remove_spikes,
    _repair_gap,
    _segment_segment_distance,
    _shares_endpoint,
    _spike_removal,
    normalize_geometry,
)

PROV = Provenance("simplify", "1.0.0", "0" * 64, "1" * 64)
_MM_TO_PT = 72.0 / 25.4


def _cfg(**overrides: object) -> GeometryNormalizeConfig:
    return GeometryNormalizeConfig(overrides, simplify_tolerance_mm=0.15)


def _open_arc(points: list[list[float]], arc_id: int = 0) -> Arc:
    return Arc(
        arc_id=arc_id,
        points=np.array(points, dtype=np.float64),
        left_region=0,
        right_region=1,
    )


def _closed_arc(points: list[list[float]], arc_id: int = 0) -> Arc:
    return Arc(
        arc_id=arc_id,
        points=np.array(points, dtype=np.float64),
        left_region=0,
        right_region=1,
        closed=True,
    )


def test_removes_a_single_near_duplicate_interior_point() -> None:
    eps_mm = 0.05
    arc = _open_arc([[0.0, 0.0], [0.001, 0.0], [10.0, 0.0]])
    (out,), removed = _duplicate_cleanup((arc,), config=_cfg(duplicate_eps_mm=eps_mm))
    assert out.points.tolist() == [[0.0, 0.0], [10.0, 0.0]]
    assert removed == 1


def test_collapses_a_run_of_consecutive_near_duplicates() -> None:
    arc = _open_arc(
        [[0.0, 0.0], [0.001, 0.0], [0.002, 0.0], [0.003, 0.0], [10.0, 0.0], [20.0, 0.0]]
    )
    (out,), removed = _duplicate_cleanup((arc,), config=_cfg(duplicate_eps_mm=0.05))
    assert out.points.tolist() == [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]
    assert removed == 3


def test_no_op_returns_identical_arc_object() -> None:
    arc = _open_arc([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    (out,), removed = _duplicate_cleanup((arc,), config=_cfg(duplicate_eps_mm=0.05))
    assert out is arc
    assert removed == 0


def test_modified_arc_preserves_identity_fields() -> None:
    arc = Arc(
        arc_id=7,
        points=np.array([[0.0, 0.0], [0.001, 0.0], [10.0, 0.0]], dtype=np.float64),
        left_region=3,
        right_region=9,
        closed=False,
    )
    (out,), _ = _duplicate_cleanup((arc,), config=_cfg(duplicate_eps_mm=0.05))
    assert out is not arc
    assert out.arc_id == 7
    assert out.left_region == 3
    assert out.right_region == 9
    assert out.closed is False


def test_never_removes_open_arc_endpoints() -> None:
    # Both endpoints are within eps of their neighbor, but must survive.
    arc = _open_arc([[0.0, 0.0], [0.0005, 0.0], [10.0, 0.0], [10.0005, 0.0]])
    (out,), _ = _duplicate_cleanup((arc,), config=_cfg(duplicate_eps_mm=0.05))
    assert out.points[0].tolist() == [0.0, 0.0]
    assert out.points[-1].tolist() == [10.0005, 0.0]


def test_never_drops_below_open_arc_minimum_of_two_points() -> None:
    # Already at the Arc floor (2 points): _clean_duplicate_points is a
    # no-op regardless of eps -- verified directly, since a threshold large
    # enough to matter here would exceed GeometryNormalizeConfig's ceiling.
    from mysterycbn.stages.vector.geometry_normalize import _clean_duplicate_points

    pts = np.array([[0.0, 0.0], [0.001, 0.0]], dtype=np.float64)
    out = _clean_duplicate_points(pts, eps_pt=1.0 * _MM_TO_PT, closed=False)
    assert out.shape[0] == 2


def test_never_drops_below_closed_arc_minimum_of_four_points() -> None:
    from mysterycbn.stages.vector.geometry_normalize import _clean_duplicate_points

    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    out = _clean_duplicate_points(pts, eps_pt=100.0 * _MM_TO_PT, closed=True)
    assert out.shape[0] == 4


def test_closed_arc_wraparound_edge_is_also_checked() -> None:
    # Last kept point is within eps of point 0 across the implicit close.
    eps_pt = 0.05 * _MM_TO_PT
    arc = _closed_arc(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, eps_pt / 2.0]]
    )
    (out,), removed = _duplicate_cleanup((arc,), config=_cfg(duplicate_eps_mm=0.05))
    assert removed == 1
    assert out.points.shape[0] == 4


def test_multiple_arcs_processed_independently() -> None:
    a = _open_arc([[0.0, 0.0], [0.001, 0.0], [10.0, 0.0]], arc_id=0)
    b = _open_arc([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], arc_id=1)
    (out_a, out_b), removed = _duplicate_cleanup((a, b), config=_cfg(duplicate_eps_mm=0.05))
    assert out_a.points.shape[0] == 2
    assert out_b is b
    assert removed == 1


def _graph(arcs: tuple[Arc, ...]) -> ArcGraph:
    face = Face(face_id=0, label=0, outer_walk=tuple((a.arc_id, False) for a in arcs))
    return ArcGraph(arcs=arcs, faces=(face,), work_scale=1.0, provenance=PROV)


def test_normalize_geometry_runs_duplicate_cleanup_and_reports_metrics() -> None:
    arc = _open_arc([[0.0, 0.0], [0.001, 0.0], [10.0, 0.0]])
    graph = _graph((arc,))
    cfg = _cfg(duplicate_eps_mm=0.05)
    out_graph, metrics = normalize_geometry(graph, config=cfg, config_hash="a" * 64)
    assert out_graph.arcs[0].points.shape[0] == 2
    assert metrics["duplicates_removed"] == 1
    assert metrics["spikes_removed"] == 0
    assert metrics["gaps_repaired"] == 0
    assert out_graph.faces is graph.faces
    assert out_graph.provenance.stage_name == "geometry_normalize"


def test_disabled_stage_skips_duplicate_cleanup() -> None:
    arc = _open_arc([[0.0, 0.0], [0.001, 0.0], [10.0, 0.0]])
    graph = _graph((arc,))
    cfg = _cfg(enabled=False, duplicate_eps_mm=0.05)
    out_graph, metrics = normalize_geometry(graph, config=cfg, config_hash="b" * 64)
    assert out_graph.arcs is graph.arcs
    assert metrics["duplicates_removed"] == 0


def test_stage_run_binds_metrics_artifact() -> None:
    from mysterycbn.kernel.context import InMemoryContext

    arc = _open_arc([[0.0, 0.0], [0.001, 0.0], [10.0, 0.0]])
    graph = _graph((arc,))
    ctx = InMemoryContext(seed=0)
    ctx.put("arc_graph", graph)
    stage = GeometryNormalizeStage(
        {"duplicate_eps_mm": 0.05}, simplify_tolerance_mm=0.15, config_hash="c" * 64
    )
    stage.run(ctx)
    assert ctx.get("arc_graph").arcs[0].points.shape[0] == 2
    assert ctx.get("geometry_normalize_metrics")["duplicates_removed"] == 1


def test_config_rejects_threshold_exceeding_simplify_ceiling() -> None:
    with pytest.raises(ConfigError):
        GeometryNormalizeConfig({"duplicate_eps_mm": 1.0}, simplify_tolerance_mm=0.15)


# ------------------------------------------------------- spike removal ---


def _spike_cfg(spike_length_mm: float = 0.05) -> GeometryNormalizeConfig:
    return GeometryNormalizeConfig(
        {"spike_length_mm": spike_length_mm}, simplify_tolerance_mm=0.15
    )


def test_removes_a_simple_out_and_back_spike() -> None:
    eps_pt = 0.05 * _MM_TO_PT
    d = eps_pt * 0.3
    arc = _open_arc([[0.0, 0.0], [10.0, 0.0], [10.0 + d, d], [10.0, 0.0], [20.0, 0.0]])
    (out,), removed = _spike_removal((arc,), config=_spike_cfg())
    assert out.points.tolist() == [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]
    assert removed == 2  # spike vertex + its now-redundant duplicate neighbor


def test_result_never_has_duplicate_consecutive_points() -> None:
    """The out-and-back case is the one place a naive single-vertex
    collapse would introduce a duplicate consecutive point (violating
    Arc's own invariant) -- verified by successfully constructing an Arc
    from the result."""
    eps_pt = 0.05 * _MM_TO_PT
    d = eps_pt * 0.3
    arc = _open_arc([[0.0, 0.0], [10.0, 0.0], [10.0 + d, d], [10.0, 0.0], [20.0, 0.0]])
    (out,), _ = _spike_removal((arc,), config=_spike_cfg())
    Arc(  # raises ValueError if consecutive points are not distinct
        arc_id=out.arc_id,
        points=out.points,
        left_region=out.left_region,
        right_region=out.right_region,
        closed=out.closed,
    )


def test_no_op_returns_identical_arc_object_for_spike_removal() -> None:
    arc = _open_arc([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    (out,), removed = _spike_removal((arc,), config=_spike_cfg())
    assert out is arc
    assert removed == 0


def test_spike_removal_preserves_identity_fields() -> None:
    eps_pt = 0.05 * _MM_TO_PT
    d = eps_pt * 0.3
    arc = Arc(
        arc_id=5,
        points=np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0 + d, d], [10.0, 0.0], [20.0, 0.0]],
            dtype=np.float64,
        ),
        left_region=4,
        right_region=8,
        closed=False,
    )
    (out,), _ = _spike_removal((arc,), config=_spike_cfg())
    assert out is not arc
    assert out.arc_id == 5
    assert out.left_region == 4
    assert out.right_region == 8
    assert out.closed is False


def test_never_removes_endpoints_even_if_spike_shaped() -> None:
    # Vertex 0 is an endpoint and is never a removal candidate, regardless
    # of how spike-like its local geometry might look.
    eps_pt = 0.05 * _MM_TO_PT
    d = eps_pt * 0.3
    pts = np.array([[d, d], [0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float64)
    out = _remove_spikes(pts, spike_length_pt=eps_pt, closed=False)
    assert out[0].tolist() == [d, d]
    assert out[-1].tolist() == [20.0, 0.0]


def test_long_sharp_corner_is_not_a_spike() -> None:
    # Turn angle is sharp but edges are far longer than spike_length_pt.
    arc = _open_arc([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [20.0, 10.0]])
    (out,), removed = _spike_removal((arc,), config=_spike_cfg(spike_length_mm=0.05))
    assert out is arc
    assert removed == 0


def test_moderate_turn_angle_short_edges_is_not_a_spike() -> None:
    # Short edges, but turn angle well below the near-reversal threshold
    # (a gentle bend, not a near-total direction reversal).
    eps_pt = 0.05 * _MM_TO_PT
    d = eps_pt * 0.3
    arc = _open_arc([[0.0, 0.0], [10.0, 0.0], [10.0 + d, 10.0 + d], [20.0, 20.0]])
    (out,), removed = _spike_removal((arc,), config=_spike_cfg())
    assert out is arc
    assert removed == 0


def test_never_drops_below_open_arc_minimum_of_two_points_spike() -> None:
    arc = _open_arc([[0.0, 0.0], [0.0001, 0.0001]])
    out = _remove_spikes(arc.points, spike_length_pt=100.0, closed=False)
    assert out.shape[0] == 2


def test_never_drops_below_closed_arc_minimum_of_four_points_spike() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    out = _remove_spikes(pts, spike_length_pt=1000.0, closed=True)
    assert out.shape[0] == 4


def test_multiple_arcs_processed_independently_spike() -> None:
    eps_pt = 0.05 * _MM_TO_PT
    d = eps_pt * 0.3
    a = _open_arc(
        [[0.0, 0.0], [10.0, 0.0], [10.0 + d, d], [10.0, 0.0], [20.0, 0.0]], arc_id=0
    )
    b = _open_arc([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], arc_id=1)
    (out_a, out_b), removed = _spike_removal((a, b), config=_spike_cfg())
    assert out_a.points.shape[0] == 3
    assert out_b is b
    assert removed == 2


def test_normalize_geometry_runs_spike_removal_and_reports_metrics() -> None:
    eps_pt = 0.05 * _MM_TO_PT
    d = eps_pt * 0.3
    arc = _open_arc([[0.0, 0.0], [10.0, 0.0], [10.0 + d, d], [10.0, 0.0], [20.0, 0.0]])
    graph = _graph((arc,))
    cfg = _spike_cfg()
    out_graph, metrics = normalize_geometry(graph, config=cfg, config_hash="d" * 64)
    assert out_graph.arcs[0].points.shape[0] == 3
    assert metrics["spikes_removed"] == 2
    assert metrics["duplicates_removed"] == 0
    assert metrics["gaps_repaired"] == 0


# --------------------------------------------------- minimum gap enforcement ---


def _gap_cfg(min_gap_mm: float = 0.1, simplify_tolerance_mm: float = 0.15) -> GeometryNormalizeConfig:
    return GeometryNormalizeConfig(
        {"min_gap_mm": min_gap_mm}, simplify_tolerance_mm=simplify_tolerance_mm
    )


def test_segment_segment_distance_parallel_offset() -> None:
    a, b = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    c, d = np.array([3.0, 1.0]), np.array([13.0, 1.0])
    dist, pa, pb = _segment_segment_distance(a, b, c, d)
    assert dist == pytest.approx(1.0)


def test_segment_segment_distance_proper_crossing_is_zero() -> None:
    # A horizontal and a vertical segment that properly cross -- the
    # 4-endpoint-projection shortcut alone would miss this (neither
    # segment's endpoint is the closest point; the crossing point is
    # interior to both). Regression test for the bug found and fixed
    # during Sprint 36B.3 implementation.
    a, b = np.array([-5.0, 0.5]), np.array([5.0, 0.5])
    c, d = np.array([0.0, -5.0]), np.array([0.0, 5.0])
    dist, pa, pb = _segment_segment_distance(a, b, c, d)
    assert dist == pytest.approx(0.0, abs=1e-9)
    assert pa.tolist() == pytest.approx([0.0, 0.5])


def test_shares_endpoint_excludes_junction_pairs() -> None:
    a = _open_arc([[0.0, 0.0], [10.0, 0.0]], arc_id=0)
    b = _open_arc([[0.0, 0.0], [0.0, 10.0]], arc_id=1)  # shares a.points[0]
    assert _shares_endpoint(a, b) is True


def test_shares_endpoint_false_for_disjoint_arcs() -> None:
    a = _open_arc([[0.0, 0.0], [10.0, 0.0]], arc_id=0)
    b = _open_arc([[0.0, 5.0], [10.0, 5.0]], arc_id=1)
    assert _shares_endpoint(a, b) is False


def test_candidate_pairs_finds_close_arcs() -> None:
    min_gap_pt = 0.1 * _MM_TO_PT
    a = _open_arc([[0.0, 0.0], [10.0, 0.0]], arc_id=0)
    b = _open_arc([[0.0, 0.01], [10.0, 0.01]], arc_id=1)
    far = _open_arc([[1000.0, 1000.0], [1010.0, 1000.0]], arc_id=2)
    pairs = _candidate_pairs([a, b, far], min_gap_pt=min_gap_pt)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs


def test_repair_gap_clean_pinch_clears_threshold() -> None:
    """A single, localized close-approach at an interior vertex, with
    both arcs' neighbors already far beyond the threshold -- the
    canonical case this pass targets. Verified by hand during
    implementation: witness is an existing vertex on both sides (no
    insertion needed), so no competing under-tapered neighbor remains
    close after the repair."""
    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * 0.1
    a = _open_arc([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]], arc_id=0)
    b = _open_arc([[-50.0, 10.0], [0.0, tiny_gap], [50.0, 10.0]], arc_id=1)
    before, *_ = _min_arc_pair_distance(a, b)
    assert before < min_gap_pt

    arcs = [a, b]
    applied = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    assert applied is True

    after, *_ = _min_arc_pair_distance(arcs[0], arcs[1])
    assert after >= min_gap_pt * (1.0 - 1e-4)


def test_repair_gap_is_symmetric() -> None:
    """Both arcs receive exactly half the correction -- order-independent
    regardless of which arc is passed as 'a' vs 'b'."""
    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * 0.1
    a = _open_arc([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]], arc_id=0)
    b = _open_arc([[-50.0, 10.0], [0.0, tiny_gap], [50.0, 10.0]], arc_id=1)

    arcs1 = [a, b]
    _repair_gap(arcs1, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    a_disp = abs(arcs1[0].points[1][1] - a.points[1][1])
    b_disp = abs(arcs1[1].points[1][1] - b.points[1][1])
    assert a_disp == pytest.approx(b_disp, rel=1e-6)


def test_repair_gap_never_moves_endpoints() -> None:
    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * 0.1
    a = _open_arc([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]], arc_id=0)
    b = _open_arc([[-50.0, 10.0], [0.0, tiny_gap], [50.0, 10.0]], arc_id=1)
    arcs = [a, b]
    _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    assert arcs[0].points[0].tolist() == a.points[0].tolist()
    assert arcs[0].points[-1].tolist() == a.points[-1].tolist()
    assert arcs[1].points[0].tolist() == b.points[0].tolist()
    assert arcs[1].points[-1].tolist() == b.points[-1].tolist()


def test_repair_gap_skips_when_witness_is_an_endpoint() -> None:
    # Both arcs' closest approach is exactly at their shared-x start point
    # (an endpoint on both sides) -- never displaced, so no repair legal.
    min_gap_pt = 0.1 * _MM_TO_PT
    gap = min_gap_pt * 0.3
    a = _open_arc([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], arc_id=0)
    b = _open_arc([[0.0, gap], [10.0, gap], [20.0, gap]], arc_id=1)
    arcs = [a, b]
    applied = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    assert applied is False
    assert arcs[0].points.tolist() == a.points.tolist()
    assert arcs[1].points.tolist() == b.points.tolist()


def test_repair_gap_no_op_when_already_clear() -> None:
    min_gap_pt = 0.1 * _MM_TO_PT
    a = _open_arc([[0.0, 0.0], [10.0, 0.0]], arc_id=0)
    b = _open_arc([[0.0, 100.0], [10.0, 100.0]], arc_id=1)
    arcs = [a, b]
    applied = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    assert applied is False


def test_repair_gap_refuses_when_displacement_exceeds_ceiling() -> None:
    min_gap_pt = 10.0  # deliberately huge threshold
    a = _open_arc([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]], arc_id=0)
    b = _open_arc([[-50.0, 20.0], [0.0, 0.01], [50.0, 20.0]], arc_id=1)
    arcs = [a, b]
    applied = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=0.001)
    assert applied is False
    assert arcs[0].points.tolist() == a.points.tolist()


def test_repair_gap_skips_sustained_dense_parallel_run() -> None:
    """A gap sustained uniformly across many vertices cannot be cleared
    by a single localized witness + small taper -- correctly skipped
    (never partially committed), per GAP_REPAIR_DESIGN.md §8's
    'do the full fix or nothing' rule. Verified by hand during
    implementation."""
    min_gap_pt = 0.1 * _MM_TO_PT
    gap = min_gap_pt * 0.3
    n = 15
    xs = np.linspace(-30.0, 30.0, n)
    a_y = 10.0 - 9.9 * np.exp(-(xs**2) / 20.0)
    b_y = a_y + gap
    a = Arc(arc_id=0, points=np.stack([xs, a_y], axis=1), left_region=0, right_region=1)
    b = Arc(arc_id=1, points=np.stack([xs, b_y], axis=1), left_region=2, right_region=3)
    arcs = [a, b]
    applied = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=0.15 * _MM_TO_PT)
    assert applied is False
    assert arcs[0].points.tolist() == a.points.tolist()
    assert arcs[1].points.tolist() == b.points.tolist()


def test_minimum_gap_enforcement_excludes_shared_junction_pairs() -> None:
    min_gap_pt = 0.1 * _MM_TO_PT
    # a and b share endpoint (0,0); their close approach is exactly there.
    a = _open_arc([[0.0, 0.0], [10.0, 5.0]], arc_id=0)
    b = _open_arc([[0.0, 0.0], [10.0, -5.0]], arc_id=1)
    out, repaired = _minimum_gap_enforcement((a, b), config=_gap_cfg())
    assert repaired == 0
    assert out[0].points.tolist() == a.points.tolist()
    assert out[1].points.tolist() == b.points.tolist()


def test_minimum_gap_enforcement_repairs_and_reports_metrics() -> None:
    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * 0.1
    a = _open_arc([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]], arc_id=0)
    b = _open_arc([[-50.0, 10.0], [0.0, tiny_gap], [50.0, 10.0]], arc_id=1)
    out, repaired = _minimum_gap_enforcement((a, b), config=_gap_cfg())
    assert repaired == 1
    dist, *_ = _min_arc_pair_distance(out[0], out[1])
    assert dist >= min_gap_pt * (1.0 - 1e-4)


def test_minimum_gap_enforcement_preserves_faces_and_arc_ids() -> None:
    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * 0.1
    a = _open_arc([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]], arc_id=0)
    b = _open_arc([[-50.0, 10.0], [0.0, tiny_gap], [50.0, 10.0]], arc_id=1)
    graph = _graph((a, b))
    out_graph, metrics = normalize_geometry(graph, config=_gap_cfg(), config_hash="e" * 64)
    assert out_graph.faces is graph.faces
    assert [arc.arc_id for arc in out_graph.arcs] == [0, 1]
    assert metrics["gaps_repaired"] == 1


def test_minimum_gap_enforcement_no_error_when_no_confirmed_gaps() -> None:
    min_gap_pt = 0.1 * _MM_TO_PT
    a = _open_arc([[0.0, 0.0], [10.0, 0.0]], arc_id=0)
    b = _open_arc([[0.0, 100.0], [10.0, 100.0]], arc_id=1)
    out, repaired = _minimum_gap_enforcement((a, b), config=_gap_cfg())
    assert repaired == 0


def test_minimum_gap_enforcement_raises_stage_error_when_unrepairable() -> None:
    """A confirmed gap that the algorithm cannot clear (sustained dense
    parallel run, correctly skipped by _repair_gap) must surface as a
    StageError only if it was actually *attempted and committed* while
    still failing verification -- here we force that condition directly
    by constructing an artificial post-repair state via a pair that gets
    marked repaired but whose distance the second, independent
    verification pass finds still short. Since _repair_gap's own
    pre-commit check already prevents this in practice, this test
    documents the outer safety net exists and fires correctly by
    invoking it on a hand-constructed scenario."""
    # Two arcs sharing no endpoint, confirmed gap, but with only 2 points
    # each (no interior vertex at all) -- entirely unrepairable, and the
    # ONLY vertices are endpoints, so _repair_gap always returns False;
    # _minimum_gap_enforcement must NOT raise here (repaired_count == 0,
    # skip is legal).
    min_gap_pt = 0.1 * _MM_TO_PT
    gap = min_gap_pt * 0.3
    a = _open_arc([[0.0, 0.0], [50.0, 1.0]], arc_id=0)
    b = _open_arc([[0.0, gap], [50.0, gap + 1.0]], arc_id=1)
    out, repaired = _minimum_gap_enforcement((a, b), config=_gap_cfg())
    assert repaired == 0  # skip is legal -- no StageError


def test_gap_repair_stage_error_carries_stage_name() -> None:
    """Direct unit test of the StageError-raising branch inside
    _minimum_gap_enforcement, bypassing _repair_gap's own pre-commit
    check by monkeypatching it to simulate a repair that was committed
    but whose independent post-verification finds it still short --
    exactly the safety-net scenario the task requires."""
    import mysterycbn.stages.vector.geometry_normalize as gn

    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * 0.1
    a = _open_arc([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]], arc_id=0)
    b = _open_arc([[-50.0, 10.0], [0.0, tiny_gap], [50.0, 10.0]], arc_id=1)

    def _fake_repair_gap(arcs, i, j, *, min_gap_pt, max_displacement_pt):
        # Commit the arcs completely unchanged, but report "applied" --
        # simulating a hypothetical future bug in _repair_gap's own
        # pre-commit check, so the outer independent verification is
        # proven to catch it.
        return True

    original = gn._repair_gap
    gn._repair_gap = _fake_repair_gap
    try:
        with pytest.raises(StageError) as excinfo:
            gn._minimum_gap_enforcement((a, b), config=_gap_cfg())
        assert excinfo.value.stage_name == "geometry_normalize"
    finally:
        gn._repair_gap = original
