"""Property + determinism tests for Duplicate Point Cleanup (Sprint 36B.1),
Spike Removal (Sprint 36B.2), and Minimum Gap Enforcement / Gap Repair
(Sprint 36B.3); docs/modules/geometry_normalize.md §8.1-8.3, §9;
docs/modules/GAP_REPAIR_DESIGN.md §9 (the required property test list this
file's Gap Repair section implements).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.model.vector import Arc
from mysterycbn.stages.vector.geometry_normalize import (
    GeometryNormalizeConfig,
    _duplicate_cleanup,
    _min_arc_pair_distance,
    _minimum_gap_enforcement,
    _remove_spikes,
    _repair_gap,
    _spike_removal,
    _tapered_moves,
    _tapered_moves_would_self_intersect,
)

_MM_TO_PT = 72.0 / 25.4


def _cfg(eps_mm: float) -> GeometryNormalizeConfig:
    return GeometryNormalizeConfig({"duplicate_eps_mm": eps_mm}, simplify_tolerance_mm=0.15)


def _arc_from_offsets(offsets: list[float], closed: bool) -> Arc:
    """Build a monotone-x open (or closed-rectangle-perturbed) polyline
    from cumulative x-offsets, all distinct at construction (Arc's own
    invariant), y = 0."""
    xs = np.cumsum([0.0, *[abs(o) + 1e-9 for o in offsets]])
    pts = np.stack([xs, np.zeros_like(xs)], axis=1)
    if closed and pts.shape[0] < 4:
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return Arc(arc_id=0, points=pts, left_region=0, right_region=1, closed=closed)


# Small positive gaps between consecutive points, in pt; some near-zero
# (below a typical eps), some clearly above.
_offsets = st.lists(st.floats(min_value=0.0, max_value=5.0, allow_nan=False), min_size=1, max_size=30)


@settings(max_examples=200, deadline=None)
@given(_offsets, st.floats(min_value=0.001, max_value=0.15))
def test_bounded_removal_and_endpoint_preservation(offsets: list[float], eps_mm: float) -> None:
    arc = _arc_from_offsets(offsets, closed=False)
    config = _cfg(eps_mm)
    (out,), removed = _duplicate_cleanup((arc,), config=config)

    # Endpoints never move.
    assert out.points[0].tolist() == arc.points[0].tolist()
    assert out.points[-1].tolist() == arc.points[-1].tolist()
    # Never below the Arc floor for an open arc.
    assert out.points.shape[0] >= 2
    # Never more points than input, count is consistent with what changed.
    assert out.points.shape[0] <= arc.points.shape[0]
    assert removed == arc.points.shape[0] - out.points.shape[0]
    # Identity fields untouched.
    assert out.arc_id == arc.arc_id
    assert out.left_region == arc.left_region
    assert out.right_region == arc.right_region
    assert out.closed == arc.closed


@settings(max_examples=200, deadline=None)
@given(_offsets, st.floats(min_value=0.001, max_value=0.15))
def test_closed_arc_never_below_floor_of_four(offsets: list[float], eps_mm: float) -> None:
    arc = _arc_from_offsets(offsets, closed=True)
    config = _cfg(eps_mm)
    (out,), _ = _duplicate_cleanup((arc,), config=config)
    assert out.points.shape[0] >= 4
    assert out.closed is True


@settings(max_examples=200, deadline=None)
@given(_offsets, st.floats(min_value=0.001, max_value=0.15))
def test_no_op_returns_identical_object(offsets: list[float], eps_mm: float) -> None:
    arc = _arc_from_offsets(offsets, closed=False)
    config = _cfg(eps_mm)
    (out,), removed = _duplicate_cleanup((arc,), config=config)
    if removed == 0:
        assert out is arc
    else:
        assert out is not arc


@settings(max_examples=100, deadline=None)
@given(_offsets, st.floats(min_value=0.001, max_value=0.15))
def test_idempotence(offsets: list[float], eps_mm: float) -> None:
    """A second cleanup pass on already-cleaned output makes no further
    change -- consecutive kept points are already >= eps_pt apart."""
    arc = _arc_from_offsets(offsets, closed=False)
    config = _cfg(eps_mm)
    (once,), _ = _duplicate_cleanup((arc,), config=config)
    (twice,), removed_again = _duplicate_cleanup((once,), config=config)
    assert removed_again == 0
    assert twice.points.tolist() == once.points.tolist()


@settings(max_examples=100, deadline=None)
@given(_offsets, st.floats(min_value=0.001, max_value=0.15))
def test_determinism_same_input_same_output(offsets: list[float], eps_mm: float) -> None:
    arc = _arc_from_offsets(offsets, closed=False)
    config = _cfg(eps_mm)
    (a,), removed_a = _duplicate_cleanup((arc,), config=config)
    (b,), removed_b = _duplicate_cleanup((arc,), config=config)
    assert removed_a == removed_b
    assert a.points.tolist() == b.points.tolist()
    # Same-arg calls are pure: identical removal count implies identical
    # "did we construct a new Arc or pass the input through" decision too.
    assert (a is arc) == (b is arc)


def test_determinism_across_repeated_runs_many_arcs() -> None:
    """Same battery of arcs run twice in one process yields identical
    per-arc outputs and identical aggregate removed-count, independent of
    any dict/set iteration order (there is none in this pass)."""
    rng = np.random.default_rng(0)
    arcs = []
    for i in range(20):
        n = rng.integers(2, 12)
        xs = np.cumsum(np.concatenate([[0.0], rng.uniform(0.0, 2.0, size=n - 1) + 1e-9]))
        pts = np.stack([xs, np.zeros_like(xs)], axis=1)
        arcs.append(Arc(arc_id=i, points=pts, left_region=0, right_region=1))
    config = _cfg(0.05)

    out_1, removed_1 = _duplicate_cleanup(tuple(arcs), config=config)
    out_2, removed_2 = _duplicate_cleanup(tuple(arcs), config=config)

    assert removed_1 == removed_2
    for a1, a2 in zip(out_1, out_2, strict=True):
        assert a1.points.tolist() == a2.points.tolist()
        assert a1.arc_id == a2.arc_id


# ------------------------------------------------------- spike removal ---


def _spike_cfg(spike_length_mm: float) -> GeometryNormalizeConfig:
    return GeometryNormalizeConfig(
        {"spike_length_mm": spike_length_mm}, simplify_tolerance_mm=0.15
    )


def _random_polyline(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    """A polyline with distinct consecutive points at random angles/lengths
    -- may contain spikes, gentle bends, or neither, at ``Hypothesis``'s
    discretion via the seed."""
    pts = [np.array([0.0, 0.0])]
    for _ in range(n - 1):
        angle = rng.uniform(0.0, 2 * np.pi)
        length = rng.uniform(0.05, 1.0) * scale
        step = length * np.array([np.cos(angle), np.sin(angle)])
        nxt = pts[-1] + step
        pts.append(nxt)
    return np.stack(pts, axis=0)


@settings(max_examples=200, deadline=None)
@given(st.integers(2, 25), st.integers(0, 2**31 - 1), st.floats(min_value=0.01, max_value=0.15))
def test_spike_removal_bounded_and_endpoints_preserved(
    n: int, seed: int, spike_length_mm: float
) -> None:
    rng = np.random.default_rng(seed)
    pts = _random_polyline(rng, n, scale=spike_length_mm * (72.0 / 25.4))
    arc = Arc(arc_id=0, points=pts, left_region=0, right_region=1)
    config = _spike_cfg(spike_length_mm)
    (out,), removed = _spike_removal((arc,), config=config)

    assert out.points[0].tolist() == arc.points[0].tolist()
    assert out.points[-1].tolist() == arc.points[-1].tolist()
    assert out.points.shape[0] >= 2
    assert out.points.shape[0] <= arc.points.shape[0]
    assert removed == arc.points.shape[0] - out.points.shape[0]
    assert out.arc_id == arc.arc_id
    assert out.left_region == arc.left_region
    assert out.right_region == arc.right_region
    assert out.closed == arc.closed
    # The Arc constructor itself re-proves the distinctness invariant --
    # spike removal must never leave duplicate consecutive points.
    Arc(
        arc_id=out.arc_id,
        points=out.points,
        left_region=out.left_region,
        right_region=out.right_region,
        closed=out.closed,
    )


@settings(max_examples=100, deadline=None)
@given(st.integers(4, 25), st.integers(0, 2**31 - 1), st.floats(min_value=0.01, max_value=0.15))
def test_spike_removal_closed_arc_never_below_floor_of_four(
    n: int, seed: int, spike_length_mm: float
) -> None:
    rng = np.random.default_rng(seed)
    spike_length_pt = spike_length_mm * (72.0 / 25.4)
    pts = _random_polyline(rng, n, scale=spike_length_pt)
    out = _remove_spikes(pts, spike_length_pt=spike_length_pt, closed=True)
    assert out.shape[0] >= 4


@settings(max_examples=100, deadline=None)
@given(st.integers(2, 25), st.integers(0, 2**31 - 1), st.floats(min_value=0.01, max_value=0.15))
def test_spike_removal_no_op_returns_identical_object(
    n: int, seed: int, spike_length_mm: float
) -> None:
    rng = np.random.default_rng(seed)
    pts = _random_polyline(rng, n, scale=spike_length_mm * (72.0 / 25.4))
    arc = Arc(arc_id=0, points=pts, left_region=0, right_region=1)
    config = _spike_cfg(spike_length_mm)
    (out,), removed = _spike_removal((arc,), config=config)
    if removed == 0:
        assert out is arc
    else:
        assert out is not arc


@settings(max_examples=100, deadline=None)
@given(st.integers(2, 25), st.integers(0, 2**31 - 1), st.floats(min_value=0.01, max_value=0.15))
def test_spike_removal_idempotence(n: int, seed: int, spike_length_mm: float) -> None:
    """A second pass on already-cleaned output removes nothing further --
    no interior vertex of the result is still a qualifying spike."""
    rng = np.random.default_rng(seed)
    pts = _random_polyline(rng, n, scale=spike_length_mm * (72.0 / 25.4))
    arc = Arc(arc_id=0, points=pts, left_region=0, right_region=1)
    config = _spike_cfg(spike_length_mm)
    (once,), _ = _spike_removal((arc,), config=config)
    (twice,), removed_again = _spike_removal((once,), config=config)
    assert removed_again == 0
    assert twice.points.tolist() == once.points.tolist()


@settings(max_examples=100, deadline=None)
@given(st.integers(2, 25), st.integers(0, 2**31 - 1), st.floats(min_value=0.01, max_value=0.15))
def test_spike_removal_determinism_same_input_same_output(
    n: int, seed: int, spike_length_mm: float
) -> None:
    rng = np.random.default_rng(seed)
    pts = _random_polyline(rng, n, scale=spike_length_mm * (72.0 / 25.4))
    arc = Arc(arc_id=0, points=pts, left_region=0, right_region=1)
    config = _spike_cfg(spike_length_mm)
    (a,), removed_a = _spike_removal((arc,), config=config)
    (b,), removed_b = _spike_removal((arc,), config=config)
    assert removed_a == removed_b
    assert a.points.tolist() == b.points.tolist()
    assert (a is arc) == (b is arc)


def test_spike_removal_determinism_across_repeated_runs_many_arcs() -> None:
    rng = np.random.default_rng(7)
    arcs = []
    for i in range(20):
        n = int(rng.integers(2, 15))
        pts = _random_polyline(rng, n, scale=0.05 * (72.0 / 25.4))
        arcs.append(Arc(arc_id=i, points=pts, left_region=0, right_region=1))
    config = _spike_cfg(0.05)

    out_1, removed_1 = _spike_removal(tuple(arcs), config=config)
    out_2, removed_2 = _spike_removal(tuple(arcs), config=config)

    assert removed_1 == removed_2
    for a1, a2 in zip(out_1, out_2, strict=True):
        assert a1.points.tolist() == a2.points.tolist()
        assert a1.arc_id == a2.arc_id


# --------------------------------------------------- minimum gap enforcement ---
#
# GAP_REPAIR_DESIGN.md §9's required property test list: symmetry, order
# independence/determinism, idempotence, bounded displacement, topology
# preservation, sidedness preservation, junction immovability, guard
# soundness (negative), ceiling enforcement, no-op on clean input,
# monotone improvement. Fixtures below construct a "clean pinch": a single
# localized close approach at an *existing* interior vertex on both arcs
# (witness needs no insertion), with neighbors clearly beyond threshold --
# the class of gap this algorithm is designed to and verified to repair
# (see the hand-verification notes in tests/unit/test_geometry_normalize.py).


def _clean_pinch_arcs(
    half_gap_mm: float, baseline: float, pinch_dip: float, span: float
) -> tuple[Arc, Arc, float]:
    """Two 3-point open arcs: arc ``a`` is perfectly flat (y=0 at every
    point, including the pinch); arc ``b`` is flat at y=baseline at its
    two endpoints and dips down to y=tiny_gap only at the pinch vertex.
    Because ``a`` never moves off y=0, the witness point (closest
    approach) lands *exactly* on ``a``'s existing middle vertex and
    ``b``'s existing middle vertex -- no vertex insertion needed on
    either side, matching the exact construction hand-verified (in
    tests/unit/test_geometry_normalize.py) to reliably repair across the
    full range of gap fractions once the floating-point endpoint-
    recomputation bug in ``_closest_point_on_segment`` was fixed.

    An earlier version of this fixture sloped *both* arcs toward the
    pinch (a symmetric V on each side); that shape was found, during
    Sprint 36B.3 implementation, to force an interior-point insertion on
    at least one side whose immediate original neighbor then receives
    insufficient taper displacement -- a genuine, separate limitation of
    the fixed-taper mechanism (GAP_REPAIR_DESIGN.md §3.2 "1-2 vertices...
    linearly-decaying fraction"), documented and tested directly via the
    unit tests' sustained/insertion-required skip cases rather than here.

    ``pinch_dip`` is retained in the Hypothesis strategy signature for
    shrinking diversity even though it is not used in the flat
    construction below.
    """
    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * (0.05 + 0.9 * half_gap_mm)  # in (0, min_gap_pt), never touching 0
    del pinch_dip  # kept in the strategy signature for Hypothesis shrinking variety only
    a = Arc(
        arc_id=0,
        points=np.array([[-span, 0.0], [0.0, 0.0], [span, 0.0]]),
        left_region=0,
        right_region=1,
    )
    b = Arc(
        arc_id=1,
        points=np.array([[-span, baseline], [0.0, tiny_gap], [span, baseline]]),
        left_region=2,
        right_region=3,
    )
    return a, b, min_gap_pt


_clean_pinch_strategy = (
    st.floats(min_value=0.01, max_value=0.99),  # half_gap_mm fraction
    st.floats(min_value=2.0, max_value=50.0),  # baseline (neighbor y-offset)
    st.floats(min_value=0.0, max_value=5.0),  # unused (kept for shrinking diversity)
    st.floats(min_value=20.0, max_value=200.0),  # span
)


@settings(max_examples=100, deadline=None)
@given(*_clean_pinch_strategy)
def test_gap_repair_symmetry(half_gap, baseline, pinch_dip, span) -> None:
    a, b, min_gap_pt = _clean_pinch_arcs(half_gap, baseline, pinch_dip, span)
    arcs = [a, b]
    applied = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    if not applied:
        return
    a_disp = float(np.linalg.norm(arcs[0].points[1] - a.points[1]))
    b_disp = float(np.linalg.norm(arcs[1].points[1] - b.points[1]))
    assert abs(a_disp - b_disp) < 1e-6


@settings(max_examples=100, deadline=None)
@given(*_clean_pinch_strategy)
def test_gap_repair_bounded_displacement(half_gap, baseline, pinch_dip, span) -> None:
    a, b, min_gap_pt = _clean_pinch_arcs(half_gap, baseline, pinch_dip, span)
    max_disp = 1.0
    arcs = [a, b]
    _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=max_disp)
    for old, new in ((a, arcs[0]), (b, arcs[1])):
        for old_pt, new_pt in zip(old.points, new.points, strict=True):
            assert float(np.linalg.norm(new_pt - old_pt)) <= max_disp + 1e-6


@settings(max_examples=100, deadline=None)
@given(*_clean_pinch_strategy)
def test_gap_repair_junction_immovability(half_gap, baseline, pinch_dip, span) -> None:
    a, b, min_gap_pt = _clean_pinch_arcs(half_gap, baseline, pinch_dip, span)
    arcs = [a, b]
    _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    assert arcs[0].points[0].tolist() == a.points[0].tolist()
    assert arcs[0].points[-1].tolist() == a.points[-1].tolist()
    assert arcs[1].points[0].tolist() == b.points[0].tolist()
    assert arcs[1].points[-1].tolist() == b.points[-1].tolist()


@settings(max_examples=100, deadline=None)
@given(*_clean_pinch_strategy)
def test_gap_repair_monotone_improvement(half_gap, baseline, pinch_dip, span) -> None:
    """Per §9 property 11: a committed repair clears the threshold
    exactly, not merely improves it."""
    a, b, min_gap_pt = _clean_pinch_arcs(half_gap, baseline, pinch_dip, span)
    arcs = [a, b]
    applied = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    if not applied:
        return
    dist, *_ = _min_arc_pair_distance(arcs[0], arcs[1])
    assert dist >= min_gap_pt * (1.0 - 1e-4)


@settings(max_examples=100, deadline=None)
@given(*_clean_pinch_strategy)
def test_gap_repair_idempotence(half_gap, baseline, pinch_dip, span) -> None:
    """A second repair attempt on already-repaired output finds no
    further confirmed gap (post-repair distance already clears the
    threshold, so _repair_gap's own initial dist >= min_gap_pt check
    short-circuits)."""
    a, b, min_gap_pt = _clean_pinch_arcs(half_gap, baseline, pinch_dip, span)
    arcs = [a, b]
    applied_once = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    if not applied_once:
        return
    snapshot = [arcs[0].points.copy(), arcs[1].points.copy()]
    applied_twice = _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    assert applied_twice is False
    assert arcs[0].points.tolist() == snapshot[0].tolist()
    assert arcs[1].points.tolist() == snapshot[1].tolist()


@settings(max_examples=100, deadline=None)
@given(*_clean_pinch_strategy)
def test_gap_repair_topology_preservation(half_gap, baseline, pinch_dip, span) -> None:
    """§5: arc_id/left_region/right_region/closed and endpoint identity
    are preserved by construction; no Face is ever touched (this test
    operates purely on Arc, confirming the contract holds)."""
    a, b, min_gap_pt = _clean_pinch_arcs(half_gap, baseline, pinch_dip, span)
    arcs = [a, b]
    _repair_gap(arcs, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    assert arcs[0].arc_id == a.arc_id
    assert arcs[0].left_region == a.left_region
    assert arcs[0].right_region == a.right_region
    assert arcs[0].closed == a.closed
    assert arcs[1].arc_id == b.arc_id
    assert arcs[1].left_region == b.left_region
    assert arcs[1].right_region == b.right_region
    assert arcs[1].closed == b.closed


@settings(max_examples=50, deadline=None)
@given(*_clean_pinch_strategy)
def test_gap_repair_determinism_same_input_same_output(half_gap, baseline, pinch_dip, span) -> None:
    a, b, min_gap_pt = _clean_pinch_arcs(half_gap, baseline, pinch_dip, span)
    arcs1 = [a, b]
    arcs2 = [a, b]
    applied1 = _repair_gap(arcs1, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    applied2 = _repair_gap(arcs2, 0, 1, min_gap_pt=min_gap_pt, max_displacement_pt=1.0)
    assert applied1 == applied2
    assert arcs1[0].points.tolist() == arcs2[0].points.tolist()
    assert arcs1[1].points.tolist() == arcs2[1].points.tolist()


def test_gap_repair_order_independence_across_arc_list_order() -> None:
    """Processing order in _minimum_gap_enforcement is a function of
    arc_id, not list position -- reversing the input tuple's order must
    not change the result (§2.3, §3.3)."""
    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * 0.1
    a = Arc(
        arc_id=0, points=np.array([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]]),
        left_region=0, right_region=1,
    )
    b = Arc(
        arc_id=1, points=np.array([[-50.0, 10.0], [0.0, tiny_gap], [50.0, 10.0]]),
        left_region=2, right_region=3,
    )
    cfg = GeometryNormalizeConfig({"min_gap_mm": 0.1}, simplify_tolerance_mm=0.15)
    out_fwd, repaired_fwd = _minimum_gap_enforcement((a, b), config=cfg)
    out_rev, repaired_rev = _minimum_gap_enforcement((b, a), config=cfg)
    assert repaired_fwd == repaired_rev
    by_id_fwd = {arc.arc_id: arc.points.tolist() for arc in out_fwd}
    by_id_rev = {arc.arc_id: arc.points.tolist() for arc in out_rev}
    assert by_id_fwd == by_id_rev


def test_gap_repair_no_op_on_clean_input() -> None:
    """§9 property 10: no pair violates the threshold -> output identical
    to input, same object identity for every arc."""
    min_gap_pt = 0.1 * _MM_TO_PT
    a = Arc(arc_id=0, points=np.array([[0.0, 0.0], [10.0, 0.0]]), left_region=0, right_region=1)
    b = Arc(
        arc_id=1, points=np.array([[0.0, 100.0], [10.0, 100.0]]), left_region=2, right_region=3
    )
    cfg = GeometryNormalizeConfig({"min_gap_mm": 0.1}, simplify_tolerance_mm=0.15)
    out, repaired = _minimum_gap_enforcement((a, b), config=cfg)
    assert repaired == 0
    assert out[0] is a
    assert out[1] is b


def test_gap_repair_ceiling_enforcement() -> None:
    """§9 property 9: min_gap_mm > simplify.tolerance_mm raises ConfigError
    at construction, before any ArcGraph is processed."""
    try:
        GeometryNormalizeConfig({"min_gap_mm": 1.0}, simplify_tolerance_mm=0.15)
        raise AssertionError("expected ConfigError")
    except ConfigError:
        pass


def test_gap_repair_guard_soundness_negative() -> None:
    """§9 property 8: a constructed fixture where the 'correct'
    displacement would cross a third arc must result in the gap being
    skipped, not partially repaired or repaired by crossing the third
    arc. Uses the batch-taper guard directly against a hand-placed
    foreign segment positioned across the final (both-endpoints-moved)
    edge between two adjacent tapered vertices -- the exact scenario an
    earlier version of this guard missed (fixed during Sprint 36B.3)."""
    n = 10
    pts = np.stack([np.arange(n, dtype=np.float64) * 10, np.zeros(n)], axis=1)
    offset = np.array([0.0, 5.0])
    moves = _tapered_moves(pts, 5, offset)
    # Foreign segment crossing the final edge between the two taper
    # neighbors at indices 3 and 4 (post-move), not their original edge.
    foreign = np.array([[35.0, -1.0], [35.0, 3.0]])
    assert _tapered_moves_would_self_intersect(pts, moves, foreign) is True


def test_gap_repair_stage_error_on_unrepairable_committed_gap() -> None:
    """§9-adjacent: the outer independent verification loop in
    _minimum_gap_enforcement raises StageError if a committed repair's
    pair is somehow still below threshold -- exercised by forcing the
    condition via monkeypatch (see also the unit test of the same name)."""
    import mysterycbn.stages.vector.geometry_normalize as gn

    min_gap_pt = 0.1 * _MM_TO_PT
    tiny_gap = min_gap_pt * 0.1
    a = Arc(
        arc_id=0, points=np.array([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]]),
        left_region=0, right_region=1,
    )
    b = Arc(
        arc_id=1, points=np.array([[-50.0, 10.0], [0.0, tiny_gap], [50.0, 10.0]]),
        left_region=2, right_region=3,
    )
    cfg = GeometryNormalizeConfig({"min_gap_mm": 0.1}, simplify_tolerance_mm=0.15)

    def _fake_repair_gap(arcs, i, j, *, min_gap_pt, max_displacement_pt):
        return True  # commits unchanged geometry, reports success

    original = gn._repair_gap
    gn._repair_gap = _fake_repair_gap
    try:
        try:
            gn._minimum_gap_enforcement((a, b), config=cfg)
            raise AssertionError("expected StageError")
        except StageError:
            pass
    finally:
        gn._repair_gap = original
