"""Benchmarks for Minimum Gap Enforcement / Gap Repair (Sprint 36B.3;
GAP_REPAIR_DESIGN.md §7, §10).

Two rows, matching the design's own complexity split:

- **Average case**: many arcs at low boundary density (the engine's
  standing assumption, MATH_SPEC §8.2 -- arcs disjoint except at
  junctions), where broad-phase candidate generation stays O(A).
- **Worst case**: many arcs densely clustered in a small region (fine
  parallel hatching), where candidate count degrades toward O(A^2) -- a
  named, tracked concern (GAP_REPAIR_DESIGN.md §7), not assumed away.

Reference budget: ``merge_tiny``'s documented "20 000 -> ~800 regions at
1600 px <= 1.0 s" (ENGINE_SPEC §11) is the comparable-scale precedent this
module doc points to; no fixed budget is asserted here yet pending
real-fixture arc-count measurement (as GAP_REPAIR_DESIGN.md §10 notes),
but both benchmarks assert the run completes and produces a well-formed
result, so a future regression shows up as a benchmark-tracked timing
delta even before a hard budget is set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.vector import Arc
from mysterycbn.stages.vector.geometry_normalize import GeometryNormalizeConfig, _minimum_gap_enforcement

_MM_TO_PT = 72.0 / 25.4


def _sparse_arcs(n_arcs: int, *, seed: int = 0) -> tuple[Arc, ...]:
    """``n_arcs`` short arcs scattered far apart across a large page --
    low boundary density, the engine's standing assumption for broad-phase
    O(A) candidate generation."""
    rng = np.random.default_rng(seed)
    arcs = []
    for i in range(n_arcs):
        x0 = rng.uniform(0.0, 5000.0)
        y0 = rng.uniform(0.0, 5000.0)
        pts = np.array(
            [[x0, y0], [x0 + 10.0, y0 + 1.0], [x0 + 20.0, y0]], dtype=np.float64
        )
        arcs.append(Arc(arc_id=i, points=pts, left_region=2 * i, right_region=2 * i + 1))
    return tuple(arcs)


def _dense_parallel_arcs(n_arcs: int, *, seed: int = 0) -> tuple[Arc, ...]:
    """``n_arcs`` short parallel arcs packed into a small region -- the
    worst-case density scenario named in GAP_REPAIR_DESIGN.md §7 (fine
    parallel hatching), where broad-phase candidate count degrades toward
    O(A^2)."""
    rng = np.random.default_rng(seed)
    arcs = []
    for i in range(n_arcs):
        y = i * 0.05  # tightly packed lines, well within a shared bbox
        x_jitter = rng.uniform(-0.5, 0.5)
        pts = np.array(
            [[0.0 + x_jitter, y], [50.0 + x_jitter, y + 0.5], [100.0 + x_jitter, y]],
            dtype=np.float64,
        )
        arcs.append(Arc(arc_id=i, points=pts, left_region=2 * i, right_region=2 * i + 1))
    return tuple(arcs)


_CONFIG = GeometryNormalizeConfig({"min_gap_mm": 0.1}, simplify_tolerance_mm=0.15)


def test_bench_gap_enforcement_sparse_2000_arcs(benchmark: Any) -> None:
    arcs = _sparse_arcs(2000)
    out, _repaired = benchmark(_minimum_gap_enforcement, arcs, config=_CONFIG)
    assert len(out) == len(arcs)


def test_bench_gap_enforcement_dense_parallel_300_arcs(benchmark: Any) -> None:
    """Worst-case density row (GAP_REPAIR_DESIGN.md §7): kept at a
    smaller arc count than the sparse benchmark since O(A^2) candidate
    growth makes this the actual bottleneck case to track."""
    arcs = _dense_parallel_arcs(300)
    out, _repaired = benchmark(_minimum_gap_enforcement, arcs, config=_CONFIG)
    assert len(out) == len(arcs)
