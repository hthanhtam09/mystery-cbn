"""Benchmarks for the Topology Graph stage (budget: ≤ 0.3 s at 1600 px,
ENGINE_SPEC §14/§26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.stages.graph.components import label_components
from mysterycbn.stages.vector.topology import build_topology_graph, validate_topology

RNG = np.random.default_rng(0)

# Post-merge map at 1600 px: §14 consumes the graph AFTER Tiny Region Merge
# (~800–1500 regions, ENGINE_SPEC §11 benchmark regime), so blocks are sized
# to land in that region count with no sub-floor speckle.
_BASE = np.repeat(np.repeat(RNG.integers(0, 16, (38, 50)), 32, axis=0), 32, axis=1).astype(np.int32)
_CMAP = label_components(_BASE)
assert 800 <= int(_CMAP.max()) + 1 <= 2000


def test_bench_topology_build_1600(benchmark: Any) -> None:
    graph = benchmark(build_topology_graph, _CMAP)
    b = (
        int((_CMAP[:, :-1] != _CMAP[:, 1:]).sum())
        + int((_CMAP[:-1, :] != _CMAP[1:, :]).sum())
        + 2 * sum(_CMAP.shape)
    )
    assert sum(len(a.points) - 1 for a in graph.arcs) == b


def test_bench_topology_validate_1600(benchmark: Any) -> None:
    graph = build_topology_graph(_CMAP)
    benchmark(validate_topology, graph, _CMAP)
