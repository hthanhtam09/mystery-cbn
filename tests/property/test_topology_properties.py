"""Property tests for the Topology Graph stage (ENGINE_SPEC §14)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mysterycbn.stages.graph.components import label_components
from mysterycbn.stages.vector.topology import build_topology_graph, validate_topology


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 9), st.integers(1, 9), st.integers(0, 2**31 - 1), st.integers(2, 4))
def test_topology_properties_on_random_maps(h: int, w: int, seed: int, k: int) -> None:
    labels = np.random.default_rng(seed).integers(0, k, (h, w)).astype(np.int32)
    cmap = label_components(labels)
    graph = build_topology_graph(cmap)
    # No gaps / no overlaps / adjacency consistency / Euler — the validator
    # is the property.
    validate_topology(graph, cmap)
    # Σ arc lengths = B (crack-edge count), directly.
    b = (
        int((cmap[:, :-1] != cmap[:, 1:]).sum())
        + int((cmap[:-1, :] != cmap[1:, :]).sum())
        + 2 * (h + w)
    )
    assert sum(len(a.points) - 1 for a in graph.arcs) == b
    # Open arc endpoints are junctions; closed arcs anchor at their
    # lexicographically smallest corner (per-axis min need not be a vertex).
    for arc in graph.arcs:
        if arc.closed:
            assert tuple(arc.points[0]) == min(map(tuple, arc.points.tolist()))
    # Determinism.
    assert [a.to_dict() for a in build_topology_graph(cmap).arcs] == [
        a.to_dict() for a in graph.arcs
    ]
