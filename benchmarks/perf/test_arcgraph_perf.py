"""Benchmarks for the Arc Graph stage (budget: ≤ 0.3 s for A ≤ 20 000,
ENGINE_SPEC §15/§26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.stages.graph.components import label_components
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.topology import build_topology_graph

RNG = np.random.default_rng(0)

# Post-merge map at 1600 px (same regime as the topology benchmark): ~1200
# regions, arc count well inside the A ≤ 20 000 contract.
_BASE = np.repeat(np.repeat(RNG.integers(0, 16, (38, 50)), 32, axis=0), 32, axis=1).astype(np.int32)
_CMAP = label_components(_BASE)
_BOX = content_box_pt((215.9, 279.4, 12.7))


def _region_graph():  # type: ignore[no-untyped-def]
    from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
    from mysterycbn.stages.graph.components import build_region_graph

    prov = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)
    palette = Palette(
        colors=tuple(PaletteColor.from_lab(i, (5.0 + 6.0 * i, 0.0, 0.0), 1000) for i in range(16)),
        provenance=prov,
    )
    return build_region_graph(LabelMap(labels=_BASE, provenance=prov), palette)


_REGION_GRAPH = _region_graph()
_TOPOLOGY = build_topology_graph(_CMAP)
assert len(_TOPOLOGY.arcs) <= 20_000


def test_bench_arcgraph_1600(benchmark: Any) -> None:
    graph = benchmark(build_arc_graph, _TOPOLOGY, _REGION_GRAPH, content_box=_BOX)
    assert len(graph.faces) == len(_REGION_GRAPH.regions)
