"""Benchmarks for the Label Placement stage (budget: ≤ 1.0 s for F = 800,
ENGINE_SPEC §19/§26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import place_labels
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph

RNG = np.random.default_rng(0)
PROV = Provenance("bezier", "1.0.0", "0" * 64, "1" * 64)

_PALETTE = Palette(
    colors=tuple(PaletteColor.from_lab(i, (5.0 + 6.0 * i, 0.0, 0.0), 1000) for i in range(16)),
    provenance=PROV,
)

# ~800 faces on a letter page: 30×27 blocks of 16 distinct labels.
_BASE = np.repeat(np.repeat(RNG.integers(0, 16, (27, 30)), 12, axis=0), 12, axis=1).astype(np.int32)
_RG = build_region_graph(LabelMap(labels=_BASE, provenance=PROV), _PALETTE)
_CURVES = fit_curves(
    build_arc_graph(
        build_topology_graph(_RG.component_map),
        _RG,
        content_box=content_box_pt((215.9, 279.4, 12.7)),
    )
)
assert 600 <= len(_CURVES.faces) <= 1000


def test_bench_labels_800_faces(benchmark: Any) -> None:
    plan, findings = benchmark(place_labels, _CURVES, _RG)
    assert len(plan.labels) + len(findings) == len(_CURVES.faces)
