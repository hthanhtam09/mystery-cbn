"""Benchmarks for the Tiny Region Merge stage (budget: 20 000 → ~800 regions
at 1600 px ≤ 1.0 s, ENGINE_SPEC §11)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.merge import merge_tiny_regions

RNG = np.random.default_rng(0)
PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)

_PALETTE = Palette(
    colors=tuple(PaletteColor.from_lab(i, (5.0 + 6.0 * i, 0.0, 0.0), 1000) for i in range(16)),
    provenance=PROV,
)

# Heavy speckle on coherent blocks at 1600 px: tens of thousands of pre-merge
# regions, matching the noisy-input worst case of ENGINE_SPEC §9/§11.
_BASE = np.repeat(np.repeat(RNG.integers(0, 16, (75, 100)), 16, axis=0), 16, axis=1).astype(
    np.int32
)
_NOISE = RNG.random(_BASE.shape) < 0.02
_BASE[_NOISE] = RNG.integers(0, 16, int(_NOISE.sum()))
_GRAPH = build_region_graph(LabelMap(labels=_BASE, provenance=PROV), _PALETTE)
_A_MIN = 300.0  # drives region count from ~40k to the ~800-region regime


def test_bench_merge_tiny_1600_k16(benchmark: Any) -> None:
    merged, _, _ = benchmark(merge_tiny_regions, _GRAPH, _PALETTE, a_min=_A_MIN)
    assert len(merged.regions) < len(_GRAPH.regions)
    assert all(r.area_px >= _A_MIN for r in merged.regions)
