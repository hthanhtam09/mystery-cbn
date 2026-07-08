"""Benchmarks for the Connected Components stage (budgets: labeling ≤ 0.2 s,
graph build ≤ 0.3 s at 1600 px, ENGINE_SPEC §9–§10)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph, label_components

RNG = np.random.default_rng(0)
PROV = Provenance("denoise", "1.0.0", "0" * 64, "1" * 64)

_PALETTE = Palette(
    colors=tuple(PaletteColor.from_lab(i, (5.0 + 6.0 * i, 0.0, 0.0), 1000) for i in range(16)),
    provenance=PROV,
)

# Realistic post-denoise map at 1600 px: coherent blocks + residual speckle,
# giving a region count in the pre-merge thousands.
_BASE = np.repeat(np.repeat(RNG.integers(0, 16, (75, 100)), 16, axis=0), 16, axis=1).astype(
    np.int32
)
_NOISE = RNG.random(_BASE.shape) < 0.005
_BASE[_NOISE] = RNG.integers(0, 16, int(_NOISE.sum()))
_LABEL_MAP = LabelMap(labels=_BASE, provenance=PROV)


def test_bench_label_components_1600(benchmark: Any) -> None:
    cmap = benchmark(label_components, _BASE)
    assert cmap.shape == _BASE.shape


def test_bench_region_graph_1600_k16(benchmark: Any) -> None:
    graph = benchmark(build_region_graph, _LABEL_MAP, _PALETTE)
    assert graph.component_map.shape == _BASE.shape
    assert len(graph.regions) >= 1
