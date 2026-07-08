"""Benchmarks for the Noise Removal stage (budget: ≤ 0.5 s at 1600 px,
ENGINE_SPEC §26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.raster.denoise import denoise_label_map

RNG = np.random.default_rng(0)
PROV = Provenance("quantize", "1.0.0", "0" * 64, "1" * 64)

_PALETTE = Palette(
    colors=tuple(PaletteColor.from_lab(i, (5.0 + 6.0 * i, 0.0, 0.0), 1000) for i in range(16)),
    provenance=PROV,
)

# Realistic post-quantize map: coherent blocks + speckle noise.
_BASE = np.repeat(np.repeat(RNG.integers(0, 16, (75, 100)), 16, axis=0), 16, axis=1).astype(
    np.int32
)
_NOISE = RNG.random(_BASE.shape) < 0.02
_BASE[_NOISE] = RNG.integers(0, 16, int(_NOISE.sum()))
_LABEL_MAP = LabelMap(labels=_BASE, provenance=PROV)


def test_bench_denoise_1600_k16(benchmark: Any) -> None:
    out = benchmark(denoise_label_map, _LABEL_MAP, _PALETTE, speck_px=10)
    assert out.labels.shape == _BASE.shape
