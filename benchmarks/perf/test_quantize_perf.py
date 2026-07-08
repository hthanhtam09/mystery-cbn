"""Benchmarks for the Quantization stage (budget: 1600 px, K=16 ≤ 2.0 s,
ENGINE_SPEC §26) — including the labkmeans-vs-mediancut comparison backing
the default-algorithm decision."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.records import Provenance, RasterImage
from mysterycbn.stages.raster.quantize import quantize_raster

RNG = np.random.default_rng(0)
_RASTER = RasterImage(
    np.clip(RNG.random((1200, 1600, 3)), 0, 1).astype(np.float32),
    0.4,
    1.0,
    False,
    1,
    Provenance("preprocess", "1.0.0", "0" * 64, "1" * 64),
)


def test_bench_labkmeans_k16_1600(benchmark: Any) -> None:
    _, pal = benchmark(quantize_raster, _RASTER, n_colors=16)
    assert 2 <= pal.size <= 16


def test_bench_mediancut_k16_1600(benchmark: Any) -> None:
    _, pal = benchmark(quantize_raster, _RASTER, n_colors=16, impl="mediancut")
    assert 2 <= pal.size <= 16


def test_bench_labkmeans_k30_1600(benchmark: Any) -> None:
    _, pal = benchmark(quantize_raster, _RASTER, n_colors=30)
    assert 2 <= pal.size <= 30
