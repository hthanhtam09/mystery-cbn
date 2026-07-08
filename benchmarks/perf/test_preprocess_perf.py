"""Benchmarks for the Preprocessing stage (budget: 1600 px, 2 passes ≤ 0.8 s,
ENGINE_SPEC §5) — including the bilateral-vs-guided comparison that backs the
default-algorithm decision."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.records import Provenance, RasterImage
from mysterycbn.stages.raster.preprocess import (
    apply_clahe,
    preprocess_raster,
    smooth_bilateral,
    smooth_guided,
)

RNG = np.random.default_rng(0)
_WORKING = np.clip(RNG.random((1200, 1600, 3)) * 0.8 + 0.1, 0, 1).astype(np.float32)


def test_bench_bilateral_default_1600(benchmark: Any) -> None:
    out = benchmark(smooth_bilateral, _WORKING, passes=2, sigma_color=0.08, sigma_space=5.0)
    assert out.shape == _WORKING.shape


def test_bench_guided_alternative_1600(benchmark: Any) -> None:
    out = benchmark(smooth_guided, _WORKING, passes=2, radius=8, eps=1e-3)
    assert out.shape == _WORKING.shape


def test_bench_clahe_1600(benchmark: Any) -> None:
    out = benchmark(apply_clahe, _WORKING, 2.0)
    assert out.shape == _WORKING.shape


def test_bench_full_stage_from_12mp(benchmark: Any) -> None:
    src = RasterImage(
        np.clip(RNG.random((3000, 4000, 3)), 0, 1).astype(np.float32),
        0.0,
        1.0,
        False,
        1,
        Provenance("load", "1.0.0", "0" * 64, "1" * 64),
    )
    out = benchmark(preprocess_raster, src)
    assert max(out.pixels.shape[:2]) == 1600
