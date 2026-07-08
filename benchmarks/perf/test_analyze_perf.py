"""Benchmarks for the Color Analysis stage (budget: ≤ 0.1 s at 1600 px,
ENGINE_SPEC §26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.records import Provenance, RasterImage
from mysterycbn.stages.raster.analyze import compute_stats, propose_overrides

RNG = np.random.default_rng(0)
_RASTER = RasterImage(
    np.clip(RNG.random((1200, 1600, 3)), 0, 1).astype(np.float32),
    0.4,
    1.0,
    False,
    1,
    Provenance("preprocess", "1.0.0", "0" * 64, "1" * 64),
)


def test_bench_compute_stats_1600(benchmark: Any) -> None:
    stats = benchmark(compute_stats, _RASTER)
    assert stats.luminance_histogram.shape == (64,)


def test_bench_full_analysis_with_proposals(benchmark: Any) -> None:
    def full() -> int:
        proposal = propose_overrides(compute_stats(_RASTER))
        k = proposal.fragment["quantize"]["n_colors"]
        assert isinstance(k, int)
        return k

    k = benchmark(full)
    assert 8 <= k <= 30
