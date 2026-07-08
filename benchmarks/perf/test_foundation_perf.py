"""Micro-benchmarks for the foundation layer (color math + geometry kernel).

Authoritative numbers come from the pinned container (BENCHMARK_SPEC.md §9);
this suite tracks relative regressions on the hot foundation paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.foundation.color import DefaultColorScience
from mysterycbn.foundation.geometry.default import DefaultGeometryKernel
from mysterycbn.foundation.geometry.primitives import PolylineData

CS = DefaultColorScience()
GK = DefaultGeometryKernel()
RNG = np.random.default_rng(0)


def test_bench_srgb_to_lab_512(benchmark: Any) -> None:
    img = RNG.random((512, 512, 3))
    lab = benchmark(CS.srgb_to_lab, img)
    assert lab.shape == img.shape


def test_bench_delta_e_2000_100k_pairs(benchmark: Any) -> None:
    a = RNG.uniform(-50, 100, (100_000, 3))
    b = RNG.uniform(-50, 100, (100_000, 3))
    out = benchmark(CS.delta_e_2000, a, b)
    assert out.shape == (100_000,)


def test_bench_trace_cracks_64(benchmark: Any) -> None:
    labels = RNG.integers(0, 4, (64, 64)).astype(np.int32)
    loops = benchmark(GK.trace_cracks, labels)
    assert len(loops) >= 1


def test_bench_simplify_staircase_20k(benchmark: Any) -> None:
    n = 10_000
    xs = np.repeat(np.arange(n + 1, dtype=np.float64), 2)[1:]
    ys = np.repeat(np.arange(n + 1, dtype=np.float64), 2)[:-1]
    line = PolylineData(np.column_stack([xs, ys]))
    out = benchmark(GK.simplify_polyline, line, 2.0)
    assert out.coords.shape[0] < line.coords.shape[0]


def test_bench_polylabel_ring_256(benchmark: Any) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    ring = PolylineData(
        np.column_stack([100.0 * np.cos(theta), 100.0 * np.sin(theta)]), is_closed=True
    )
    _, r = benchmark(GK.pole_of_inaccessibility, ring)
    assert 99.0 < r <= 100.0
