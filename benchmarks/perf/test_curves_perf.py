"""Benchmarks for the Curve Fitting stage (budget: ≤ 1.0 s for 80 000
vertices → ≤ 12 000 segments, ENGINE_SPEC §18/§26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.stages.vector.curves import fit_arc

RNG = np.random.default_rng(0)

# 250 arcs × 320 vertices = 80 000 vertices of smooth-with-jitter polylines
# (post-§17-like input; raw crack staircases never reach the fitter).
_ARCS: list[np.ndarray] = []
for i in range(250):
    t = np.linspace(0.0, 1.0, 320)
    freq, phase = 2.0 + (i % 5), 0.13 * i
    pts = np.stack(
        [
            300.0 * t + 5.0 * np.sin(3.0 * np.pi * t + phase),
            40.0 * np.sin(freq * np.pi * t + phase),
        ],
        axis=1,
    )
    pts[1:-1] += RNG.normal(0.0, 0.1, (318, 2))
    _ARCS.append(pts)


def _fit_all() -> int:
    return sum(len(fit_arc(pts, tolerance_pt=0.5)[0]) for pts in _ARCS)


def test_bench_schneider_80k_vertices(benchmark: Any) -> None:
    segments = benchmark(_fit_all)
    assert segments <= 12_000
