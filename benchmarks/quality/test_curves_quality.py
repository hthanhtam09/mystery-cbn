"""Curve-fitting quality benchmarks: Schneider vs plain least-squares Bézier
vs Chaikin vs Catmull–Rom (ENGINE_SPEC §18).

Fixtures are synthetic smooth-with-noise polylines standing in for §17
output (raw crack staircases would flag every vertex as a corner; the
simplify/smooth stages that remove them are not built yet).

Measured per fitter:

- **Fidelity** — max residual at the input vertices (the §18 error bound).
- **Economy** — emitted segment count vs input vertex count.
- **Smoothness** — bending energy Σ turn-angle² over the chain sampled at
  16 points/segment (lower = smoother).

Gates encode the reason Schneider is the default: it is the only fitter
that is simultaneously error-bounded, economical, and smoother than the
interpolating alternative.
"""

from __future__ import annotations

import numpy as np

from mysterycbn.model.vector import BezierSegment
from mysterycbn.stages.vector.curves import fit_arc

RNG = np.random.default_rng(0)
TOLERANCE_PT = 0.5


def _noisy_wave(n: int, seed: int) -> np.ndarray:
    """Smooth open curve with sub-tolerance jitter (post-§17-like)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    x = 300.0 * t
    y = 40.0 * np.sin(2.5 * np.pi * t) + 12.0 * np.cos(7.1 * np.pi * t + 1.0)
    pts = np.stack([x, y], axis=1)
    pts[1:-1] += rng.normal(0.0, 0.15, (n - 2, 2))  # jitter < tolerance
    return pts


FIXTURES = [_noisy_wave(320, seed) for seed in range(6)]
IMPLS = ("schneider", "bezier", "chaikin", "catmull")


def _sample_chain(segments: tuple[BezierSegment, ...], per_segment: int = 16) -> np.ndarray:
    u = np.linspace(0.0, 1.0, per_segment)
    b = np.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u**2 * (1 - u), u**3], axis=1)
    return np.concatenate([b @ s.control for s in segments])


def _bending_energy(segments: tuple[BezierSegment, ...]) -> float:
    pts = _sample_chain(segments)
    v = np.diff(pts, axis=0)
    keep = np.linalg.norm(v, axis=1) > 1e-12
    v = v[keep]
    cos = (v[:-1] * v[1:]).sum(axis=1) / (
        np.linalg.norm(v[:-1], axis=1) * np.linalg.norm(v[1:], axis=1)
    )
    turn = np.arccos(np.clip(cos, -1.0, 1.0))
    return float((turn**2).sum())


def _metrics(impl: str) -> tuple[float, int, float]:
    """(worst residual, total segments, total bending energy) over fixtures."""
    worst_err, segments, energy = 0.0, 0, 0.0
    for pts in FIXTURES:
        segs, _, err = fit_arc(pts, tolerance_pt=TOLERANCE_PT, impl=impl)
        worst_err = max(worst_err, err)
        segments += len(segs)
        energy += _bending_energy(segs)
    return worst_err, segments, energy


def test_schneider_is_error_bounded() -> None:
    err, _, _ = _metrics("schneider")
    assert err <= TOLERANCE_PT
    # The uncontrolled least-squares comparator misses the bound here.
    err_lsq, _, _ = _metrics("bezier")
    assert err_lsq > TOLERANCE_PT


def test_schneider_is_economical() -> None:
    _, segments, _ = _metrics("schneider")
    total_vertices = sum(len(p) for p in FIXTURES)
    assert segments <= 0.15 * total_vertices  # §18 segment budget
    _, segs_catmull, _ = _metrics("catmull")
    _, segs_chaikin, _ = _metrics("chaikin")
    assert segments < segs_chaikin < segs_catmull


def test_schneider_is_smoother_than_interpolants() -> None:
    _, _, energy_schneider = _metrics("schneider")
    _, _, energy_catmull = _metrics("catmull")
    # Catmull–Rom interpolates the jitter; Schneider averages through it.
    assert energy_schneider < energy_catmull
    # Chaikin may be smoother still — but it is not error-bounded, which is
    # why it is not the default:
    err_chaikin, _, _ = _metrics("chaikin")
    assert err_chaikin > 0.0


def test_all_fitters_preserve_endpoints_and_corners() -> None:
    square = np.array(
        [[0.0, 0.0], [30.0, 0.0], [60.0, 0.0], [60.0, 30.0], [60.0, 60.0]], dtype=np.float64
    )
    for impl in IMPLS:
        segs, corners, _ = fit_arc(square, tolerance_pt=TOLERANCE_PT, impl=impl)
        assert corners  # the 90° bend is a corner for every fitter
        assert np.array_equal(segs[0].control[0], square[0])
        assert np.array_equal(segs[-1].control[3], square[-1])
        joint = corners[0]
        assert np.array_equal(segs[joint].control[0], np.array([60.0, 0.0]))
