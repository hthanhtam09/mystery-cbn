"""Synthetic fixture generators (BENCHMARK_SPEC.md §2, §3 tier-1 "analytic
ground truth"). No copyrighted imagery is used anywhere in this repo
(ARCHITECTURE.md §10 legal invariant); fixtures are deterministic in-repo
generators, not photographs -- the framework runs the real pipeline stages
starting from a synthetic ``LabelMap`` (post-quantize artifact) rather than
raw pixels, since no raster fixture assets exist yet.

Each fixture is pinned by content hash in ``FIXTURE_MANIFEST`` so a run
records exactly which generator + parameters produced it (the in-repo
analogue of BENCHMARK_SPEC §2.1's SHA-256 manifest).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DATASET_VERSION = 1


@dataclass(frozen=True)
class Fixture:
    """One synthetic benchmark fixture: a label map + its declared category."""

    fixture_id: str
    category: str
    labels: np.ndarray
    n_colors: int

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.labels.tobytes()).hexdigest()

    @property
    def megapixels(self) -> float:
        return self.labels.size / 1_000_000.0


def _checkerboard(seed: int, grid: int, block: int, k: int) -> np.ndarray:
    """Blocky "photograph-like" fixture: irregular region sizes, moderate
    palette, no exact analytic region count (stands in for F-photo-*)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, k, (grid, grid))
    return np.repeat(np.repeat(base, block, axis=0), block, axis=1).astype(np.int32)


def _flat_bands(width: int, height: int, k: int) -> np.ndarray:
    """Hard-edged flat art: k horizontal bands, no gradients (F-flat-2)."""
    rows = np.linspace(0, k, height, endpoint=False).astype(np.int32)
    return np.repeat(rows[:, None], width, axis=1)


def _illustration_blobs(seed: int, width: int, height: int, k: int) -> np.ndarray:
    """Flat-shaded illustration: few large Voronoi-like regions (F-illu-2)."""
    rng = np.random.default_rng(seed)
    n_seeds = k * 3
    seeds_y = rng.integers(0, height, n_seeds)
    seeds_x = rng.integers(0, width, n_seeds)
    seed_label = rng.integers(0, k, n_seeds)
    yy, xx = np.mgrid[0:height, 0:width]
    d2 = (yy[..., None] - seeds_y) ** 2 + (xx[..., None] - seeds_x) ** 2
    nearest = np.argmin(d2, axis=-1)
    return seed_label[nearest].astype(np.int32)


def _degenerate(width: int, height: int) -> np.ndarray:
    """Single flat color, the whole page (F-degen-1)."""
    return np.zeros((height, width), dtype=np.int32)


def _high_noise(seed: int, width: int, height: int, k: int, block: int = 6) -> np.ndarray:
    """High-frequency speckle: small independently-assigned blocks
    (F-noise-2). Per-pixel noise (block=1) is pathological for the crack
    tracer -- it produces tens of thousands of 1-px regions on even a small
    canvas -- so ``block`` keeps region count high-but-tractable while still
    exercising the "many small regions" stress case the fixture is for."""
    rng = np.random.default_rng(seed)
    gy, gx = height // block, width // block
    base = rng.integers(0, k, (gy, gx))
    return np.repeat(np.repeat(base, block, axis=0), block, axis=1).astype(np.int32)


def _thin_structure(width: int, height: int, k: int) -> np.ndarray:
    """Background plus 2px-wide lines: known analytic region/line count
    (F-thin-2's synthetic ground-truth chart)."""
    labels = np.zeros((height, width), dtype=np.int32)
    for i in range(1, k):
        y = i * height // k
        labels[max(0, y - 1) : y + 1, :] = i
    return labels


_GENERATORS: dict[str, Callable[[], np.ndarray]] = {
    "F-photo-05": lambda: _checkerboard(seed=0, grid=9, block=8, k=10),
    "F-photo-2": lambda: _checkerboard(seed=1, grid=27, block=12, k=10),
    "F-illu-2": lambda: _illustration_blobs(seed=2, width=576, height=576, k=8),
    "F-flat-2": lambda: _flat_bands(width=576, height=576, k=6),
    "F-noise-2": lambda: _high_noise(seed=3, width=240, height=240, k=6),
    "F-degen-1": lambda: _degenerate(width=400, height=400),
    "F-thin-2": lambda: _thin_structure(width=576, height=576, k=10),
}

_CATEGORY: dict[str, str] = {
    "F-photo-05": "photograph",
    "F-photo-2": "photograph",
    "F-illu-2": "illustration",
    "F-flat-2": "flat_art",
    "F-noise-2": "high_noise",
    "F-degen-1": "degenerate",
    "F-thin-2": "thin_structure",
}

# Capped at 10: the framework's palette builder (pipeline._palette_for)
# spreads colors around a single fixed-L, fixed-C hue wheel, which clears
# the QM-16 warn floor (DeltaE00 >= 12) only up to ~10 entries -- CIEDE2000
# compresses hue differences at this chroma/lightness combination beyond
# that. A real quantize-stage palette (varying L and C too) doesn't have
# this ceiling; this is a synthetic-fixture limitation, not an engine one.
_N_COLORS: dict[str, int] = {
    "F-photo-05": 10,
    "F-photo-2": 10,
    "F-illu-2": 8,
    "F-flat-2": 6,
    "F-noise-2": 6,
    "F-degen-1": 1,
    "F-thin-2": 10,
}


def available_fixture_ids() -> tuple[str, ...]:
    return tuple(sorted(_GENERATORS))


def load_fixture(fixture_id: str) -> Fixture:
    """Deterministically generate one named fixture."""
    if fixture_id not in _GENERATORS:
        raise KeyError(f"unknown fixture {fixture_id!r}; available: {available_fixture_ids()}")
    labels = _GENERATORS[fixture_id]()
    n_colors = _N_COLORS[fixture_id] if fixture_id != "F-degen-1" else 1
    return Fixture(
        fixture_id=fixture_id,
        category=_CATEGORY[fixture_id],
        labels=labels,
        n_colors=max(n_colors, int(labels.max()) + 1),
    )


def load_smoke_fixtures() -> tuple[Fixture, ...]:
    """The smoke suite's 2-fixture minimal ladder (BENCHMARK_SPEC §1)."""
    return (load_fixture("F-photo-05"), load_fixture("F-flat-2"))


def load_full_ladder() -> tuple[Fixture, ...]:
    """All fixtures, in a fixed deterministic order."""
    return tuple(load_fixture(fid) for fid in available_fixture_ids())


def fixture_manifest() -> dict[str, dict[str, object]]:
    """Content-hash manifest of every fixture this generator produces
    (the in-repo analogue of ``assets/fixtures/MANIFEST.json``)."""
    manifest: dict[str, dict[str, object]] = {}
    for fixture_id in available_fixture_ids():
        fx = load_fixture(fixture_id)
        manifest[fixture_id] = {
            "category": fx.category,
            "content_hash": fx.content_hash,
            "shape": list(fx.labels.shape),
            "n_colors": fx.n_colors,
            "dataset_version": DATASET_VERSION,
        }
    return manifest
