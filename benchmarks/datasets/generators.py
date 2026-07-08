"""Category-specific synthetic label-map generators (docs/DATASET_STANDARDS.md §3).

Each generator is a deterministic, seeded, in-repo procedure -- never a
photograph or external asset (ARCHITECTURE.md §10 legal invariant). They
follow the same style as benchmarks/framework/fixtures.py: pure functions
from parameters to an int32 label map, so a fixture is fully reproducible
from its recorded generator name + params.

Category "content" is evoked structurally (silhouette placement, region
arrangement, edge density) rather than photographically -- e.g. "animals"
places a small number of rounded blob silhouettes on a background, while
"architecture" tiles rectilinear blocks. This keeps every fixture legally
and technically a synthetic construction with analytic ground truth for
palette size and approximate region count.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _blob_silhouettes(
    seed: int, width: int, height: int, k: int, n_blobs: int, blob_radius_frac: float
) -> np.ndarray:
    """Rounded blob shapes on a flat background (animals, flowers, food)."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((height, width), dtype=np.int32)
    yy, xx = np.mgrid[0:height, 0:width]
    radius = int(min(height, width) * blob_radius_frac)
    for i in range(n_blobs):
        cy = rng.integers(radius, height - radius) if height > 2 * radius else height // 2
        cx = rng.integers(radius, width - radius) if width > 2 * radius else width // 2
        wobble = 1.0 + 0.15 * np.sin(4 * np.arctan2(yy - cy, xx - cx))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= (radius * wobble) ** 2
        labels[mask] = 1 + (i % (k - 1)) if k > 1 else 1
    return labels


def _portrait_regions(seed: int, width: int, height: int, k: int) -> np.ndarray:
    """Head-and-shoulders style nested ellipse regions (people)."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((height, width), dtype=np.int32)
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height * 0.4, width * 0.5
    for i, (ry_frac, rx_frac) in enumerate(
        [(0.42, 0.30), (0.28, 0.20), (0.16, 0.12), (0.08, 0.06)]
    ):
        ry, rx = height * ry_frac, width * rx_frac
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        labels[mask] = 1 + (i % (k - 1)) if k > 1 else 1
    jitter = rng.integers(0, k, size=(4, 4))
    block_h, block_w = height // 4, width // 4
    for by in range(4):
        for bx in range(4):
            region = labels[by * block_h : (by + 1) * block_h, bx * block_w : (bx + 1) * block_w]
            region[region == 0] = jitter[by, bx]
    return labels


def _horizon_bands(seed: int, width: int, height: int, k: int, n_bands: int) -> np.ndarray:
    """Horizontal terrain bands with a perturbed skyline (landscape)."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((height, width), dtype=np.int32)
    edges = np.sort(rng.integers(0, height, n_bands - 1))
    edges = np.concatenate(([0], edges, [height]))
    x = np.arange(width)
    for i in range(n_bands):
        base = int(edges[i])
        wobble = (np.sin(x / width * 2 * np.pi * (i + 1) + seed) * height * 0.03).astype(np.int32)
        row_top = np.clip(base + wobble, 0, height - 1)
        for col in range(width):
            labels[row_top[col] :, col] = i % k
    return labels


def _rectilinear_blocks(seed: int, width: int, height: int, k: int, grid: int) -> np.ndarray:
    """Building-facade style rectilinear grid (architecture)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, k, (grid, grid))
    block_h, block_w = height // grid, width // grid
    labels = np.repeat(np.repeat(base, block_h, axis=0), block_w, axis=1).astype(np.int32)
    pad_h, pad_w = height - labels.shape[0], width - labels.shape[1]
    if pad_h or pad_w:
        labels = np.pad(labels, ((0, pad_h), (0, pad_w)), mode="edge")
    return labels


def _radial_wedges(seed: int, width: int, height: int, k: int, n_wedges: int) -> np.ndarray:
    """Pie-slice wedges radiating from center (vehicles: wheel/body panel proxy)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    angle = np.arctan2(yy - height / 2, xx - width / 2)
    wedge_idx = ((angle + np.pi) / (2 * np.pi) * n_wedges).astype(np.int32) % n_wedges
    palette = rng.integers(0, k, n_wedges)
    result: np.ndarray = palette[wedge_idx].astype(np.int32)
    return result


def _bold_outline_cells(seed: int, width: int, height: int, k: int, grid: int) -> np.ndarray:
    """Large flat cells with thick borders (cartoons: bold flat-shaded look)."""
    rng = np.random.default_rng(seed)
    n_seeds = grid
    seeds_y = rng.integers(0, height, n_seeds)
    seeds_x = rng.integers(0, width, n_seeds)
    seed_label = rng.integers(0, k, n_seeds)
    yy, xx = np.mgrid[0:height, 0:width]
    d2 = (yy[..., None] - seeds_y) ** 2 + (xx[..., None] - seeds_x) ** 2
    nearest = np.argmin(d2, axis=-1)
    return seed_label[nearest].astype(np.int32)


_CATEGORY_GENERATORS: dict[str, Callable[[int, int, int, int], np.ndarray]] = {
    "animals": lambda seed, w, h, k: _blob_silhouettes(
        seed, w, h, k, n_blobs=3, blob_radius_frac=0.22
    ),
    "flowers": lambda seed, w, h, k: _blob_silhouettes(
        seed, w, h, k, n_blobs=6, blob_radius_frac=0.10
    ),
    "people": _portrait_regions,
    "landscape": lambda seed, w, h, k: _horizon_bands(seed, w, h, k, n_bands=5),
    "architecture": lambda seed, w, h, k: _rectilinear_blocks(seed, w, h, k, grid=8),
    "food": lambda seed, w, h, k: _blob_silhouettes(
        seed, w, h, k, n_blobs=1, blob_radius_frac=0.35
    ),
    "vehicles": lambda seed, w, h, k: _radial_wedges(seed, w, h, k, n_wedges=10),
    "cartoons": lambda seed, w, h, k: _bold_outline_cells(seed, w, h, k, grid=12),
}


def generate_category_labels(
    category: str, *, seed: int, width: int, height: int, k: int
) -> np.ndarray:
    """Deterministically generate a label map for the given category."""
    if category not in _CATEGORY_GENERATORS:
        raise KeyError(f"unknown category {category!r}; available: {sorted(_CATEGORY_GENERATORS)}")
    labels = _CATEGORY_GENERATORS[category](seed, width, height, k)
    if labels.shape != (height, width):
        raise AssertionError(
            f"generator for {category!r} produced shape {labels.shape}, expected {(height, width)}"
        )
    return labels
