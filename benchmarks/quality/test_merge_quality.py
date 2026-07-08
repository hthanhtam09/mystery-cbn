"""Merge quality benchmarks (QUALITY_SPEC QM-11 + ENGINE_SPEC §11 guards).

Measured on deterministic synthetic fixtures (the in-repo asset ladder is not
yet populated; these use the same generator family as the perf suite, seed 0).

- **QM-11 Tiny Region Percentage (Gate).** 0 % of regions below the area
  floor after the merge stage (degenerate single-region page exempt).
- **Fidelity guard (Gate).** Mean ΔE00 of merged pixels vs their new palette
  color ≤ 15.
- **Coverage conservation (sanity).** The merge reduces, never increases,
  region count and preserves total pixel coverage in graph and palette.
"""

from __future__ import annotations

import numpy as np

from mysterycbn.foundation.color import DefaultColorScience
from mysterycbn.model.records import (
    LabelMap,
    Palette,
    PaletteColor,
    Provenance,
    RegionGraph,
)
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.merge import merge_tiny_regions

PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)
FLOOR = 100.0  # px²; well above the speckle scale, below the block scale
FIXTURES = ((0, 12, 0.02), (1, 8, 0.05), (2, 24, 0.01))  # (seed, K, speckle)


def _fixture(seed: int, k: int, speckle: float) -> tuple[LabelMap, Palette]:
    rng = np.random.default_rng(seed)
    base = np.repeat(np.repeat(rng.integers(0, k, (40, 50)), 12, axis=0), 12, axis=1)
    # Quantization-like speckle: noise labels are perceptual neighbors of the
    # local color (adjacent palette indices), not uniform-random colors.
    noise = rng.random(base.shape) < speckle
    jitter = rng.choice([-1, 1], int(noise.sum()))
    base[noise] = np.clip(base[noise] + jitter, 0, k - 1)
    palette = Palette(
        colors=tuple(
            PaletteColor.from_lab(
                i, (15.0 + 70.0 * i / (k - 1), 3.0 * ((i % 3) - 1), 2.0 * ((i % 4) - 1.5)), 100
            )
            for i in range(k)
        ),
        provenance=PROV,
    )
    return LabelMap(labels=base.astype(np.int32), provenance=PROV), palette


def _merged_pixel_delta_e(
    label_map: LabelMap,
    old_palette: Palette,
    merged: RegionGraph,
    new_palette: Palette,
    renumber: tuple[int, ...],
) -> float:
    """Mean ΔE00 between each *recolored* pixel's original palette color and
    its post-merge color (the §11 fidelity metric)."""
    region_label = np.array([r.label for r in merged.regions], dtype=np.int64)
    new_labels = region_label[merged.component_map]
    old_in_new = np.array(renumber, dtype=np.int64)[label_map.labels]
    changed = old_in_new != new_labels  # includes pixels of dropped colors (−1)
    if not changed.any():
        return 0.0
    old_lab = np.array([c.lab for c in old_palette.colors], dtype=np.float64)
    new_lab = np.array([c.lab for c in new_palette.colors], dtype=np.float64)
    deltas = DefaultColorScience().delta_e_2000(
        old_lab[label_map.labels[changed]], new_lab[new_labels[changed]]
    )
    return float(np.mean(deltas))


def _run(seed: int, k: int, speckle: float):  # type: ignore[no-untyped-def]
    label_map, palette = _fixture(seed, k, speckle)
    graph = build_region_graph(label_map, palette)
    merged, new_palette, renumber = merge_tiny_regions(graph, palette, a_min=FLOOR)
    return label_map, palette, graph, merged, new_palette, renumber


def test_qm11_zero_tiny_regions_after_merge() -> None:
    for seed, k, speckle in FIXTURES:
        _, _, graph, merged, _, _ = _run(seed, k, speckle)
        tiny = [r for r in merged.regions if r.area_px < FLOOR]
        assert len(merged.regions) == 1 or not tiny, (
            f"seed {seed}: {len(tiny)} sub-floor regions survive"
        )
        assert len(merged.regions) <= len(graph.regions)


def test_merge_fidelity_mean_delta_e() -> None:
    for seed, k, speckle in FIXTURES:
        label_map, palette, _, merged, new_palette, renumber = _run(seed, k, speckle)
        mean_de = _merged_pixel_delta_e(label_map, palette, merged, new_palette, renumber)
        assert mean_de <= 15.0, f"seed {seed}: mean merged-pixel ΔE00 {mean_de:.2f} > 15"


def test_coverage_conservation() -> None:
    label_map, _, _, merged, new_palette, _ = _run(0, 12, 0.02)
    assert sum(r.area_px for r in merged.regions) == label_map.labels.size
    assert sum(c.coverage_px for c in new_palette.colors) == label_map.labels.size
