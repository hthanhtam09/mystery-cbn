"""Golden test for the Tiny Region Merge stage.

Digest = SHA-256 of (merged component-map bytes ‖ canonical JSON of regions,
edges, palette, renumber map; floats rounded to 6 decimals) on the same
deterministic seed-0 fixture as the components golden. Pure integer
combinatorics + the cached ΔE00 table; a digest change is a reviewed
golden-update event (BENCHMARK_SPEC §4.3).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.merge import merge_tiny_regions

_GOLDEN = "567772b1d9dc9be92617a77b94527025defdc1a33b60291554cebb43a25773db"

PROV = Provenance("denoise", "1.0.0", "0" * 64, "1" * 64)


def _fixture() -> tuple[LabelMap, Palette]:
    rng = np.random.default_rng(0)
    base = np.repeat(np.repeat(rng.integers(0, 6, (12, 16)), 8, axis=0), 8, axis=1)
    noise = rng.random(base.shape) < 0.01
    base[noise] = rng.integers(0, 6, int(noise.sum()))
    palette = Palette(
        colors=tuple(
            PaletteColor.from_lab(i, (10.0 + 15.0 * i, 5.0 * i - 12.0, 8.0 - 3.0 * i), 100)
            for i in range(6)
        ),
        provenance=PROV,
    )
    return LabelMap(labels=base.astype(np.int32), provenance=PROV), palette


def _digest() -> str:
    label_map, palette = _fixture()
    graph = build_region_graph(label_map, palette)
    merged, new_palette, renumber = merge_tiny_regions(graph, palette, a_min=64.0)
    canonical = json.dumps(
        {
            "regions": [
                {**r.to_dict(), "centroid": [round(v, 6) for v in r.centroid]}
                for r in merged.regions
            ],
            "edges": [[a, b, round(d, 6), w] for a, b, d, w in merged.edges],
            "palette": [
                [c.index, [round(v, 6) for v in c.lab], c.coverage_px] for c in new_palette.colors
            ],
            "renumber": list(renumber),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(merged.component_map.tobytes() + canonical).hexdigest()


def test_golden_merged_region_graph() -> None:
    assert _digest() == _GOLDEN
