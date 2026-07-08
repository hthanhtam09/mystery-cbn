"""Golden test for the Connected Components stage.

Digest = SHA-256 of (component-map bytes ‖ canonical JSON of region records
and edges, floats rounded to 6 decimals) on a deterministic synthetic label
map, seed 0. Pure integer combinatorics plus the cached palette ΔE00 table,
so digests are expected stable across platforms of the pinned container;
a change is a reviewed golden-update event (BENCHMARK_SPEC §4.3).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph

_GOLDEN = "67662fce0b0fcdd6cb58e29548a339f3956db42b66eef9e07e2efb8aa848786f"

PROV = Provenance("denoise", "1.0.0", "0" * 64, "1" * 64)


def _fixture() -> tuple[LabelMap, Palette]:
    # Coherent blocks + speckle: realistic post-denoise structure, seed 0.
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
    canonical = json.dumps(
        {
            "regions": [
                {
                    **r.to_dict(),
                    "centroid": [round(v, 6) for v in r.centroid],
                }
                for r in graph.regions
            ],
            "edges": [[a, b, round(d, 6), w] for a, b, d, w in graph.edges],
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(graph.component_map.tobytes() + canonical).hexdigest()


def test_golden_region_graph() -> None:
    assert _digest() == _GOLDEN
