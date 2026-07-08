"""Golden test for the Arc Graph stage.

Digest = SHA-256 of the canonical JSON dump of the ArcGraph (coordinates
rounded to 6 decimals) built from the same deterministic seed-0 fixture as
the components/merge goldens. Face walking and doubled-integer topology are
exact; Φ is a single affine map, so digests are expected stable across
platforms of the pinned container; a change is a reviewed golden-update
event (BENCHMARK_SPEC §4.3).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.topology import build_topology_graph

_GOLDEN = "f4dabfb53cd2dffed3d0d5f33fa1ce7c3c9b54727340b5331a5419f37f9c1064"

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
    region_graph = build_region_graph(label_map, palette)
    topology = build_topology_graph(region_graph.component_map)
    graph = build_arc_graph(
        region_graph=region_graph,
        topology=topology,
        content_box=content_box_pt((215.9, 279.4, 12.7)),
    )
    canonical = json.dumps(
        {
            "arcs": [
                {
                    **a.to_dict(),
                    "points": [[round(x, 6), round(y, 6)] for x, y in a.points.tolist()],
                }
                for a in graph.arcs
            ],
            "faces": [f.to_dict() for f in graph.faces],
            "work_scale": round(graph.work_scale, 6),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def test_golden_arc_graph() -> None:
    assert _digest() == _GOLDEN
