"""Golden test for the Organic Region Partition stage (ADR-003).

Digest = SHA-256 of (component-map bytes ‖ canonical JSON of region records,
edges, and filler/render-filler id sets, floats rounded to 6 decimals) on a
deterministic synthetic label map, fixed stage seed. A change is a reviewed
golden-update event (BENCHMARK_SPEC §4.3), same convention as
``test_components_golden.py``/other graph-stage golden tests.

Golden updated after ``SKIP_BACKGROUND_DEFAULT`` was introduced (the fixture's
label-0 region is the page-border-touching background and is now excluded
from organic partitioning by default -- see ``organic_partition.py``'s
``_background_region_id``), and again after regions whose palette color is
near-black (typically a source image's own pre-drawn outline stroke) are
now folded into a neighboring region before partitioning, rather than left
as their own standalone region (see ``fold_regions_where`` in
``_organic_common.py`` and ``SKIP_DARK_LAB_L_THRESHOLD``).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.organic_partition import organic_partition_regions, stage_seed

_GOLDEN = "c30a3d9df0e8279561a37982a8ae6e8a272af3fb5191b9d118dba2e4c53375ed"

PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)


def _fixture() -> tuple[LabelMap, Palette]:
    # Two coherent flat blocks (upscaled blocky noise) -- large enough for
    # several organic cells per region, small enough to run fast.
    rng = np.random.default_rng(0)
    base = np.repeat(np.repeat(rng.integers(0, 3, (6, 8)), 8, axis=0), 8, axis=1)
    palette = Palette(
        colors=tuple(
            PaletteColor.from_lab(i, (15.0 + 25.0 * i, 4.0 * i - 6.0, 2.0 - i), 100)
            for i in range(3)
        ),
        provenance=PROV,
    )
    return LabelMap(labels=base.astype(np.int32), provenance=PROV), palette


def _digest() -> str:
    label_map, palette = _fixture()
    graph = build_region_graph(label_map, palette)
    new_graph, filler_ids, render_filler_ids, _ = organic_partition_regions(
        graph,
        palette,
        mode="streamline",
        min_area_px=30.0,
        seed_density_px=25.0,
        rim_px=1.0,
        warp_px=2.0,
        noise_scale_px=6.0,
        ribbon_elongation=0.3,
        island_probability=0.2,
        island_min_area_px=10.0,
        fold_a_min_px=8.0,
        warp_seed=stage_seed(0),
    )
    canonical = json.dumps(
        {
            "regions": [
                {
                    **r.to_dict(),
                    "centroid": [round(v, 6) for v in r.centroid],
                }
                for r in new_graph.regions
            ],
            "edges": [[a, b, round(d, 6), w] for a, b, d, w in new_graph.edges],
            "filler_ids": sorted(filler_ids),
            "render_filler_ids": sorted(render_filler_ids),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(new_graph.component_map.tobytes() + canonical).hexdigest()


def test_golden_organic_partition() -> None:
    assert _digest() == _GOLDEN
