"""Golden test for Minimum Gap Enforcement / Gap Repair (Sprint 36B.3;
docs/modules/GAP_REPAIR_DESIGN.md).

Digest = SHA-256 of the canonical JSON dump of the ``ArcGraph`` after
``_minimum_gap_enforcement`` (coordinates rounded to 6 decimals), built
from the same deterministic seed-0 label-map fixture as the
arcgraph/merge/components/spike-removal goldens, run through the full
raster -> vector chain (regions -> topology -> arcgraph), with two
synthetic "pinch" arcs appended -- a controlled, seed-independent
perturbation constructed the same way the property/unit tests verify Gap
Repair actually succeeds (single localized close approach at an existing
interior vertex, neighbors well beyond threshold on both sides). Real
crack-traced arcs are zigzaggy enough that engineering a *repairable*
pinch directly out of them (rather than one that lands on an endpoint or
crosses unrelated real geometry) is not reliably reproducible by
translation; appending controlled synthetic arcs is the same discipline
the Spike Removal golden test already uses (``_stitch_spike``) to
guarantee the fixture actually exercises the algorithm. A digest change is
a reviewed golden-update event (BENCHMARK_SPEC §4.3).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.model.vector import Arc
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.geometry_normalize import (
    GeometryNormalizeConfig,
    _minimum_gap_enforcement,
)
from mysterycbn.stages.vector.topology import build_topology_graph

_GOLDEN = "e90ec6102c0a72d02345a37dfad690e0a4f96208bff9151149c8b3d3080268c0"

PROV = Provenance("simplify", "1.0.0", "0" * 64, "1" * 64)
_MM_TO_PT = 72.0 / 25.4
_MIN_GAP_MM = 0.1


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


def _pinch_arc_pair(*, min_gap_pt: float, first_arc_id: int) -> tuple[Arc, Arc]:
    """Two synthetic arcs, far from the real fixture's geometry (large
    fixed offset), forming a single localized close approach at an
    existing interior vertex on both sides -- the exact construction
    proven (by the unit/property tests) to be reliably repairable. The
    initial gap is a small fraction of the threshold (0.1x, matching the
    unit-test fixture that is hand-verified to succeed): a larger initial
    fraction was found during implementation to sometimes leave the
    taper's immediate original neighbor insufficiently displaced relative
    to the still-substantial baseline separation, correctly triggering
    the monotone-improvement check to skip -- this fixture is
    deliberately chosen from the region of parameter space verified to
    repair successfully, so the golden digest documents Gap Repair's
    actual committed-repair behavior, not its skip path (already covered
    by unit/property tests)."""
    tiny_gap = min_gap_pt * 0.1
    offset = 10_000.0  # translate well clear of the real fixture's arcs
    a = Arc(
        arc_id=first_arc_id,
        points=np.array([[-50.0 + offset, 0.0], [0.0 + offset, 0.0], [50.0 + offset, 0.0]]),
        left_region=900,
        right_region=901,
    )
    b = Arc(
        arc_id=first_arc_id + 1,
        points=np.array(
            [[-50.0 + offset, 10.0], [0.0 + offset, tiny_gap], [50.0 + offset, 10.0]]
        ),
        left_region=902,
        right_region=903,
    )
    return a, b


def _digest() -> str:
    label_map, palette = _fixture()
    region_graph = build_region_graph(label_map, palette)
    topology = build_topology_graph(region_graph.component_map)
    graph = build_arc_graph(
        region_graph=region_graph,
        topology=topology,
        content_box=content_box_pt((215.9, 279.4, 12.7)),
    )

    min_gap_pt = _MIN_GAP_MM * _MM_TO_PT
    pinch_a, pinch_b = _pinch_arc_pair(min_gap_pt=min_gap_pt, first_arc_id=len(graph.arcs))
    all_arcs = (*graph.arcs, pinch_a, pinch_b)

    # _minimum_gap_enforcement operates purely on Arc.points -- no Face
    # traversal (GAP_REPAIR_DESIGN.md's "no face traversal" requirement),
    # so appending arcs without a corresponding Face is sufficient here.
    config = GeometryNormalizeConfig({"min_gap_mm": _MIN_GAP_MM}, simplify_tolerance_mm=0.15)
    repaired_arcs, repaired_count = _minimum_gap_enforcement(all_arcs, config=config)

    canonical = json.dumps(
        {
            "arcs": [
                {
                    **a.to_dict(),
                    "points": [[round(x, 6), round(y, 6)] for x, y in a.points.tolist()],
                }
                for a in repaired_arcs
            ],
            "repaired": repaired_count,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), repaired_count


def test_gap_repair_golden_digest() -> None:
    digest, repaired = _digest()
    assert digest == _GOLDEN
    assert repaired == 1  # the synthetic pinch pair must be actually repaired


def test_gap_repair_golden_digest_is_deterministic() -> None:
    digest1, repaired1 = _digest()
    digest2, repaired2 = _digest()
    assert digest1 == digest2
    assert repaired1 == repaired2
