"""Golden test for Spike Removal (Sprint 36B.2;
docs/modules/geometry_normalize.md §8.2).

Digest = SHA-256 of the canonical JSON dump of the ``ArcGraph`` after
``_spike_removal`` (coordinates rounded to 6 decimals), built from the same
deterministic seed-0 label-map fixture as the arcgraph/merge/components
goldens, run through the full raster -> vector chain (regions -> topology
-> arcgraph) with a handful of synthetic spikes stitched onto the resulting
arcs before spike removal runs, so the fixture actually exercises the
algorithm rather than passing through unchanged. A digest change is a
reviewed golden-update event (BENCHMARK_SPEC §4.3).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.model.vector import Arc, ArcGraph
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.geometry_normalize import GeometryNormalizeConfig, _spike_removal
from mysterycbn.stages.vector.topology import build_topology_graph

_GOLDEN = "c1d153d416ea0b7a64d138a2cf64616a78659c515be1db581761166df1a5b94c"

PROV = Provenance("simplify", "1.0.0", "0" * 64, "1" * 64)
_MM_TO_PT = 72.0 / 25.4
_SPIKE_LENGTH_MM = 0.1


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


def _stitch_spike(points: np.ndarray, *, spike_length_pt: float) -> np.ndarray:
    """Deterministically insert an out-and-back spike near the midpoint of
    an open polyline (if it has an interior to insert into) -- a fixed,
    seed-independent perturbation so the golden fixture actually exercises
    Spike Removal rather than passing through unchanged."""
    if points.shape[0] < 3:
        return points
    mid = points.shape[0] // 2
    d = spike_length_pt * 0.3
    direction = points[mid] - points[mid - 1]
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return points
    perp = np.array([-direction[1], direction[0]]) / norm
    spike_point = points[mid] + perp * d
    return np.concatenate([points[: mid + 1], spike_point[None, :], points[mid:]], axis=0)


def _digest() -> str:
    label_map, palette = _fixture()
    region_graph = build_region_graph(label_map, palette)
    topology = build_topology_graph(region_graph.component_map)
    graph = build_arc_graph(
        region_graph=region_graph,
        topology=topology,
        content_box=content_box_pt((215.9, 279.4, 12.7)),
    )
    spike_length_pt = _SPIKE_LENGTH_MM * _MM_TO_PT
    stitched_arcs = tuple(
        Arc(
            arc_id=arc.arc_id,
            points=_stitch_spike(arc.points, spike_length_pt=spike_length_pt),
            left_region=arc.left_region,
            right_region=arc.right_region,
            closed=arc.closed,
        )
        for arc in graph.arcs
    )
    stitched_graph = ArcGraph(
        arcs=stitched_arcs, faces=graph.faces, work_scale=graph.work_scale, provenance=PROV
    )

    config = GeometryNormalizeConfig(
        {"spike_length_mm": _SPIKE_LENGTH_MM}, simplify_tolerance_mm=0.15
    )
    cleaned_arcs, removed = _spike_removal(stitched_graph.arcs, config=config)
    assert removed > 0, "fixture must actually exercise spike removal"

    canonical = json.dumps(
        {
            "arcs": [
                {
                    **a.to_dict(),
                    "points": [[round(x, 6), round(y, 6)] for x, y in a.points.tolist()],
                }
                for a in cleaned_arcs
            ],
            "removed": removed,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_spike_removal_golden_digest() -> None:
    assert _digest() == _GOLDEN


def test_spike_removal_golden_digest_is_deterministic() -> None:
    assert _digest() == _digest()
