"""Property tests for the Arc Graph stage (ENGINE_SPEC §15)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.vector.arcgraph import build_arc_graph
from mysterycbn.stages.vector.topology import build_topology_graph

PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)
PAL4 = Palette(
    colors=tuple(PaletteColor.from_lab(i, (10.0 + 25.0 * i, 0.0, 0.0), 100) for i in range(4)),
    provenance=PROV,
)
BOX = (10.0, 20.0, 100.0, 200.0)  # (x0, y0, w, h) pt


@settings(max_examples=50, deadline=None)
@given(st.integers(1, 8), st.integers(1, 8), st.integers(0, 2**31 - 1), st.integers(2, 4))
def test_planar_map_properties_on_random_maps(h: int, w: int, seed: int, k: int) -> None:
    labels = np.random.default_rng(seed).integers(0, k, (h, w)).astype(np.int32)
    lm = LabelMap(labels=labels, provenance=PROV)
    rg = build_region_graph(lm, PAL4)
    topo = build_topology_graph(rg.component_map)
    graph = build_arc_graph(topo, rg, content_box=BOX)
    # Euler + area partition + walk consistency are asserted inside the
    # builder (StageError on violation) — reaching here IS the property.
    assert len(graph.faces) == len(rg.regions)
    # Each arc is referenced once per non-exterior side.
    refs: dict[int, int] = {}
    for face in graph.faces:
        for walk in face.all_walks():
            for arc_id, _ in walk:
                refs[arc_id] = refs.get(arc_id, 0) + 1
    for arc in graph.arcs:
        expected = int(arc.left_region != -1) + int(arc.right_region != -1)
        assert refs.get(arc.arc_id, 0) == expected
    # Determinism.
    again = build_arc_graph(topo, rg, content_box=BOX)
    assert graph.to_dict() == again.to_dict()
