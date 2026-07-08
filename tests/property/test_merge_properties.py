"""Property tests for the Tiny Region Merge stage (ENGINE_SPEC §11, MATH_SPEC §11)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.graph.merge import merge_tiny_regions

PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)

# 0 dark, 1 mid, 2 light, 3 very light: ΔE(1,2) < ΔE(1,0) etc.
_LABS = [(10.0, 0.0, 0.0), (40.0, 0.0, 0.0), (70.0, 0.0, 0.0), (95.0, 0.0, 0.0)]
PAL4 = Palette(
    colors=tuple(PaletteColor.from_lab(i, lab, 100) for i, lab in enumerate(_LABS)),
    provenance=PROV,
)


@settings(max_examples=40, deadline=None)
@given(st.integers(2, 7), st.integers(2, 7), st.integers(0, 2**31 - 1), st.integers(0, 30))
def test_termination_and_floor_property(h: int, w: int, seed: int, lam: int) -> None:
    labels = np.random.default_rng(seed).integers(0, 4, (h, w)).astype(np.int32)
    graph = build_region_graph(LabelMap(labels=labels, provenance=PROV), PAL4)
    a_min = min(4.0, labels.size / 2)
    merged, palette, _ = merge_tiny_regions(graph, PAL4, a_min=a_min, lambda_boundary=float(lam))
    # Post-stage invariant: no sub-floor region unless the page degenerated.
    if len(merged.regions) > 1:
        assert all(r.area_px >= a_min for r in merged.regions)
    assert sum(r.area_px for r in merged.regions) == labels.size
    assert all(r.label < palette.size for r in merged.regions)
    # Boundary double-entry identity survives the rebuild.
    cm = merged.component_map
    internal = int((cm[:, :-1] != cm[:, 1:]).sum()) + int((cm[:-1, :] != cm[1:, :]).sum())
    total = sum(w for *_, w in merged.edges)
    assert total == internal
    assert sum(r.perimeter_px for r in merged.regions) == 2 * (h + w) + 2 * total
