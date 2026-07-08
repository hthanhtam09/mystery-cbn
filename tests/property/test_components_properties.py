"""Property tests for the Connected Components stage (ENGINE_SPEC §9–§10)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph, label_components

PROV = Provenance("denoise", "1.0.0", "0" * 64, "1" * 64)


def _palette(k: int) -> Palette:
    colors = tuple(
        PaletteColor.from_lab(i, (5.0 + 90.0 * i / max(k - 1, 1), 0.0, 0.0), 100) for i in range(k)
    )
    return Palette(colors=colors, provenance=PROV)


PAL4 = _palette(4)


def _flood_fill_reference(labels: np.ndarray) -> np.ndarray:
    """Brute-force 4-connected labeling in raster-scan first-occurrence order."""
    h, w = labels.shape
    out = np.full((h, w), -1, dtype=np.int32)
    next_id = 0
    for r in range(h):
        for c in range(w):
            if out[r, c] >= 0:
                continue
            stack = [(r, c)]
            out[r, c] = next_id
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and out[ny, nx] < 0
                        and labels[ny, nx] == labels[y, x]
                    ):
                        out[ny, nx] = next_id
                        stack.append((ny, nx))
            next_id += 1
    return out


def _boundary_identity(graph_labels: np.ndarray) -> None:
    lm = LabelMap(labels=graph_labels, provenance=PROV)
    graph = build_region_graph(lm, PAL4)
    h, w = graph_labels.shape
    total = (
        int((graph_labels[:, :-1] != graph_labels[:, 1:]).sum())
        + int((graph_labels[:-1, :] != graph_labels[1:, :]).sum())
        + 2 * (h + w)
    )
    assert sum(r.perimeter_px for r in graph.regions) == total + sum(
        w_len for *_, w_len in graph.edges
    )  # each shared crack is counted once per side


@settings(max_examples=50, deadline=None)
@given(
    st.integers(1, 8),
    st.integers(1, 8),
    st.integers(0, 2**31 - 1),
)
def test_agrees_with_flood_fill_and_identity(h: int, w: int, seed: int) -> None:
    labels = np.random.default_rng(seed).integers(0, 4, (h, w)).astype(np.int32)
    assert label_components(labels).tolist() == _flood_fill_reference(labels).tolist()
    _boundary_identity(labels)
