"""Golden byte-hash for the SVG renderer — invariant I2's test surface.

Digest = SHA-256 of the exact output bytes on a deterministic pipeline run
(components → topology → arc graph → curves → labels → svg, seed-0 fixture).
Any byte change — attribute order, coordinate formatting, layer order — is
a reviewed golden-update event (BENCHMARK_SPEC §4.3).
"""

from __future__ import annotations

import hashlib

import numpy as np

from mysterycbn.model.layout import Legend
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.render.svg import render_svg, validate_svg
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import place_labels
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph

# Updated for the added (empty when ink disabled) "ink" layer between
# "blackout" and "labels" -- render-only line-art overlay stage.
# Updated again: the ink layer's stroke color changed from black (#000) to
# the same gray used for region-boundary strokes (#999), so a black outline
# no longer gives the subject's silhouette away.
_GOLDEN = "d85fa49fca8a8b80b52d4b888f0276c44678806e5c450e1b059038d96435c07e"

PROV = Provenance("denoise", "1.0.0", "0" * 64, "1" * 64)
PAGE_MM = (215.9, 279.4, 12.7)


def _render() -> bytes:
    rng = np.random.default_rng(0)
    base = np.repeat(np.repeat(rng.integers(0, 6, (6, 8)), 8, axis=0), 8, axis=1)
    palette = Palette(
        colors=tuple(
            PaletteColor.from_lab(i, (10.0 + 15.0 * i, 5.0 * i - 12.0, 8.0 - 3.0 * i), 100)
            for i in range(6)
        ),
        provenance=PROV,
    )
    rg = build_region_graph(LabelMap(labels=base.astype(np.int32), provenance=PROV), palette)
    ag = build_arc_graph(
        build_topology_graph(rg.component_map), rg, content_box=content_box_pt(PAGE_MM)
    )
    curve_set = fit_curves(ag)
    plan, _ = place_labels(curve_set, rg)
    legend = Legend(
        permutation=tuple(range(6)),
        chips=tuple((i, (40.0 + 60.0 * i, 745.0), 17.0) for i in range(6)),
        band_rect=(36.0, 740.0, 540.0, 27.0),
        number_font_pt=8.0,
        provenance=PROV,
    )
    data = render_svg(curve_set, plan, legend, palette, page_mm=PAGE_MM)
    validate_svg(data, curve_set)
    return data


def test_golden_svg_bytes() -> None:
    assert hashlib.sha256(_render()).hexdigest() == _GOLDEN
