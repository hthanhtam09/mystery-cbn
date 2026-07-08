"""Renderer-agreement contract (ARCHITECTURE.md §"Contract tests",
ENGINE_SPEC §23): PDF-space and SVG-space arc positions agree within
0.05 pt. Both renderers consume the same plans; this test proves neither
one distorts the shared geometry."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("reportlab")
fitz = pytest.importorskip("fitz")

from mysterycbn.model.layout import Legend
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.render.pdf import render_pdf
from mysterycbn.render.svg import render_svg
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import place_labels
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph

PROV = Provenance("labels", "1.0.0", "0" * 64, "1" * 64)
PAGE_MM = (215.9, 279.4, 12.7)
TOLERANCE_PT = 0.05
NS = "{http://www.w3.org/2000/svg}"


def _plans(seed: int):
    rng = np.random.default_rng(seed)
    k = 6
    base = np.repeat(np.repeat(rng.integers(0, k, (6, 8)), 8, axis=0), 8, axis=1)
    palette = Palette(
        colors=tuple(
            PaletteColor.from_lab(i, (10.0 + 15.0 * i, 5.0 * i - 12.0, 8.0 - 3.0 * i), 100)
            for i in range(k)
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
        permutation=tuple(range(k)),
        chips=tuple((i, (40.0 + 60.0 * i, 745.0), 17.0) for i in range(k)),
        band_rect=(36.0, 740.0, 540.0, 27.0),
        provenance=PROV,
        number_font_pt=8.0,
    )
    return curve_set, plan, legend, palette


def _svg_arc_segments(svg_data: bytes) -> np.ndarray:
    """Cubic control quads ``(N, 4, 2)`` from the regions layer, path order.

    Each SVG ``C`` command yields (start, c1, c2, end); the start point is
    the previous command's endpoint (or the ``M`` point).
    """
    root = ET.fromstring(svg_data)
    regions = root.find(f"{NS}g[@id='regions']")
    assert regions is not None
    quads = []
    for path in regions.findall(f"{NS}path"):
        numbers = [float(v) for v in re.findall(r"-?\d+\.\d+", path.get("d", ""))]
        points = list(zip(numbers[::2], numbers[1::2], strict=True))
        current = points[0]
        for i in range(1, len(points), 3):
            quads.append([current, points[i], points[i + 1], points[i + 2]])
            current = points[i + 2]
    return np.asarray(quads)


def _pdf_arc_segments(pdf_data: bytes) -> np.ndarray:
    """Cubic control quads ``(N, 4, 2)`` from the PDF drawings, stream order
    (fitz reports them in the page's y-down coordinate frame)."""
    quads = []
    with fitz.open(stream=pdf_data, filetype="pdf") as doc:
        for drawing in doc[0].get_drawings():
            if drawing["type"] != "s":  # arcs are stroke-only; chips are fill+stroke
                continue
            for item in drawing["items"]:
                if item[0] == "c":  # cubic Bézier: p0, c1, c2, p3
                    quads.append([(p.x, p.y) for p in item[1:]])
    return np.asarray(quads)


def _sample(quads: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Evaluate every cubic at parameters ``ts`` — deterministic samples
    *along* the arcs, not just at control points."""
    p0, c1, c2, p3 = (quads[:, i, None, :] for i in range(4))
    t = ts[None, :, None]
    u = 1.0 - t
    return (u**3 * p0 + 3 * u**2 * t * c1 + 3 * u * t**2 * c2 + t**3 * p3).reshape(-1, 2)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_pdf_svg_geometric_agreement(seed: int) -> None:
    curve_set, plan, legend, palette = _plans(seed)
    svg = _svg_arc_segments(render_svg(curve_set, plan, legend, palette, page_mm=PAGE_MM))
    pdf = _pdf_arc_segments(render_pdf(curve_set, plan, legend, palette, page_mm=PAGE_MM))
    assert svg.shape == pdf.shape, "renderers must emit the same segments in the same order"
    ts = np.linspace(0.0, 1.0, max(7, -(-1000 // len(svg))))  # ≥ 1000 samples per fixture
    svg_samples, pdf_samples = _sample(svg, ts), _sample(pdf, ts)
    assert len(svg_samples) >= 1000
    worst = float(np.max(np.hypot(*(svg_samples - pdf_samples).T)))
    assert worst <= TOLERANCE_PT, f"worst deviation {worst} pt > {TOLERANCE_PT} pt"
