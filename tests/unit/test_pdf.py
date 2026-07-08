"""Unit tests for the PDF Export renderer (ENGINE_SPEC §23)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

pytest.importorskip("reportlab")
fitz = pytest.importorskip("fitz")

from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.layout import Legend
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.render.pdf import (
    FONT_SHA256,
    PdfDocument,
    PdfExportStage,
    bundled_font_path,
    render_pdf,
    render_preview_png,
    validate_pdf,
)
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import place_labels
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph

PROV = Provenance("labels", "1.0.0", "0" * 64, "1" * 64)
PAGE_MM = (215.9, 279.4, 12.7)
PAL4 = Palette(
    colors=tuple(PaletteColor.from_lab(i, (10.0 + 25.0 * i, 0.0, 0.0), 100) for i in range(4)),
    provenance=PROV,
)


def _fixture(rows: list[list[int]]):
    lm = LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV)
    rg = build_region_graph(lm, PAL4)
    ag = build_arc_graph(
        build_topology_graph(rg.component_map), rg, content_box=content_box_pt(PAGE_MM)
    )
    curve_set = fit_curves(ag)
    plan, findings = place_labels(curve_set, rg)
    assert findings == ()
    legend = Legend(
        permutation=(0, 1, 2, 3),
        chips=tuple((i, (40.0 + 40.0 * i, 740.0), 17.0) for i in range(4)),
        band_rect=(36.0, 736.0, 540.0, 25.0),
        number_font_pt=8.0,
        provenance=PROV,
    )
    return curve_set, plan, legend


# An asymmetric layout: distinct region bands top vs bottom, so any y-flip
# mistake moves every label to the wrong half of the page.
ASYM = [[0] * 8] * 2 + [[1] * 8] * 4 + [[2] * 8] * 10


def _render(rows=None) -> bytes:
    curve_set, plan, legend = _fixture(rows or ASYM)
    return render_pdf(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)


def test_font_asset_pinned() -> None:
    import hashlib

    path = bundled_font_path()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FONT_SHA256


def test_trim_box_exact() -> None:
    data = _render()
    with fitz.open(stream=data, filetype="pdf") as doc:
        page = doc[0]
        to_pt = PT_PER_INCH / MM_PER_INCH
        assert page.rect.width == pytest.approx(PAGE_MM[0] * to_pt, abs=1e-3)
        assert page.rect.height == pytest.approx(PAGE_MM[1] * to_pt, abs=1e-3)


def test_y_flip_places_labels_correctly() -> None:
    """Anchors are y-down page coords; fitz reports words in the same frame,
    so the printed numbers must land on their anchors after the flip."""
    (
        curve_set,
        plan,
        legend,
    ) = _fixture(ASYM)
    data = render_pdf(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)
    with fitz.open(stream=data, filetype="pdf") as doc:
        words = doc[0].get_text("words")
    region_words = {}
    for x0, y0, x1, y1, text, *_ in words:
        region_words.setdefault(text, []).append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    for label in plan.labels:
        candidates = region_words[str(label.printed_number)]
        assert any(
            abs(cx - label.anchor[0]) < 1.0 and abs(cy - label.anchor[1]) < 1.0
            for cx, cy in candidates
        ), f"label {label.printed_number} not at anchor {label.anchor}"


def test_fonts_embedded_and_no_system_font() -> None:
    data = _render()
    with fitz.open(stream=data, filetype="pdf") as doc:
        fonts = doc[0].get_fonts(full=True)
        assert fonts, "labels must reference the bundled font"
        for xref, _ext, _type, basename, *_ in fonts:
            assert basename.endswith("DejaVuSans"), f"unexpected font {basename}"
            extracted = doc.extract_font(xref)
            assert extracted[-1], f"font {basename} referenced but not embedded"


def test_deterministic_bytes_and_fixed_date() -> None:
    a, b = _render(), _render()
    assert a == b
    assert b"D:20000101000000+00'00'" in a  # invariant-mode fixed epoch, no wall clock
    assert b"config:" + b"0" * 64 in a  # resolved-config hash in the Info dict


def test_metadata() -> None:
    with fitz.open(stream=_render(), filetype="pdf") as doc:
        meta = doc.metadata
    assert meta["title"] == "mystery-cbn puzzle"
    assert meta["creator"].startswith("mystery-cbn pdf ")
    assert meta["subject"].startswith("config:")


def test_preview_png_is_300_dpi() -> None:
    png = render_preview_png(_render())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert width == round(PAGE_MM[0] / MM_PER_INCH * 300)
    assert height == round(PAGE_MM[1] / MM_PER_INCH * 300)


def test_validate_pdf_rejects_garbage_and_wrong_trim() -> None:
    with pytest.raises(StageError):
        validate_pdf(b"%PDF-1.4 truncated")
    with pytest.raises(StageError, match="trim box"):
        validate_pdf(_render(), page_mm=(100.0, 100.0, 10.0))


def test_stage_config_validation() -> None:
    with pytest.raises(ConfigError):
        PdfExportStage({"stroke_pt": 5.0})
    with pytest.raises(ConfigError):
        PdfExportStage({"preview_dpi": 10})


def test_stage_end_to_end() -> None:
    curve_set, plan, legend = _fixture(ASYM)
    ctx = InMemoryContext(seed=0)
    ctx.put("curve_set", curve_set)
    ctx.put("label_plan", plan)
    ctx.put("legend", legend)
    ctx.put("palette", PAL4)
    stage = PdfExportStage(page_mm=PAGE_MM)
    assert stage.requires == ("curve_set", "label_plan", "legend", "palette")
    assert stage.provides == ("pdf",)
    stage.run(ctx)
    doc = ctx.get("pdf")
    assert isinstance(doc, PdfDocument)
    validate_pdf(doc.data, page_mm=PAGE_MM)
    assert doc.preview_png is not None and doc.preview_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert doc.provenance.stage_name == "pdf"
