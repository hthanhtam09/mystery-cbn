"""PDF Export: the self-contained print deliverable (ENGINE_SPEC.md §23).

Native vector re-render of the same plans the SVG renderer consumes — no
SVG rasterization, no conversion layer. Both outputs sit downstream of the
*same* geometry, which is what the renderer-agreement contract test proves
(arc positions within 0.05 pt of §22's SVG space).

Page = exact trim size from config, in points. The engine's page frame is
y-down (SVG convention); PDF is y-up, so the y-axis is flipped exactly once
at the canvas transform (§1.3 / MATH_SPEC §1) — no intermediate y-up frame
exists anywhere else. Text is drawn through the same flip with a local
counter-flip per anchor so glyphs are not mirrored; central vertical
alignment is computed from the embedded font's ascent/descent.

Fonts: the bundled OFL-licensed DejaVu Sans (``assets/fonts/``), pinned by
SHA-256 and embedded as a subset. No system font may ever be referenced —
cross-machine determinism of *metrics*. The PDF bytes themselves are not
hash-gated (ReportLab object numbering is not canonical across versions);
the golden surface is the page *content stream* plus geometric agreement.

Metadata: title, engine version and resolved-config hash in the Info dict;
ReportLab's invariant mode pins the creation date and file ID (no wall
clock — same inputs, same bytes within a ReportLab version).

The 300 DPI raster preview (``render_preview_png``) rasterizes the finished
PDF via PyMuPDF, so the preview shows exactly what the print file contains.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mysterycbn.foundation.codes import code_for_number
from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.layout import LabelMode, LabelPlan, Legend
from mysterycbn.model.records import Palette, Provenance
from mysterycbn.model.vector import CurveSet

STAGE_NAME = "pdf"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

STROKE_PT_DEFAULT = 0.3
_FILLER_STROKE_PT_DEFAULT = 0.12  # matches render/svg.py's fine seam stroke
PREVIEW_DPI_DEFAULT = 300
_LEADER_STROKE_PT = 0.25
_CHIP_CORNER_PT = 1.5
_CHIP_PAD_PT = 2.0
_DEFAULT_PAGE_MM = (215.9, 279.4, 12.7)

FONT_NAME = "DejaVuSans"
_FONT_FILE = "DejaVuSans.ttf"
FONT_SHA256 = "7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954"
_ASSETS_FONTS = Path(__file__).resolve().parents[3] / "assets" / "fonts"


@dataclass(frozen=True)
class PdfDocument:
    """Rendered PDF bytes (plus optional preview) as a context artifact."""

    data: bytes
    preview_png: bytes | None
    provenance: Provenance


def _fail(message: str) -> StageError:
    return StageError(message, stage_name=STAGE_NAME, config_hash=_UNSET_HASH)


def bundled_font_path() -> Path:
    """Locate and integrity-check the bundled font (asset pinned by hash).

    Raises ``StageError`` if the asset is missing or its SHA-256 does not
    match the pin — a corrupted or substituted font would silently change
    printed metrics across machines.
    """
    path = _ASSETS_FONTS / _FONT_FILE
    if not path.is_file():
        raise _fail(f"bundled font missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FONT_SHA256:
        raise _fail(f"bundled font hash mismatch: {digest} != pinned {FONT_SHA256}")
    return path


def _register_font() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(bundled_font_path())))


def _central_baseline_y(font_size_pt: float) -> float:
    """Offset from a text row's vertical center down to its baseline, so
    glyphs sit centrally (SVG ``dominant-baseline=central`` equivalent)."""
    from reportlab.pdfbase import pdfmetrics

    ascent, descent = pdfmetrics.getAscentDescent(FONT_NAME, font_size_pt)
    return (float(ascent) + float(descent)) / 2.0


def render_pdf(
    curve_set: CurveSet,
    label_plan: LabelPlan,
    legend: Legend,
    palette: Palette,
    *,
    page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
    stroke_pt: float = STROKE_PT_DEFAULT,
    title: str = "mystery-cbn puzzle",
    config_hash: str = _UNSET_HASH,
    filler_ids: frozenset[int] = frozenset(),  # noqa: ARG001 - kept for stage API compatibility
    filler_stroke_pt: float | None = None,  # noqa: ARG001 - kept for stage API compatibility
) -> bytes:
    """Draw the full page natively (§23): same primitives, same order as
    the SVG renderer — arcs, labels, leaders, legend, frame. Every arc is
    drawn at the same gray stroke color/width (no subject/filler distinction),
    matching the SVG renderer."""
    from reportlab.lib.colors import Color, black
    from reportlab.pdfgen.canvas import Canvas

    _register_font()
    width_mm, height_mm, margin_mm = page_mm
    to_pt = PT_PER_INCH / MM_PER_INCH
    width_pt, height_pt, margin_pt = width_mm * to_pt, height_mm * to_pt, margin_mm * to_pt

    buffer = io.BytesIO()
    # initialFontName overrides ReportLab's Helvetica default, which would
    # otherwise appear in the page resources as an unembedded system font.
    canvas = Canvas(
        buffer,
        pagesize=(width_pt, height_pt),
        invariant=1,
        pageCompression=0,
        initialFontName=FONT_NAME,
    )
    canvas.setTitle(title)
    canvas.setCreator(f"mystery-cbn pdf {STAGE_VERSION}")
    canvas.setSubject(f"config:{config_hash}")
    canvas.setProducer("mystery-cbn")

    # The single y-flip (§1.3): page frame is y-down, PDF user space y-up.
    canvas.translate(0.0, height_pt)
    canvas.scale(1.0, -1.0)

    gray = Color(0.6, 0.6, 0.6)
    canvas.setLineCap(1)
    canvas.setLineJoin(1)
    canvas.setStrokeColor(gray)
    canvas.setFillColor(black)

    # Regions: one path per arc, each shared boundary exactly once, id order,
    # all drawn at the same stroke color/width (no subject/filler distinction).
    canvas.setLineWidth(stroke_pt)
    for arc_id in sorted(c.arc_id for c in curve_set.curves):
        curve = next(c for c in curve_set.curves if c.arc_id == arc_id)
        path = canvas.beginPath()
        first = curve.segments[0].control[0]
        path.moveTo(float(first[0]), float(first[1]))
        for segment in curve.segments:
            c = segment.control
            path.curveTo(
                float(c[1][0]),
                float(c[1][1]),
                float(c[2][0]),
                float(c[2][1]),
                float(c[3][0]),
                float(c[3][1]),
            )
        if bool((curve.segments[0].control[0] == curve.segments[-1].control[3]).all()):
            path.close()
        canvas.drawPath(path, stroke=1, fill=0)

    # Labels: centered on the anchor, counter-flipped so glyphs read upright.
    for label in label_plan.labels:
        canvas.saveState()
        canvas.translate(label.anchor[0], label.anchor[1])
        canvas.scale(1.0, -1.0)
        canvas.setFont(FONT_NAME, label.font_size_pt)
        canvas.drawCentredString(
            0.0, -_central_baseline_y(label.font_size_pt), code_for_number(label.printed_number)
        )
        canvas.restoreState()

    # Leaders.
    canvas.setLineWidth(_LEADER_STROKE_PT)
    for label in label_plan.labels:
        if label.mode is not LabelMode.LEADER or label.leader is None:
            continue
        (x1, y1), (x2, y2) = label.leader
        canvas.line(x1, y1, x2, y2)

    # Legend: chips (rounded rect, palette sRGB fill, black outline) + numbers.
    canvas.setLineWidth(stroke_pt)
    for palette_index, (cx, cy), side in legend.chips:
        r, g, b = palette.colors[palette_index].srgb
        canvas.setFillColor(Color(r, g, b))
        canvas.roundRect(cx, cy, side, side, _CHIP_CORNER_PT, stroke=1, fill=1)
        canvas.setFillColor(black)
        canvas.saveState()
        canvas.translate(cx + side + _CHIP_PAD_PT, cy + side / 2.0)
        canvas.scale(1.0, -1.0)
        canvas.setFont(FONT_NAME, legend.number_font_pt)
        canvas.drawString(
            0.0,
            -_central_baseline_y(legend.number_font_pt),
            code_for_number(legend.printed_number(palette_index)),
        )
        canvas.restoreState()

    # Frame: content-box furniture.
    canvas.rect(margin_pt, margin_pt, width_pt - 2 * margin_pt, height_pt - 2 * margin_pt)

    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def render_preview_png(pdf_data: bytes, *, dpi: int = PREVIEW_DPI_DEFAULT) -> bytes:
    """Rasterize page 1 of the finished PDF at ``dpi`` (default 300) so the
    preview shows exactly what the print file contains."""
    import fitz  # PyMuPDF

    with fitz.open(stream=pdf_data, filetype="pdf") as doc:
        page = doc[0]
        zoom = dpi / PT_PER_INCH
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return bytes(pixmap.tobytes("png"))


def validate_pdf(data: bytes, *, page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM) -> None:
    """Structural validation (§23 quality contract): parseable single-page
    PDF, exact trim (media) box, and every referenced font embedded."""
    import fitz

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise _fail(f"PDF is not parseable: {exc}") from exc
    with doc:
        if doc.page_count != 1:
            raise _fail(f"expected a single page, got {doc.page_count}")
        page = doc[0]
        to_pt = PT_PER_INCH / MM_PER_INCH
        expected = (page_mm[0] * to_pt, page_mm[1] * to_pt)
        actual = (page.rect.width, page.rect.height)
        if any(abs(a - e) > 1e-3 for a, e in zip(actual, expected, strict=True)):
            raise _fail(f"trim box {actual} != expected {expected} pt")
        fonts = page.get_fonts(full=True)
        if not fonts:
            raise _fail("no font present — labels require the embedded bundled font")
        for xref, *_ in fonts:
            extracted = doc.extract_font(xref)
            if not extracted or not extracted[-1]:
                raise _fail(f"font xref {xref} is referenced but not embedded")


class PdfExportStage:
    """Stage wrapper: (``curve_set``, ``label_plan``, ``legend``,
    ``palette``) → ``pdf`` (validated ``PdfDocument`` + 300 DPI preview)."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        stroke = section.get("stroke_pt", STROKE_PT_DEFAULT)
        preview_dpi = section.get("preview_dpi", PREVIEW_DPI_DEFAULT)
        if not isinstance(stroke, (int, float)) or not 0.05 <= float(stroke) <= 2.0:
            raise ConfigError(f"pdf config: stroke_pt must be in [0.05, 2], got {stroke!r}")
        if not isinstance(preview_dpi, int) or not 72 <= preview_dpi <= 1200:
            raise ConfigError(f"pdf config: preview_dpi must be in [72, 1200], got {preview_dpi!r}")
        filler_stroke = section.get("filler_stroke_pt", _FILLER_STROKE_PT_DEFAULT)
        if not isinstance(filler_stroke, (int, float)) or not 0.02 <= float(filler_stroke) <= 2.0:
            raise ConfigError(
                f"pdf config: filler_stroke_pt must be in [0.02, 2], got {filler_stroke!r}"
            )
        self._filler_stroke = float(filler_stroke)
        self._stroke = float(stroke)
        self._preview_dpi = preview_dpi
        self._page_mm = page_mm
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("curve_set", "label_plan", "legend", "palette")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("pdf",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        curve_set = ctx.get("curve_set")
        label_plan = ctx.get("label_plan")
        legend = ctx.get("legend")
        palette = ctx.get("palette")
        if (
            not isinstance(curve_set, CurveSet)
            or not isinstance(label_plan, LabelPlan)
            or not isinstance(legend, Legend)
            or not isinstance(palette, Palette)
        ):
            raise ConfigError("pdf requires CurveSet + LabelPlan + Legend + Palette artifacts")
        filler_ids = (
            ctx.get("render_filler_region_ids")
            if ctx.has("render_filler_region_ids")
            else frozenset()
        )
        if not isinstance(filler_ids, (set, frozenset)):
            filler_ids = frozenset()
        data = render_pdf(
            curve_set,
            label_plan,
            legend,
            palette,
            page_mm=self._page_mm,
            stroke_pt=self._stroke,
            config_hash=self._config_hash,
            filler_ids=frozenset(filler_ids),
            filler_stroke_pt=self._filler_stroke,
        )
        validate_pdf(data, page_mm=self._page_mm)
        ctx.put(
            "pdf",
            PdfDocument(
                data=data,
                preview_png=render_preview_png(data, dpi=self._preview_dpi),
                provenance=Provenance(
                    stage_name=STAGE_NAME,
                    stage_version=STAGE_VERSION,
                    config_hash=self._config_hash,
                    source_hash=curve_set.provenance.source_hash,
                ),
            ),
        )
