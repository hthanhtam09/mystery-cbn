"""PNG Preview renderer: line-art + solved (colored) previews
(ENGINE_SPEC.md §24, ARCHITECTURE.md §15 "render/png" row; Sprint 19
orchestration gap).

No prior implementation existed for a standalone PNG renderer (confirmed
absent: ``render/`` previously contained only ``svg.py``/``pdf.py``, and
``render_preview_png`` in ``pdf.py`` only rasterizes an already-built PDF --
it produces a single line-art-only image, never a "solved" flood-filled
variant, per the Sprint 18 architecture audit). This module is new code.

Both outputs flatten each face's Bézier rings with the same chord-density
sampling ``validate/common.py::flatten_face_rings`` and
``stages/layout/labels.py::_flatten_walk`` each already implement --
duplicated here rather than imported, since ARCHITECTURE.md §3's layer
graph places ``render``, ``validate``, and ``stages`` as siblings (no
cross-imports permitted; only ``model``/``foundation`` below are shared).
Uses Pillow -- the dependency ARCHITECTURE.md's dossier names for this
module (row: "render/png | ... | pyvips/Pillow").

- **Solved**: even-odd polygon fill per face (outer ring + holes) in
  ascending face_id order, hard edges (no anti-aliasing) -- this is the I1
  SSIM-probe input, so it must match the quantized label raster's per-pixel
  color classes exactly, not a cosmetically smoothed version of them
  (ENGINE_SPEC §24 step 3's stated reason for hard edges).
- **Line art**: white canvas, black stroked face boundaries plus printed
  numbers -- what the customer actually prints.
- **Colored**: the line art with every face additionally filled with its
  palette color -- same strokes and printed numbers as the outline, colors
  underneath (the "finished page" the customer is working toward).
- **Palette**: a standalone numbered swatch chart of the page's palette,
  chip per color labeled with the same one-char cell code the line art
  prints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mysterycbn.foundation.codes import code_for_number
from mysterycbn.foundation.errors import ConfigError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.layout import LabelPlan
from mysterycbn.model.records import Palette, Provenance
from mysterycbn.model.vector import CurveSet, Face

STAGE_NAME = "png"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

PREVIEW_DPI_DEFAULT = 150
_DEFAULT_PAGE_MM = (215.9, 279.4, 12.7)
_PT_PER_INCH = 72.0
_MM_PER_INCH = 25.4
_FLATTEN_MM = 0.1
_STROKE_RGB = (153, 153, 153)


def _label_font(size_px: float, cache: dict[int, ImageFont.FreeTypeFont]) -> ImageFont.FreeTypeFont:
    """Bundled DejaVu Sans at ``size_px`` (quantized to 0.25 px), cached.

    The label plan sizes each number so its DejaVu bbox fits the region's
    largest empty circle; rendering with any other font (or Pillow's
    fixed-size default) breaks that guarantee and spills numbers over the
    line art."""
    from mysterycbn.render.pdf import bundled_font_path

    key = max(1, round(size_px * 4))
    font = cache.get(key)
    if font is None:
        font = ImageFont.truetype(str(bundled_font_path()), key / 4.0)
        cache[key] = font
    return font


def _flatten_bezier(control: np.ndarray, tolerance_pt: float) -> np.ndarray:
    """Sample one cubic segment at chord-proportional density; last point
    dropped (matches ``validate/common.py``'s primitive, duplicated per this
    module's docstring since ``render`` may not import ``validate``)."""
    chord = float(
        np.linalg.norm(control[3] - control[0])
        + np.linalg.norm(control[1] - control[0])
        + np.linalg.norm(control[2] - control[1])
        + np.linalg.norm(control[3] - control[2])
    )
    n = int(np.clip(np.ceil(chord / (4.0 * tolerance_pt)), 2, 24))
    u = np.linspace(0.0, 1.0, n + 1)
    b = np.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u**2 * (1 - u), u**3], axis=1)
    return np.asarray(b @ control)[:-1]


def _flatten_face_rings(face: Face, curve_set: CurveSet, tolerance_pt: float) -> list[np.ndarray]:
    """Every ring (outer + holes) of ``face`` flattened to a closed polyline."""
    rings = []
    for walk in face.all_walks():
        parts = []
        for arc_id, rev in walk:
            segments = curve_set.curves[arc_id].segments
            for segment in reversed(segments) if rev else segments:
                pts = _flatten_bezier(segment.control, tolerance_pt)
                parts.append(pts[::-1] if rev else pts)
        rings.append(np.concatenate(parts))
    return rings


def _page_px(page_mm: tuple[float, float, float], dpi: int) -> tuple[int, int, float]:
    """(width_px, height_px, pt_to_px scale) for the given page + DPI."""
    width_mm, height_mm, _margin_mm = page_mm
    width_in = width_mm / _MM_PER_INCH
    height_in = height_mm / _MM_PER_INCH
    scale = dpi / _PT_PER_INCH  # px per pt
    return round(width_in * dpi), round(height_in * dpi), scale


def _to_px(ring: np.ndarray, scale: float) -> list[tuple[float, float]]:
    return [(float(x) * scale, float(y) * scale) for x, y in ring]


def render_solved_png(
    curve_set: CurveSet,
    palette: Palette,
    *,
    page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
    dpi: int = PREVIEW_DPI_DEFAULT,
) -> bytes:
    """Flood-filled preview: every face filled with its palette sRGB color,
    hard edges, no labels (ENGINE_SPEC §24 step 3)."""
    width_px, height_px, scale = _page_px(page_mm, dpi)
    tolerance_pt = _FLATTEN_MM * _MM_PER_INCH / _PT_PER_INCH  # mm -> pt
    img = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for face in sorted(curve_set.faces, key=lambda f: f.face_id):
        rings = _flatten_face_rings(face, curve_set, tolerance_pt)
        _fill_rings(draw, rings, _face_rgb(palette, face), scale)

    return _encode_png(img)


def _face_rgb(palette: Palette, face: Face) -> tuple[int, int, int]:
    r, g, b = (min(255, max(0, round(255 * c))) for c in palette.colors[face.label].srgb)
    return (r, g, b)


def _fill_rings(
    draw: ImageDraw.ImageDraw, rings: list[np.ndarray], rgb: tuple[int, int, int], scale: float
) -> None:
    """Even-odd fill of one face: outer ring in ``rgb``, holes back to white."""
    outer = _to_px(rings[0], scale)
    if len(outer) >= 3:
        draw.polygon(outer, fill=rgb)
    for hole in rings[1:]:
        hole_px = _to_px(hole, scale)
        if len(hole_px) >= 3:
            draw.polygon(hole_px, fill=(255, 255, 255))


def _encode_png(img: Image.Image) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _render_page_png(
    curve_set: CurveSet,
    label_plan: LabelPlan,
    palette: Palette | None,
    *,
    page_mm: tuple[float, float, float],
    dpi: int,
    stroke_px: int | None,
    blackout_ids: frozenset[int],
) -> bytes:
    """Shared line-art page renderer; with ``palette`` each face is filled
    with its color first, strokes and printed numbers drawn identically on
    top -- the colored variant differs from the outline only by the fills."""
    width_px, height_px, scale = _page_px(page_mm, dpi)
    tolerance_pt = _FLATTEN_MM * _MM_PER_INCH / _PT_PER_INCH
    img = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if stroke_px is None:
        # Match the SVG/PDF plan: 0.3 pt line weight scaled to this DPI. The
        # label plan budgets whitespace against that weight; a thicker preview
        # stroke eats clearance and makes numbers look like they touch lines.
        stroke_px = max(1, round(0.3 * scale))

    for face in sorted(curve_set.faces, key=lambda f: f.face_id):
        rings = _flatten_face_rings(face, curve_set, tolerance_pt)
        if face.face_id in blackout_ids:
            # Sliver too thin for any legible number: solid line-art fill,
            # no label (matches the SVG/PDF "blackout" layer).
            _fill_rings(draw, rings, _STROKE_RGB, scale)
        elif palette is not None:
            _fill_rings(draw, rings, _face_rgb(palette, face), scale)
        for ring in rings:
            pts = _to_px(ring, scale)
            if len(pts) >= 2:
                draw.line([*pts, pts[0]], fill=_STROKE_RGB, width=stroke_px, joint="curve")

    font_cache: dict[int, ImageFont.FreeTypeFont] = {}
    for label in label_plan.labels:
        x, y = label.anchor[0] * scale, label.anchor[1] * scale
        font = _label_font(label.font_size_pt * scale, font_cache)
        draw.text((x, y), code_for_number(label.printed_number), fill=(0, 0, 0), anchor="mm", font=font)

    return _encode_png(img)


def render_lineart_png(
    curve_set: CurveSet,
    label_plan: LabelPlan,
    *,
    page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
    dpi: int = PREVIEW_DPI_DEFAULT,
    stroke_px: int | None = None,
    filler_ids: frozenset[int] = frozenset(),  # noqa: ARG001 - kept for stage API compatibility
    blackout_ids: frozenset[int] = frozenset(),
) -> bytes:
    """White canvas, gray stroked face boundaries + printed numbers
    (ENGINE_SPEC §24 step 4) -- what the customer prints.

    Every face boundary is drawn at the same gray color and stroke width
    (no subject/filler distinction), matching the SVG/PDF renderers."""
    return _render_page_png(
        curve_set,
        label_plan,
        None,
        page_mm=page_mm,
        dpi=dpi,
        stroke_px=stroke_px,
        blackout_ids=blackout_ids,
    )


def render_colored_png(
    curve_set: CurveSet,
    label_plan: LabelPlan,
    palette: Palette,
    *,
    page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
    dpi: int = PREVIEW_DPI_DEFAULT,
    stroke_px: int | None = None,
    blackout_ids: frozenset[int] = frozenset(),
) -> bytes:
    """The outline page with its solution colors underneath: identical
    strokes and printed numbers to ``render_lineart_png``, plus each face
    filled with its palette color."""
    return _render_page_png(
        curve_set,
        label_plan,
        palette,
        page_mm=page_mm,
        dpi=dpi,
        stroke_px=stroke_px,
        blackout_ids=blackout_ids,
    )


_PALETTE_CHIP_PX = 96
_PALETTE_GAP_PX = 24
_PALETTE_MARGIN_PX = 48
_PALETTE_PER_ROW = 6


def render_palette_png(palette: Palette, *, permutation: tuple[int, ...] | None = None) -> bytes:
    """Numbered swatch chart of the page's palette: one bordered chip per
    color, labeled with the same one-char cell code the line art prints
    (``permutation`` is the legend's printed-number shuffle; identity when
    absent)."""
    n = len(palette.colors)
    if permutation is None:
        permutation = tuple(range(n))
    per_row = min(_PALETTE_PER_ROW, max(1, n))
    n_rows = -(-n // per_row)  # ceil division
    cell = _PALETTE_CHIP_PX + _PALETTE_GAP_PX
    label_h = _PALETTE_CHIP_PX // 2
    width = 2 * _PALETTE_MARGIN_PX + per_row * cell - _PALETTE_GAP_PX
    height = 2 * _PALETTE_MARGIN_PX + n_rows * (cell + label_h) - _PALETTE_GAP_PX
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_cache: dict[int, ImageFont.FreeTypeFont] = {}
    font = _label_font(float(label_h) * 0.6, font_cache)

    # Chips laid out in printed-number order so the chart reads 1, 2, 3, ...
    by_printed = sorted(range(n), key=lambda i: permutation[i])
    for pos, palette_index in enumerate(by_printed):
        row, col = divmod(pos, per_row)
        x = _PALETTE_MARGIN_PX + col * cell
        y = _PALETTE_MARGIN_PX + row * (cell + label_h)
        r, g, b = (
            min(255, max(0, round(255 * c))) for c in palette.colors[palette_index].srgb
        )
        draw.rectangle(
            [x, y, x + _PALETTE_CHIP_PX, y + _PALETTE_CHIP_PX],
            fill=(r, g, b),
            outline=_STROKE_RGB,
            width=2,
        )
        code = code_for_number(permutation[palette_index] + 1)
        draw.text(
            (x + _PALETTE_CHIP_PX / 2, y + _PALETTE_CHIP_PX + label_h / 2),
            code,
            fill=(0, 0, 0),
            anchor="mm",
            font=font,
        )

    return _encode_png(img)


@dataclass(frozen=True)
class PngPreviews:
    """Rendered PNG previews as a context-transportable artifact (matches
    the ``SvgDocument``/``PdfDocument`` pattern -- ``.previews`` carries the
    ``{"lineart", "solved", "colored", "palette"} -> bytes`` mapping
    ``OutputBundle`` expects)."""

    previews: Mapping[str, bytes]
    provenance: Provenance


class PngPreviewStage:
    """Stage wrapper: (``curve_set``, ``label_plan``, ``palette``) ->
    ``png_previews`` (a ``{"lineart", "solved", "colored", "palette"} ->
    bytes`` mapping)."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
    ) -> None:
        section = section or {}
        dpi = section.get("dpi", PREVIEW_DPI_DEFAULT)
        if not isinstance(dpi, int) or not 72 <= dpi <= 300:
            raise ConfigError(f"png config: dpi must be in [72, 300], got {dpi!r}")
        self._dpi = dpi
        self._page_mm = page_mm

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("curve_set", "label_plan", "palette")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("png_previews",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        curve_set = ctx.get("curve_set")
        label_plan = ctx.get("label_plan")
        palette = ctx.get("palette")
        if (
            not isinstance(curve_set, CurveSet)
            or not isinstance(label_plan, LabelPlan)
            or not isinstance(palette, Palette)
        ):
            raise ConfigError("png requires CurveSet + LabelPlan + Palette artifacts")
        filler_ids = (
            ctx.get("render_filler_region_ids")
            if ctx.has("render_filler_region_ids")
            else frozenset()
        )
        if not isinstance(filler_ids, (set, frozenset)):
            filler_ids = frozenset()
        blackout_ids = (
            ctx.get("blackout_region_ids") if ctx.has("blackout_region_ids") else frozenset()
        )
        if not isinstance(blackout_ids, (set, frozenset)):
            blackout_ids = frozenset()
        # Printed numbers on the palette chart must match the page's legend
        # shuffle; fall back to identity when no legend artifact is present
        # (e.g. unit tests running this stage in isolation).
        permutation: tuple[int, ...] | None = None
        if ctx.has("legend"):
            legend = ctx.get("legend")
            legend_permutation = getattr(legend, "permutation", None)
            if isinstance(legend_permutation, tuple):
                permutation = legend_permutation
        ctx.put(
            "png_previews",
            PngPreviews(
                previews={
                    "lineart": render_lineart_png(
                        curve_set,
                        label_plan,
                        page_mm=self._page_mm,
                        dpi=self._dpi,
                        filler_ids=frozenset(filler_ids),
                        blackout_ids=frozenset(blackout_ids),
                    ),
                    "solved": render_solved_png(
                        curve_set, palette, page_mm=self._page_mm, dpi=self._dpi
                    ),
                    "colored": render_colored_png(
                        curve_set,
                        label_plan,
                        palette,
                        page_mm=self._page_mm,
                        dpi=self._dpi,
                        blackout_ids=frozenset(blackout_ids),
                    ),
                    "palette": render_palette_png(palette, permutation=permutation),
                },
                provenance=Provenance(
                    stage_name=STAGE_NAME,
                    stage_version=STAGE_VERSION,
                    config_hash=_UNSET_HASH,
                    source_hash=curve_set.provenance.source_hash,
                ),
            ),
        )
