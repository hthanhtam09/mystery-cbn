"""SVG Export: the canonical byte-deterministic output renderer
(ENGINE_SPEC.md §22).

A direct string serializer — no svgwrite/lxml (writer libraries do not
guarantee attribute ordering across versions; I2's byte-identical promise
cannot be delegated). Document structure, in fixed layer order:

- ``<g id="regions">`` — one ``<path>`` per **arc** (each shared boundary
  drawn once: half the ink, no double-stroke darkening), ``M``/``C``
  commands from the Bézier chains, ``data-left``/``data-right`` printed
  numbers (0 = page exterior).
- ``<g id="labels">`` — ``<text>`` at each anchor, middle/central aligned,
  bundled font family by name.
- ``<g id="leaders">`` — 0.25 pt ``<line>`` per leader label.
- ``<g id="legend">`` — chips (rounded rects filled with the palette sRGB,
  0.3 pt outline) + numbers per the §21 geometry carried in the Legend
  artifact.
- ``<g id="frame">`` — the content-box frame (page furniture).

Print-safe contract: explicit physical size (``width``/``height`` in mm,
``viewBox`` in pt), pure black line art, embedded-nothing (fonts referenced
by name — the PDF is the self-contained deliverable), no scripts, no
external references, no timestamps.

Determinism rules (I2's test surface): every coordinate formatted to exactly
``decimals`` places with negative zero normalized; elements emitted in id
order; attribute order fixed by this serializer; LF newlines; the only
comment is the engine version (constant per build).

``validate_svg`` re-checks the structural contract on the produced bytes:
well-formed XML, fixed layer order, arc-once, declared physical size.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass

from mysterycbn.foundation.codes import code_for_number
from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.ink import InkOverlay
from mysterycbn.model.layout import LabelMode, LabelPlan, Legend
from mysterycbn.model.records import Palette, Provenance
from mysterycbn.model.vector import CurveSet
from mysterycbn.render.seams import arc_sides_and_faces, thin_seam_arc_ids

STAGE_NAME = "svg"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

STROKE_PT_DEFAULT = 0.3
DECIMALS_DEFAULT = 3
_LEADER_STROKE_PT = 0.25
_STROKE_COLOR = "#999"
# All region arcs (subject silhouette and filler-seam boundaries alike)
# draw at the same weight and gray color -- see _render_region_arcs.
_FILLER_STROKE_PT_DEFAULT = STROKE_PT_DEFAULT
_FONT_FAMILY = "DejaVu Sans"
_CHIP_CORNER_PT = 1.5
_CHIP_PAD_PT = 2.0
_DEFAULT_PAGE_MM = (215.9, 279.4, 12.7)


@dataclass(frozen=True)
class SvgDocument:
    """Rendered SVG bytes as a context-transportable artifact."""

    data: bytes
    provenance: Provenance


def _fail(message: str) -> StageError:
    return StageError(message, stage_name=STAGE_NAME, config_hash=_UNSET_HASH)


def format_coord(value: float, decimals: int = DECIMALS_DEFAULT) -> str:
    """Fixed-decimals coordinate with negative zero normalized (§22.5)."""
    out = format(value, f".{decimals}f")
    if out == format(-0.0, f".{decimals}f") and out.startswith("-"):
        return out[1:]
    return out


def _srgb_hex(srgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(255 * channel):02x}" for channel in srgb)


def _path_d(curve, decimals: int, closed: bool) -> str:  # type: ignore[no-untyped-def]
    parts = []
    first = curve.segments[0].control[0]
    parts.append(f"M {format_coord(first[0], decimals)} {format_coord(first[1], decimals)}")
    for segment in curve.segments:
        c = segment.control
        parts.append(
            "C "
            + " ".join(
                f"{format_coord(c[i][0], decimals)} {format_coord(c[i][1], decimals)}"
                for i in (1, 2, 3)
            )
        )
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _face_path_d(face, curve_set: CurveSet, decimals: int) -> str:  # type: ignore[no-untyped-def]
    """Even-odd fillable path of a face: one closed subpath per walk (outer
    ring + holes), Bézier-exact (reversed arcs emit reversed control order)."""
    parts: list[str] = []
    for walk in face.all_walks():
        first_arc, first_rev = walk[0]
        segments = curve_set.curves[first_arc].segments
        start = segments[-1].control[3] if first_rev else segments[0].control[0]
        parts.append(f"M {format_coord(start[0], decimals)} {format_coord(start[1], decimals)}")
        for arc_id, rev in walk:
            arc_segments = curve_set.curves[arc_id].segments
            for segment in reversed(arc_segments) if rev else arc_segments:
                c = segment.control[::-1] if rev else segment.control
                parts.append(
                    "C "
                    + " ".join(
                        f"{format_coord(c[i][0], decimals)} {format_coord(c[i][1], decimals)}"
                        for i in (1, 2, 3)
                    )
                )
        parts.append("Z")
    return " ".join(parts)


def _render_blackout(
    curve_set: CurveSet, blackout_ids: frozenset[int], decimals: int
) -> list[str]:
    """Solid line-art-color fill for slivers that carry no number (too thin
    for legible ink); always emitted so the layer order stays fixed."""
    lines = [f'<g id="blackout" fill="{_STROKE_COLOR}" stroke="none" fill-rule="evenodd">']
    faces_by_id = {f.face_id: f for f in curve_set.faces}
    for face_id in sorted(blackout_ids):
        face = faces_by_id.get(face_id)
        if face is None:
            continue
        lines.append(f'<path id="blackout-{face_id}" d="{_face_path_d(face, curve_set, decimals)}"/>')
    lines.append("</g>")
    return lines


def _render_ink(overlay: "InkOverlay | None", stroke_pt: float, decimals: int) -> list[str]:
    """Ink line work as black polylines (preserved thin dark lines). Always
    emitted -- empty when the ink stages are disabled -- so the layer order
    stays fixed. Render-only: never a region, never numbered."""
    width = format_coord(stroke_pt if overlay is None else overlay.stroke_pt, decimals)
    lines = [
        f'<g id="ink" fill="none" stroke="#000" stroke-width="{width}" '
        'stroke-linecap="round" stroke-linejoin="round">'
    ]
    if overlay is not None:
        for i, poly in enumerate(overlay.polylines):
            pts = " ".join(
                f"{format_coord(x, decimals)},{format_coord(y, decimals)}" for x, y in poly
            )
            lines.append(f'<polyline id="ink-{i}" points="{pts}"/>')
    lines.append("</g>")
    return lines


def _render_region_arcs(
    curve_set: CurveSet,
    number_of: dict[int, int],
    stroke: str,
    decimals: int,
) -> list[str]:
    """One ``<path>`` per arc, each shared boundary drawn exactly once, in
    the same single ``regions`` group and fixed ascending arc-id order as
    before (the ``regions`` layer's "every arc, in id order" contract that
    ``validate_svg``/the PDF renderer rely on is unchanged). Every arc shares
    the same stroke color and width -- no subject/filler distinction."""
    arcs_by_id = {c.arc_id: c for c in curve_set.curves}
    sides, _ = arc_sides_and_faces(curve_set, number_of)

    lines = [
        f'<g id="regions" fill="none" stroke="{_STROKE_COLOR}" stroke-width="{stroke}" '
        'stroke-linecap="round" stroke-linejoin="round">'
    ]
    for arc_id in sorted(arcs_by_id):
        curve = arcs_by_id[arc_id]
        left, right = sides.get(arc_id, (0, 0))
        closed = bool((curve.segments[0].control[0] == curve.segments[-1].control[3]).all())
        lines.append(
            f'<path id="arc-{arc_id}" data-left="{left}" data-right="{right}" '
            f'd="{_path_d(curve, decimals, closed)}"/>'
        )
    lines.append("</g>")
    return lines


def render_svg(
    curve_set: CurveSet,
    label_plan: LabelPlan,
    legend: Legend,
    palette: Palette,
    *,
    page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
    stroke_pt: float = STROKE_PT_DEFAULT,
    decimals: int = DECIMALS_DEFAULT,
    filler_ids: frozenset[int] = frozenset(),  # noqa: ARG001 - kept for stage API compatibility
    filler_stroke_pt: float | None = None,  # noqa: ARG001 - kept for stage API compatibility
    blackout_ids: frozenset[int] = frozenset(),
    ink_overlay: "InkOverlay | None" = None,
) -> bytes:
    """Serialize the full page (§22). Byte-deterministic by construction.

    ``filler_ids``/``filler_stroke_pt`` are accepted for backward API
    compatibility but no longer change the output: every arc is drawn at the
    same ``stroke_pt`` weight and gray color (see ``_render_region_arcs``)."""
    width_mm, height_mm, margin_mm = page_mm
    to_pt = PT_PER_INCH / MM_PER_INCH
    width_pt, height_pt, margin_pt = width_mm * to_pt, height_mm * to_pt, margin_mm * to_pt
    stroke = format_coord(stroke_pt, decimals)
    number_of = {label.region_id: label.printed_number for label in label_plan.labels}

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- mystery-cbn svg {STAGE_VERSION} -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{format_coord(width_mm, decimals)}mm" '
        f'height="{format_coord(height_mm, decimals)}mm" '
        f'viewBox="0 0 {format_coord(width_pt, decimals)} {format_coord(height_pt, decimals)}">',
    ]

    # Regions: one path per arc, each shared boundary exactly once, all at
    # the same stroke color/width (see _render_region_arcs).
    lines.extend(_render_region_arcs(curve_set, number_of, stroke, decimals))

    # Blackout slivers: solid fill, no number (see _render_blackout).
    lines.extend(_render_blackout(curve_set, blackout_ids, decimals))

    # Ink: preserved thin dark line work as black polylines (see _render_ink).
    lines.extend(_render_ink(ink_overlay, stroke_pt, decimals))

    # Labels.
    lines.append(
        f'<g id="labels" font-family="{_FONT_FAMILY}" fill="#000" '
        'text-anchor="middle" dominant-baseline="central">'
    )
    for label in label_plan.labels:
        lines.append(
            f'<text id="label-{label.region_id}" '
            f'x="{format_coord(label.anchor[0], decimals)}" '
            f'y="{format_coord(label.anchor[1], decimals)}" '
            f'font-size="{format_coord(label.font_size_pt, decimals)}">'
            f"{code_for_number(label.printed_number)}</text>"
        )
    lines.append("</g>")

    # Leaders.
    leader_stroke = format_coord(_LEADER_STROKE_PT, decimals)
    lines.append(f'<g id="leaders" stroke="#000" stroke-width="{leader_stroke}">')
    for label in label_plan.labels:
        if label.mode is not LabelMode.LEADER or label.leader is None:
            continue
        (x1, y1), (x2, y2) = label.leader
        lines.append(
            f'<line id="leader-{label.region_id}" '
            f'x1="{format_coord(x1, decimals)}" y1="{format_coord(y1, decimals)}" '
            f'x2="{format_coord(x2, decimals)}" y2="{format_coord(y2, decimals)}"/>'
        )
    lines.append("</g>")

    # Legend (geometry from the Legend artifact, §21).
    corner = format_coord(_CHIP_CORNER_PT, decimals)
    number_pt = format_coord(legend.number_font_pt, decimals)
    lines.append(f'<g id="legend" font-family="{_FONT_FAMILY}" font-size="{number_pt}">')
    for palette_index, (cx, cy), side in legend.chips:
        color = _srgb_hex(palette.colors[palette_index].srgb)
        lines.append(
            f'<rect id="chip-{palette_index}" '
            f'x="{format_coord(cx, decimals)}" y="{format_coord(cy, decimals)}" '
            f'width="{format_coord(side, decimals)}" height="{format_coord(side, decimals)}" '
            f'rx="{corner}" fill="{color}" stroke="#000" stroke-width="{stroke}"/>'
        )
        lines.append(
            f'<text id="chip-number-{palette_index}" '
            f'x="{format_coord(cx + side + _CHIP_PAD_PT, decimals)}" '
            f'y="{format_coord(cy + side / 2.0, decimals)}" '
            f'fill="#000" dominant-baseline="central">'
            f"{code_for_number(legend.printed_number(palette_index))}</text>"
        )
    lines.append("</g>")

    # Frame: content-box furniture.
    lines.append(f'<g id="frame" fill="none" stroke="#000" stroke-width="{stroke}">')
    lines.append(
        f'<rect id="content-frame" '
        f'x="{format_coord(margin_pt, decimals)}" y="{format_coord(margin_pt, decimals)}" '
        f'width="{format_coord(width_pt - 2 * margin_pt, decimals)}" '
        f'height="{format_coord(height_pt - 2 * margin_pt, decimals)}"/>'
    )
    lines.append("</g>")

    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode("utf-8")


_LAYER_ORDER = ("regions", "blackout", "ink", "labels", "leaders", "legend", "frame")


def validate_svg(data: bytes, curve_set: CurveSet | None = None) -> None:
    """Structural validation of rendered bytes (§22 quality contract):
    well-formed XML, SVG 1.1 root with physical size, fixed layer order,
    and (when ``curve_set`` is given) each arc appearing exactly once."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise _fail(f"SVG is not well-formed XML: {exc}") from exc
    ns = "{http://www.w3.org/2000/svg}"
    if root.tag != f"{ns}svg":
        raise _fail(f"root element is {root.tag}, not svg")
    if root.get("version") != "1.1" or root.get("viewBox") is None:
        raise _fail("svg root must declare version 1.1 and a viewBox")
    for dim in ("width", "height"):
        value = root.get(dim, "")
        if not value.endswith("mm"):
            raise _fail(f"{dim} must carry an explicit physical (mm) size")
    groups = [g.get("id") for g in root if g.tag == f"{ns}g"]
    if tuple(groups) != _LAYER_ORDER:
        raise _fail(f"layer order {groups} != {list(_LAYER_ORDER)}")
    if curve_set is not None:
        regions = root.find(f"{ns}g[@id='regions']")
        assert regions is not None
        path_ids = [p.get("id") for p in regions.findall(f"{ns}path")]
        expected = [f"arc-{c.arc_id}" for c in curve_set.curves]
        if path_ids != expected:
            raise _fail("regions layer must contain each arc exactly once, in id order")


class SvgExportStage:
    """Stage wrapper: (``curve_set``, ``label_plan``, ``legend``,
    ``palette``) → ``svg`` (validated)."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        stroke = section.get("stroke_pt", STROKE_PT_DEFAULT)
        decimals = section.get("decimals", DECIMALS_DEFAULT)
        filler_stroke = section.get("filler_stroke_pt", _FILLER_STROKE_PT_DEFAULT)
        if not isinstance(stroke, (int, float)) or not 0.05 <= float(stroke) <= 2.0:
            raise ConfigError(f"svg config: stroke_pt must be in [0.05, 2], got {stroke!r}")
        if not isinstance(decimals, int) or not 2 <= decimals <= 5:
            raise ConfigError(f"svg config: decimals must be in [2, 5], got {decimals!r}")
        if not isinstance(filler_stroke, (int, float)) or not 0.02 <= float(filler_stroke) <= 2.0:
            raise ConfigError(
                f"svg config: filler_stroke_pt must be in [0.02, 2], got {filler_stroke!r}"
            )
        self._stroke = float(stroke)
        self._decimals = decimals
        self._filler_stroke = float(filler_stroke)
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
        return ("svg",)

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
            raise ConfigError("svg requires CurveSet + LabelPlan + Legend + Palette artifacts")
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
        ink_overlay = ctx.get("ink_overlay") if ctx.has("ink_overlay") else None
        if not isinstance(ink_overlay, InkOverlay):
            ink_overlay = None
        data = render_svg(
            curve_set,
            label_plan,
            legend,
            palette,
            page_mm=self._page_mm,
            stroke_pt=self._stroke,
            decimals=self._decimals,
            filler_ids=frozenset(filler_ids),
            filler_stroke_pt=self._filler_stroke,
            blackout_ids=frozenset(blackout_ids),
            ink_overlay=ink_overlay,
        )
        validate_svg(data, curve_set)
        ctx.put(
            "svg",
            SvgDocument(
                data=data,
                provenance=Provenance(
                    stage_name=STAGE_NAME,
                    stage_version=STAGE_VERSION,
                    config_hash=self._config_hash,
                    source_hash=curve_set.provenance.source_hash,
                ),
            ),
        )
