"""No-Color Mask stage: mark the largest subject regions as "no_color"
(the "partial" preset -- tô màu MỘT PHẦN subject).

Motivation (product): a partial color-by-number page keeps the full dense
treatment on the parts the user is meant to color, but deliberately leaves
the biggest regions of the subject *un-numbered* -- they still carry their
outline (line art) on the page, but bear no number and claim no legend
color. The colorist fills the rest; the largest flat masses are left as the
artist's blank canvas, matching the "partial" reference look.

Selection: auto-detect by area. The regions whose ``area_px`` fall in the
top ``top_area_percentile`` fraction of the (already merge-compacted) region
areas are marked no_color. This is the user-chosen mask rule for the engine
(area-based auto-detect, no hand-drawn mask): "the N% largest regions".

Position in the pipeline: immediately AFTER ``merge_tiny`` (so it selects
over clean, consolidated, floor-legal regions and the compacted palette) and
BEFORE ``organic_partition``. Every downstream stage that subdivides or
renumbers regions (``organic_partition``, ``split_large``) threads the
no_color set through per-pixel, exactly as it already threads the filler /
rim sets, so a no_color region that gets subdivided yields no_color cells --
every product cell of a no_color region is itself no_color (no number,
no legend color).

The no_color set is published as ``ctx.put("no_color_region_ids", ...)``,
a frozenset of region ids into the graph as it stands right after this
stage. It is remapped downstream just like ``filler_region_ids``.

Disabled → publishes an empty frozenset (identity). The stage is disabled
in every built-in preset except "partial", so no existing preset's output
changes (the empty set makes every consumer's no_color branch a no-op).

What "no_color" means downstream (distinct from the existing ``blackout``
concept, which solid-fills a sliver):
  * ``labels`` skips appending any Label for a no_color face entirely --
    no in-region number, no leader, no micro-label (it is not a blackout
    solid fill either; the region keeps only its outline stroke).
  * ``legend``/``merge`` recompute palette coverage from labeled
    (non-no_color) regions only: a palette color used *solely* by no_color
    regions is dropped from the legend and from the color count, so no
    legend chip shows a number that never appears on the page.
  * the printability validator's coverage/size gates exempt no_color faces
    (they legitimately carry no label), the same way filler faces are
    exempt.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Palette, PaletteColor, Provenance, RegionGraph
from mysterycbn.stages.graph._organic_common import rebuild_region_graph

STAGE_NAME = "mask"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64
# Grayscale threshold splitting "paint" (< 128) from "no_color" (>= 128) when a
# non-binary bitmap is supplied. A pure 0/1 mask lands on either side cleanly.
_MASK_THRESHOLD = 128


def decode_bitmap_mask(bitmap_b64: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode a base64 PNG hand-drawn mask into a boolean ``(H, W)`` array
    aligned to ``shape`` (the component map's ``(rows, cols)``).

    Bitmap format (what the web canvas sends):
      * A PNG image, base64-encoded (an optional ``data:image/png;base64,``
        data-URI prefix is tolerated and stripped).
      * Pixel semantics: **0 = paint this region** (the colorist fills it),
        **1 = no_color** (left blank -- outline only, no number/legend color).
        Grayscale uploads are accepted too and thresholded at
        ``_MASK_THRESHOLD`` (>= 128 → no_color), so an 8-bit 0/255 mask works
        unchanged.
      * Any alpha channel is ignored (the mask is read from luminance); the
        image is converted to mode "L" before thresholding.

    Resize behaviour: the decoded mask is resized to ``shape`` (the working-
    scale component-map grid the regions live on) with nearest-neighbour
    resampling, so binary values are preserved -- never blurred into
    intermediate grays -- regardless of the canvas preview resolution. The
    caller's grid, not the original upload, is authoritative: a mask drawn at
    800x600 over an image whose working raster is 1600x1200 maps correctly as
    long as the **aspect ratio matches**. A mismatched aspect ratio is still
    accepted (stretched to fit), but the mask will be geometrically skewed --
    the web UI is responsible for drawing at the source aspect ratio.

    Raises ``ConfigError`` on undecodable base64 or an unreadable/oversized
    PNG.
    """
    raw = bitmap_b64.strip()
    if raw.startswith("data:"):
        # Tolerate a data-URI prefix ("data:image/png;base64,....").
        _, _, raw = raw.partition(",")
    try:
        data = base64.b64decode(raw, validate=True)
    except ValueError as exc:  # binascii.Error subclasses ValueError
        raise ConfigError(f"mask config: bitmap is not valid base64: {exc}") from exc
    if not data:
        raise ConfigError("mask config: bitmap decoded to empty bytes")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ConfigError(f"mask config: cannot decode bitmap PNG: {exc}") from exc
    rows, cols = int(shape[0]), int(shape[1])
    gray = img.convert("L")
    if gray.size != (cols, rows):  # PIL size is (width, height) == (cols, rows)
        gray = gray.resize((cols, rows), Image.NEAREST)
    arr = np.asarray(gray, dtype=np.uint8)
    return arr >= _MASK_THRESHOLD


def select_no_color_region_ids_from_bitmap(
    graph: RegionGraph, mask: np.ndarray
) -> frozenset[int]:
    """Region ids covered by the hand-drawn ``mask`` (a boolean ``(H, W)``
    array aligned to ``graph.component_map``).

    A region is marked no_color iff **any** of its pixels falls under the mask
    (mask value True). This "any-pixel" rule matches the web contract: the user
    paints over the regions they want left blank; a stroke that clips a region
    at all claims the whole region (regions are atomic -- a region carries a
    single number, so it is either numbered or not). Empty selection (mask all
    False) yields an empty frozenset, an identity no-op downstream."""
    cmap = graph.component_map
    if mask.shape != cmap.shape:  # pragma: no cover - decode aligns them
        raise ConfigError(
            f"mask config: bitmap shape {mask.shape} != component map {cmap.shape}"
        )
    hit = np.unique(cmap[mask])
    return frozenset(int(rid) for rid in hit.tolist())


def select_no_color_region_ids(
    graph: RegionGraph, *, top_area_percentile: float
) -> frozenset[int]:
    """Region ids whose ``area_px`` fall in the top ``top_area_percentile``
    fraction of all regions by area (ties broken toward the larger area, then
    the lower id -- deterministic).

    ``top_area_percentile`` is the *fraction* of regions to mark, counted from
    the largest downward: 0.5 marks the larger half. 0.0 marks none; 1.0 marks
    all. The count is ``ceil(n * fraction)`` so a nonzero fraction on a small
    page always marks at least one region (the single largest)."""
    n = len(graph.regions)
    if n == 0 or top_area_percentile <= 0.0:
        return frozenset()
    # ceil so a nonzero fraction never rounds down to "mark nothing" -- the
    # smallest partial page (a handful of regions) still gets its largest
    # region left blank, which is the whole point of the preset.
    k = min(n, int(np.ceil(n * top_area_percentile)))
    if k <= 0:
        return frozenset()
    # Deterministic order: descending area, then ascending id (the region_id
    # is the natural tiebreak the rest of the engine already sorts on).
    order = sorted(graph.regions, key=lambda r: (-r.area_px, r.region_id))
    return frozenset(r.region_id for r in order[:k])


def drop_no_color_only_colors(
    graph: RegionGraph,
    palette: Palette,
    no_color_ids: frozenset[int],
    *,
    config_hash: str = _UNSET_HASH,
) -> tuple[RegionGraph, Palette]:
    """Recompute palette coverage from the LABELED (non-no_color) regions and
    drop any palette color used *only* by no_color regions, so no legend chip
    ends up showing a number that never appears on the printed page.

    A no_color region keeps its pixels and its outline -- only its palette
    *color* is suppressed. If a color survives on some labeled region too, it
    stays. If it survives only on no_color regions, it is dropped and the
    palette is recompacted (surviving colors renumbered densely, preserving
    coverage-descending order), exactly like ``merge_tiny``'s compaction. All
    region labels are remapped to the compacted palette (no_color regions
    included -- they still need a valid in-range label for the frozen model,
    they simply never print it). Region ids are preserved (the component map
    is untouched), so the ``no_color_region_ids`` set stays valid.

    Skipped (returns the inputs unchanged) when nothing would be dropped, or
    when the drop would leave < 2 colors (the model's palette floor): a legend
    needs ≥ 2 colors, so in that degenerate case the "unused" color is kept
    rather than violate the floor -- the same K ≥ 2 guard ``merge_tiny`` uses.
    """
    if not no_color_ids:
        return graph, palette
    labels = np.array([r.label for r in graph.regions], dtype=np.int64)
    areas = np.array([r.area_px for r in graph.regions], dtype=np.int64)
    is_no_color = np.array(
        [r.region_id in no_color_ids for r in graph.regions], dtype=bool
    )
    labeled_coverage = np.bincount(
        labels[~is_no_color], weights=areas[~is_no_color], minlength=palette.size
    )
    kept = np.flatnonzero(labeled_coverage > 0)
    if kept.size == palette.size or kept.size < 2:
        # Nothing to drop, or dropping would fall below the ≥ 2-color floor.
        return graph, palette

    # Full coverage (all regions, no_color included) drives the retained
    # colors' recorded coverage, matching merge_tiny's PaletteColor.coverage
    # semantics (coverage is total painted area of that color, not just
    # labeled area).
    full_coverage = np.bincount(labels, weights=areas, minlength=palette.size)
    renumber = np.full(palette.size, -1, dtype=np.int64)
    renumber[kept] = np.arange(kept.size)
    new_palette = Palette(
        colors=tuple(
            PaletteColor.from_lab(
                int(renumber[i]), palette.colors[i].lab, int(full_coverage[i])
            )
            for i in kept
        ),
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=palette.provenance.source_hash,
        ),
        min_delta_e=palette.min_delta_e,
    )
    # Remap every region's label to the compacted palette. A no_color region's
    # color was, by construction of ``kept``, only ever no_color -- but a
    # no_color region can also legitimately share a still-kept color (another
    # labeled region uses it too); either way ``renumber`` gives it a valid
    # in-range label (the dropped-color case cannot occur for a kept region,
    # and a no_color region whose sole color was dropped is remapped to its
    # nearest surviving color so the frozen model stays satisfied).
    new_labels: list[int] = []
    for r in graph.regions:
        mapped = int(renumber[r.label])
        if mapped < 0:
            # This region's color was dropped (it was a no_color region whose
            # color no labeled region shares). It never prints a number or a
            # legend color, so its exact label is cosmetic -- assign the
            # perceptually nearest surviving color to keep a valid label.
            nearest = min(
                kept.tolist(),
                key=lambda k, lab=r.label: float(palette.delta_e_table[lab, k]),
            )
            mapped = int(renumber[nearest])
        new_labels.append(mapped)

    new_graph = rebuild_region_graph(
        graph.component_map,
        new_labels,
        new_palette,
        stage_name=STAGE_NAME,
        stage_version=STAGE_VERSION,
        config_hash=config_hash,
        source_hash=graph.provenance.source_hash,
    )
    return new_graph, new_palette


class NoColorMaskStage:
    """Stage wrapper: ``region_graph`` → ``region_graph`` (unchanged) +
    ``no_color_region_ids``.

    This stage never mutates the graph or palette -- a no_color region keeps
    its outline and its pixels; only its *labeling* (number + legend color)
    is suppressed downstream. Selection is published as a context frozenset,
    mirroring ``organic_partition``'s ``filler_region_ids`` contract, so no
    ``Region`` dataclass field is added (the model stays frozen)."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        config_hash: str = "0" * 64,
    ) -> None:
        section = section or {}
        enabled = section.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"mask config: enabled must be a bool, got {enabled!r}")
        pct = section.get("top_area_percentile", 0.5)
        if not isinstance(pct, (int, float)) or not (0.0 <= float(pct) <= 1.0):
            raise ConfigError(
                f"mask config: top_area_percentile must be in [0.0, 1.0], got {pct!r}"
            )
        bitmap = section.get("bitmap", None)
        if bitmap is not None and not isinstance(bitmap, str):
            raise ConfigError(
                f"mask config: bitmap must be a base64 string or None, got {bitmap!r}"
            )
        self._enabled = enabled
        self._top_area_percentile = float(pct)
        # A non-empty bitmap string switches the stage from area auto-detect to
        # the hand-drawn mask sent by the web UI (None/"" → auto-detect).
        self._bitmap = bitmap or None
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("region_graph", "palette")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("region_graph", "palette", "no_color_region_ids")

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        graph = ctx.get("region_graph")
        palette = ctx.get("palette")
        if not isinstance(graph, RegionGraph) or not isinstance(palette, Palette):
            raise ConfigError("mask requires RegionGraph + Palette artifacts")
        if not self._enabled:
            # Every non-"partial" preset: identity. The empty set makes each
            # downstream no_color branch a no-op; the graph/palette are re-put
            # unchanged to satisfy this stage's provides-contract, so no
            # existing golden output changes.
            ctx.put("region_graph", graph)
            ctx.put("palette", palette)
            ctx.put("no_color_region_ids", frozenset())
            return
        if self._bitmap is not None:
            # Hand-drawn mask from the web UI: decode the base64 PNG, align it
            # to the region grid, and select the regions the user painted over
            # (aspect-ratio-matched; nearest-neighbour resize keeps it binary).
            mask = decode_bitmap_mask(self._bitmap, graph.component_map.shape)
            no_color_ids = select_no_color_region_ids_from_bitmap(graph, mask)
        else:
            # Fallback: area-based auto-detect (the top N% largest regions).
            no_color_ids = select_no_color_region_ids(
                graph, top_area_percentile=self._top_area_percentile
            )
        # Drop any palette color used *only* by no_color regions, so the legend
        # never shows a number that appears nowhere on the page (recompacts the
        # palette + remaps region labels; region ids are preserved, so the
        # no_color set stays valid).
        new_graph, new_palette = drop_no_color_only_colors(
            graph, palette, no_color_ids, config_hash=self._config_hash
        )
        ctx.put("region_graph", new_graph)
        ctx.put("palette", new_palette)
        ctx.put("no_color_region_ids", no_color_ids)
