"""Large Region Split stage (ENGINE_SPEC.md §12).

Breaks oversized monotone regions into compact same-color numbered cells so
the page has fewer "boring continents" and reads closer to a commercial
color-by-number sheet. Runs after ``merge_tiny`` in the graph domain,
operating purely on the ``RegionGraph`` + ``component_map`` (still raster
coordinates -- no vector geometry yet).

Per region with ``area > A_max = split_factor * A_min``:

* target cell count ``k = ceil(area / (A_max / 2))``;
* seeds placed by farthest-point sampling on the region mask's distance
  transform (deterministic: first seed at the distance-transform argmax,
  ties broken by lowest flat pixel index);
* pixels assigned to the nearest seed in (row, col) space -- a discrete
  Voronoi split into compact cells (the §12.3 "flat" branch; the watershed
  variant for textured regions is a documented future increment and is not
  needed for the flat monotone regions this stage targets);
* every product cell inherits the parent's palette label, so the colorist
  paints the whole area one color, cell by cell.

Any product cell below ``A_min`` is folded back into its lowest-index
neighbouring cell of the same split so no sub-floor region is introduced
(mirrors ``merge_tiny``'s floor contract). The whole ``RegionGraph`` is then
rebuilt from the mutated component map so region records, adjacency, edges,
and every ``RegionGraph`` invariant are re-derived from scratch rather than
patched.

Disabled → identity (the graph passes through, re-stamped with this stage's
provenance).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import ndimage

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Palette, RegionGraph
from mysterycbn.stages.graph._organic_common import (
    connected_labels,
    fold_subfloor_regions,
    label_components,
    rebuild_region_graph,
    streamline_labels,
    to_component_input,
)
from mysterycbn.stages.graph.merge import area_floor_px

STAGE_NAME = "split_large"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

SPLIT_FACTOR_DEFAULT = 40.0
_SPLIT_FACTOR_MIN = 4.0
_SPLIT_FACTOR_MAX = 400.0

# How far (in mm, converted to working px via work_scale) an oversized
# region's own boundary is protected from splitting -- keeps the Voronoi grid
# off a subject's own silhouette/detail edges (ears, face, fur), matching a
# commercial sheet's cell grid, which only tiles the middle of open
# background/flat areas.
RIM_MM_DEFAULT = 2.0
_RIM_MM_MIN = 0.0
_RIM_MM_MAX = 20.0

# Domain-warp defaults (see _organic_common.flow_field / voronoi_labels): how strongly and
# at what spatial scale filler cell boundaries flow into organic, swirled
# shapes instead of straight Voronoi edges, matching a commercial sheet's
# wood-grain/marbled background tiling. 0 warp strength is bit-for-bit the
# old straight-edged behavior.
WARP_STRENGTH_MM_DEFAULT = 6.0
_WARP_STRENGTH_MM_MIN = 0.0
_WARP_STRENGTH_MM_MAX = 20.0
NOISE_SCALE_MM_DEFAULT = 18.0
_NOISE_SCALE_MM_MIN = 0.5
_NOISE_SCALE_MM_MAX = 50.0


def _split_component_map(
    component_map: np.ndarray,
    labels_of_region: list[int],
    *,
    a_min: float,
    split_factor: float,
    rim_px: float,
    warp_px: float = 0.0,
    noise_scale_px: float = 1.0,
    warp_seed: int = 0,
    incoming_filler_ids: frozenset[int] = frozenset(),
    incoming_rim_ids: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, list[int], set[int], set[int]]:
    """Return a new component map (dense ids), a parallel list giving each new
    region's palette label, the set of *filler* region ids (core cells AND
    rim cells -- exempt from the printability readable-size floor and given
    reduced curve-fit tolerance downstream), and the set of *render-filler*
    ids (core cells only, excluding rim -- used solely to pick bold vs. fine
    stroke weight when rendering, since a rim traces a real silhouette
    boundary and must render bold despite sharing the other filler
    treatment).

    ``rim_px``: an oversized region's pixels within this distance of its own
    boundary are never split -- they stay one connected "rim" cell tracing
    the region's original outline. Only the interior "core" (eroded by
    ``rim_px``) is Voronoi-split. Splitting flush to the boundary would carve
    the seams of a subject's own outline (ears, face, fur) into fragments
    wherever that outline neighbours the oversized region, which is exactly
    what a commercial sheet's cell grid avoids: filler tiling only touches
    the middle of open background/large flat areas, never right up against
    detailed silhouette edges."""
    a_max = split_factor * a_min
    h, w = component_map.shape
    flat = component_map.ravel()
    n = len(labels_of_region)
    areas = np.bincount(flat, minlength=n)

    new_flat = np.empty_like(flat)
    new_labels: list[int] = []
    # Per-pixel flag: did this pixel come from a subdivided (filler) cell?
    # (used for curve-fit tolerance + printability exemption -- includes rim)
    filler_pixel = np.zeros(flat.size, dtype=bool)
    # Per-pixel flag: is this pixel part of an un-split RIM cell specifically
    # (a subset of filler_pixel)? Used only to exclude rim cells from the
    # render-time bold/fine stroke split -- a rim traces the *original*
    # region outline (a real silhouette boundary against the subject next to
    # it), so it must render bold like any other real boundary, even though
    # it shares the filler-cell tolerance/printability treatment above.
    rim_pixel = np.zeros(flat.size, dtype=bool)
    next_id = 0
    for rid in range(n):
        pixels = np.flatnonzero(flat == rid)
        area = int(areas[rid])
        # A region already marked filler/rim by an upstream stage (e.g.
        # organic_partition) keeps that status when it passes through this
        # stage unsplit -- tracked per-pixel (not by id) since ids get
        # renumbered by the rebuild below, same reasoning as filler_pixel/
        # rim_pixel's own per-pixel tracking.
        was_filler = rid in incoming_filler_ids
        was_rim = rid in incoming_rim_ids
        if area <= a_max:
            new_flat[pixels] = next_id
            new_labels.append(labels_of_region[rid])
            if was_filler:
                filler_pixel[pixels] = True
            if was_rim:
                rim_pixel[pixels] = True
            next_id += 1
            continue

        mask = np.zeros(h * w, dtype=bool)
        mask[pixels] = True
        mask2d = mask.reshape(h, w)
        # Distance-to-background (O(N), one pass) rather than N iterations of
        # binary_erosion -- equivalent "erode by rim_px" result, but avoids a
        # per-iteration full-image sweep that made large regions pathologically
        # slow at real rim_px values (a few mm => tens of working px).
        dist_to_bg = ndimage.distance_transform_edt(mask2d)
        core2d = dist_to_bg > rim_px
        rim2d = mask2d & ~core2d
        core_area = int(core2d.sum())

        if core_area <= a_max:
            # No splittable interior left after carving out the rim (a small
            # or very thin oversized region) -- keep it whole, unsplit.
            new_flat[pixels] = next_id
            new_labels.append(labels_of_region[rid])
            if was_filler:
                filler_pixel[pixels] = True
            if was_rim:
                rim_pixel[pixels] = True
            next_id += 1
            continue

        target_cell_area = a_max / 2.0
        stroke_canvas = streamline_labels(
            core2d,
            target_cell_area=target_cell_area,
            warp_px=warp_px,
            noise_scale_px=noise_scale_px,
            seed=warp_seed + rid,
        )
        # Strokes bound the pockets but are not themselves inside any pocket;
        # 4-connected components of "core minus strokes" gives the enclosed
        # cells (organic, freely-crossing boundaries -- see
        # streamline_labels), then every stroke pixel is assigned to its
        # nearest enclosed pocket (one more distance-transform pass) so no
        # pixel is left unlabelled.
        pockets2d = core2d & ~stroke_canvas
        pocket_ids2d = label_components(to_component_input(pockets2d))
        if stroke_canvas.any() and pockets2d.any():
            _, (pr, pc) = ndimage.distance_transform_edt(~pockets2d, return_indices=True)
            cell_local2d = np.where(pockets2d, pocket_ids2d, pocket_ids2d[pr, pc])
        else:
            cell_local2d = pocket_ids2d
        cell_local = cell_local2d[core2d]  # local cell index per core pixel, boolean-mask order

        # Compact local cell ids to a contiguous range, assign global ids.
        # (Sub-floor folding is deferred to _fold_subfloor_regions, which runs
        # on the final *connected* regions -- folding here on raw stroke-
        # bounded cell ids can leave a disconnected sliver that re-splits
        # back below floor.)
        core_pixels = np.flatnonzero(core2d.ravel())
        used = sorted(set(cell_local.tolist()))
        remap = {c: i for i, c in enumerate(used)}
        for local in used:
            sel = core_pixels[cell_local == local]
            new_flat[sel] = next_id + remap[local]
            new_labels.append(labels_of_region[rid])
        filler_pixel[core_pixels] = True
        next_id += len(used)

        # The rim stays one un-split cell tracing the original outline. It
        # shares the filler cells' curve-fit tolerance/printability treatment
        # (despite its larger area, its shape is a thin, winding strip along
        # the subject's own detail edges, which the same fixed tolerance that
        # is fine for a normal region's smoother boundary can under-fit --
        # the fidelity gap observed on real photos), but it must NOT be
        # treated as a filler seam for rendering: it traces the region's
        # *original* silhouette against its neighbours, so its boundary is a
        # real one and stays bold (rim_pixel, tracked separately below).
        rim_pixels = np.flatnonzero(rim2d.ravel())
        if rim_pixels.size:
            new_flat[rim_pixels] = next_id
            new_labels.append(labels_of_region[rid])
            filler_pixel[rim_pixels] = True
            rim_pixel[rim_pixels] = True
            next_id += 1

    provisional = new_flat.reshape(h, w).astype(np.int32)
    # A Voronoi cell (nearest-seed over the whole plane) can be spatially
    # disconnected inside a non-convex region, and sub-floor folding merges by
    # id, not adjacency -- both can leave a single cell id as two separate
    # blobs, which breaks the "one region == one connected face" planar
    # invariant the topology stage checks (Euler identity). Re-run 4-connected
    # component labelling on the provisional cell-id map so every output region
    # is genuinely connected; carry each final component's palette label from
    # any of its pixels (all pixels of a provisional cell share one label).
    per_pixel_label = np.array(new_labels, dtype=np.int64)[provisional]
    final_map, final_labels = connected_labels(provisional, per_pixel_label)
    # Fold any *final* (connected) region below the floor into an adjacent
    # region of the same palette label, then re-densify. Folding earlier (on
    # provisional cell ids) is not enough: label_components can carve a folded
    # cell's disconnected sliver back into its own sub-floor region, which is
    # exactly what pushed tiny cells past printability. Doing it here, on
    # genuinely-connected regions, is the authoritative pass.
    final_map, final_labels = fold_subfloor_regions(
        final_map, final_labels, a_min=a_min
    )
    # A final region is "filler" if a majority of its pixels came from a
    # subdivided cell (robust to the fold occasionally absorbing a sliver of a
    # non-split region into a filler cell or vice versa); same rule for "rim".
    ff = final_map.ravel()
    n_final = int(ff.max()) + 1
    total = np.bincount(ff, minlength=n_final)
    filler_count = np.bincount(ff, weights=filler_pixel.astype(np.float64), minlength=n_final)
    filler_ids = {
        i for i in range(n_final) if filler_count[i] > 0 and filler_count[i] * 2 >= total[i]
    }
    rim_count = np.bincount(ff, weights=rim_pixel.astype(np.float64), minlength=n_final)
    rim_ids = {i for i in range(n_final) if rim_count[i] > 0 and rim_count[i] * 2 >= total[i]}
    # Render-time filler set excludes rim cells: a rim traces the region's
    # *original* silhouette (a real boundary), so it must render bold even
    # though it shares filler_ids' curve-fit-tolerance/printability treatment.
    render_filler_ids = filler_ids - rim_ids
    return final_map, final_labels, filler_ids, render_filler_ids


def split_large_regions(
    graph: RegionGraph,
    palette: Palette,
    *,
    a_min: float,
    split_factor: float = SPLIT_FACTOR_DEFAULT,
    rim_px: float = 1.0,
    warp_px: float = 0.0,
    noise_scale_px: float = 1.0,
    warp_seed: int = 0,
    incoming_filler_ids: frozenset[int] = frozenset(),
    incoming_rim_ids: frozenset[int] = frozenset(),
    config_hash: str = _UNSET_HASH,
) -> tuple[RegionGraph, frozenset[int], frozenset[int]]:
    """Full §12 split. Regions with ``area > split_factor * a_min`` are
    Voronoi-split in their interior (see ``rim_px`` on ``_split_component_map``
    for why the boundary is excluded); all others pass through. ``warp_px``/
    ``noise_scale_px`` bend the split into organic, swirled cell boundaries
    (see ``_flow_field``) instead of straight Voronoi edges. ``incoming_filler_ids``/
    ``incoming_rim_ids`` let an upstream stage (e.g. ``organic_partition``)
    mark regions filler/rim before this stage sees them -- that status is
    preserved (per-pixel, since ids get renumbered) for any such region that
    passes through this stage unsplit. Returns the rebuilt graph, the
    frozenset of *filler* region ids (core + rim, exempt from the
    printability readable-size floor / reduced curve-fit tolerance), and the
    frozenset of *render-filler* ids (core only, for picking bold vs. fine
    stroke weight -- see ``_split_component_map``)."""
    labels_of_region = [r.label for r in graph.regions]
    new_map, new_labels, filler_ids, render_filler_ids = _split_component_map(
        graph.component_map,
        labels_of_region,
        a_min=a_min,
        split_factor=split_factor,
        rim_px=rim_px,
        warp_px=warp_px,
        noise_scale_px=noise_scale_px,
        warp_seed=warp_seed,
        incoming_filler_ids=incoming_filler_ids,
        incoming_rim_ids=incoming_rim_ids,
    )
    new_graph = rebuild_region_graph(
        new_map,
        new_labels,
        palette,
        stage_name=STAGE_NAME,
        stage_version=STAGE_VERSION,
        config_hash=config_hash,
        source_hash=graph.provenance.source_hash,
    )
    return new_graph, frozenset(filler_ids), frozenset(render_filler_ids)


class SplitLargeStage:
    """Stage wrapper: (``region_graph``, ``palette``) → ``region_graph`` (split)."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        d_min_mm: float = 3.5,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        enabled = section.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"split config: enabled must be a bool, got {enabled!r}")
        factor = section.get("split_factor", SPLIT_FACTOR_DEFAULT)
        if not isinstance(factor, (int, float)) or not (
            _SPLIT_FACTOR_MIN <= float(factor) <= _SPLIT_FACTOR_MAX
        ):
            raise ConfigError(
                f"split config: split_factor must be in "
                f"[{_SPLIT_FACTOR_MIN}, {_SPLIT_FACTOR_MAX}], got {factor!r}"
            )
        rim_mm = section.get("rim_mm", RIM_MM_DEFAULT)
        if not isinstance(rim_mm, (int, float)) or not (
            _RIM_MM_MIN <= float(rim_mm) <= _RIM_MM_MAX
        ):
            raise ConfigError(
                f"split config: rim_mm must be in [{_RIM_MM_MIN}, {_RIM_MM_MAX}], got {rim_mm!r}"
            )
        warp_mm = section.get("warp_strength_mm", WARP_STRENGTH_MM_DEFAULT)
        if not isinstance(warp_mm, (int, float)) or not (
            _WARP_STRENGTH_MM_MIN <= float(warp_mm) <= _WARP_STRENGTH_MM_MAX
        ):
            raise ConfigError(
                f"split config: warp_strength_mm must be in "
                f"[{_WARP_STRENGTH_MM_MIN}, {_WARP_STRENGTH_MM_MAX}], got {warp_mm!r}"
            )
        noise_scale_mm = section.get("noise_scale_mm", NOISE_SCALE_MM_DEFAULT)
        if not isinstance(noise_scale_mm, (int, float)) or not (
            _NOISE_SCALE_MM_MIN <= float(noise_scale_mm) <= _NOISE_SCALE_MM_MAX
        ):
            raise ConfigError(
                f"split config: noise_scale_mm must be in "
                f"[{_NOISE_SCALE_MM_MIN}, {_NOISE_SCALE_MM_MAX}], got {noise_scale_mm!r}"
            )
        self._enabled = enabled
        self._split_factor = float(factor)
        self._rim_mm = float(rim_mm)
        self._warp_mm = float(warp_mm)
        self._noise_scale_mm = float(noise_scale_mm)
        self._d_min_mm = d_min_mm
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return (
            "region_graph",
            "palette",
            "raster_working",
            "filler_region_ids",
            "render_filler_region_ids",
        )

    @property
    def provides(self) -> tuple[str, ...]:
        return ("region_graph", "filler_region_ids", "render_filler_region_ids")

    @property
    def config_section(self) -> str:
        return "split"

    def run(self, ctx: PipelineContext) -> None:
        graph = ctx.get("region_graph")
        palette = ctx.get("palette")
        raster = ctx.get("raster_working")
        incoming_filler_ids = ctx.get("filler_region_ids")
        incoming_rim_ids = ctx.get("filler_region_ids") - ctx.get("render_filler_region_ids")
        if not isinstance(graph, RegionGraph) or not isinstance(palette, Palette):
            raise ConfigError("split_large requires RegionGraph + Palette artifacts")
        if not self._enabled:
            # Preserve any filler/rim status an upstream stage (organic_
            # partition) already established -- this stage being disabled
            # must not silently wipe out that state.
            ctx.put("region_graph", graph)
            ctx.put("filler_region_ids", incoming_filler_ids)
            ctx.put("render_filler_region_ids", ctx.get("render_filler_region_ids"))
            return
        work_scale = getattr(raster, "work_scale", 0.0)
        a_min = area_floor_px(self._d_min_mm, work_scale)
        ppmm = 1.0 / (work_scale * MM_PER_INCH / PT_PER_INCH) if work_scale > 0 else 1.0
        new_graph, filler_ids, render_filler_ids = split_large_regions(
            graph,
            palette,
            a_min=a_min,
            split_factor=self._split_factor,
            incoming_filler_ids=frozenset(incoming_filler_ids),
            incoming_rim_ids=frozenset(incoming_rim_ids),
            rim_px=self._rim_mm * ppmm,
            warp_px=self._warp_mm * ppmm,
            noise_scale_px=self._noise_scale_mm * ppmm,
            warp_seed=ctx.seed,
            config_hash=self._config_hash,
        )
        ctx.put("region_graph", new_graph)
        ctx.put("filler_region_ids", filler_ids)
        ctx.put("render_filler_region_ids", render_filler_ids)
