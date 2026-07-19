"""Organic Region Partition stage (ADR-003).

Subdivides eligible single-color regions into organic, hand-drafted-looking
cells -- flowing spline-friendly boundaries, ribbon-like elongated cells, and
occasional nested "island" sub-cells -- as an alternative to the blocky/
axis-aligned boundaries that fall out of pixel quantization + connected-
component labeling. Runs after ``merge_tiny`` (so it partitions clean,
consolidated same-color masks) and before ``split_large`` (which still acts
as a safety net for any pathologically large leftover cell). Operates purely
on the ``RegionGraph`` + ``component_map`` in the graph/pixel domain -- the
same pixel-perfect contract every stage up through ``split_large`` shares
(ADR-002), since ``topology``'s crack-arc extraction requires a dense int32
``component_map`` regardless of how regions were seeded.

Every organic cell -- and every nested island -- inherits its parent
region's palette label unchanged: this stage only ever subdivides *shape*
within one color, never reassigns or crosses palette labels, so
color-by-number semantics (one number = one palette color) are fully
preserved.

Disabled → identity (the graph passes through, re-stamped with this stage's
provenance), matching ``merge_tiny``/``split_large``'s contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Literal

import numpy as np
from scipy import ndimage

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Palette, RegionGraph
from mysterycbn.stages.graph._organic_common import (
    connected_labels,
    farthest_point_seeds,
    flow_field,
    fold_regions_where,
    fold_subfloor_regions,
    region_inradius_px,
    grid_seeds,
    label_components,
    rebuild_region_graph,
    streamline_labels,
    to_component_input,
    voronoi_labels,
)

STAGE_NAME = "organic_partition"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

OrganicMode = Literal["voronoi", "streamline", "mixed"]
_MODES: tuple[OrganicMode, ...] = ("voronoi", "streamline", "mixed")

# Skip organic-partitioning the page background by default: without this, a
# large flat backdrop gets shattered into hundreds of small organic cells
# (confirmed via rendered preview -- reads as visual noise, and the many new
# region boundaries running close to a subject's real silhouette edge look
# like an unwanted doubled outline even though rim_mm=0 keeps the silhouette
# itself a single line). A region is "background" if it touches the page
# border and is the largest such region -- mirrors components.py's own
# border_len concept (the crack-adjacency count against the page edge).
SKIP_BACKGROUND_DEFAULT = True

# Fold any region whose palette color is this dark (LAB L* at or below this
# threshold, 0 = black, 100 = white) into a neighboring region *before*
# organic-partitioning -- not merely skipped/left untouched. A source image
# with a pre-drawn cartoon/line-art outline (e.g. a black ink silhouette
# stroke) quantizes that outline into its own thin, ring-shaped region
# (real stroke width means real area, touching both the background and the
# subject fill on its two opposite edges). Leaving it as its own standalone
# region -- even a deliberately un-partitioned one -- does not help: the
# ring's two edges are real geometry regardless of what happens inside it,
# so organic-partitioning its *neighbors* still traces both edges as real
# silhouette boundaries, which reads as a doubled outline right next to the
# subject (confirmed via rendered preview on a real cartoon photo). Folding
# removes the ring as a distinct shape entirely, leaving one silhouette
# edge. Real "subject" colors are essentially never this dark, so folding
# near-black regions targets outline strokes specifically without touching
# legitimate dark subjects (which are usually well above this threshold --
# true ink black is rare outside of a drawn outline).
SKIP_DARK_LAB_L_THRESHOLD = 15.0

# The dark fold above only targets outline STROKES -- thin ring-shaped
# regions. A large solid dark mass (black hair, a navy garment) can share the
# same near-black palette entry; folding it recolors the entire feature into
# whichever neighbour absorbs it (observed: near-black hair coming out brown
# on character art). A drawn stroke is at most a few mm wide, so a dark
# region whose inscribed-disk radius exceeds this bound is treated as real
# subject art and kept.
DARK_FOLD_MAX_INRADIUS_MM = 1.5

MIN_AREA_MM2_DEFAULT = 150.0
_MIN_AREA_MM2_MIN = 10.0
_MIN_AREA_MM2_MAX = 5000.0

# Tuned against a reference "liquid blob" color-by-number style (large,
# sparse, freely-shaped cells rather than small confetti-like ones) --
# visually validated via rendered preview at these defaults before landing.
SEED_DENSITY_MM2_DEFAULT = 400.0
_SEED_DENSITY_MM2_MIN = 1.0
_SEED_DENSITY_MM2_MAX = 2000.0

# 0 by default: a nonzero rim carves the eroded core out as its own region,
# which introduces a *second* boundary between rim and core right next to
# the region's real silhouette edge -- reads as an unwanted "double outline"
# around the subject (confirmed via rendered preview). split_large's rim
# exists for the same "protect the silhouette" reason but its cells are tiny
# filler cells deep inside a busy background, where a doubled edge is
# invisible; organic_partition's cells are large and the artifact was
# clearly visible. Callers who want the silhouette-protection behavior can
# still opt in via rim_mm.
RIM_MM_DEFAULT = 0.0
_RIM_MM_MIN = 0.0
_RIM_MM_MAX = 20.0

WARP_STRENGTH_MM_DEFAULT = 10.0
_WARP_STRENGTH_MM_MIN = 0.0
_WARP_STRENGTH_MM_MAX = 20.0

NOISE_SCALE_MM_DEFAULT = 30.0
_NOISE_SCALE_MM_MIN = 0.5
_NOISE_SCALE_MM_MAX = 60.0

# 0 by default: an elongated ribbon cell can end up running right alongside
# a subject's real silhouette edge (e.g. hugging the inside of an ear), and
# its near-parallel inner edge reads as an unwanted "double outline" even
# though it is a genuine organic cell boundary, not a duplicated silhouette
# (confirmed via rendered preview). Callers who want the more branching/
# vein-like look can still opt in via ribbon_elongation.
RIBBON_ELONGATION_DEFAULT = 0.0
_RIBBON_ELONGATION_MIN = 0.0
_RIBBON_ELONGATION_MAX = 1.0

ISLAND_PROBABILITY_DEFAULT = 0.6
_ISLAND_PROBABILITY_MIN = 0.0
_ISLAND_PROBABILITY_MAX = 1.0

ISLAND_MIN_AREA_MM2_DEFAULT = 20.0
_ISLAND_MIN_AREA_MM2_MIN = 5.0
_ISLAND_MIN_AREA_MM2_MAX = 500.0

# Streamline stroke tuning at ribbon_elongation == 1.0: longer, thinner
# strokes bias enclosed pockets toward branching, vein-like ribbons instead
# of blobby cells. At 0.0 this is a no-op (matches streamline_labels'
# un-elongated defaults).
_RIBBON_MAX_STEPS_GAIN = 2.0
_RIBBON_WIDTH_SHRINK = 0.5

# Islands are carved by a single _carve_islands() pass over each region's
# post-mode cell_local ids -- there is no recursive re-invocation on the ids
# it just created, so an island can never itself contain a sub-island. This
# keeps region-count growth bounded and avoids runaway recursion.


def stage_seed(seed: int) -> int:
    """This stage's own deterministic sub-seed, independent of every other
    stage's RNG stream (SHA-256(seed || stage_name)[:8] as uint64) -- needed
    here more than most stages since one run touches several independent
    random streams (seed placement, warp, streamline tracing, island
    sub-seeding)."""
    digest = hashlib.sha256(f"{seed}|{STAGE_NAME}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _background_region_id(component_map: np.ndarray, n: int) -> int | None:
    """The single region id most likely to be "the page background": among
    regions that touch the page border at all, the one with the largest
    border contact length (ties broken by lowest id, deterministic). Returns
    ``None`` if no region touches the border (should not happen on a real
    page, but a synthetic/degenerate map is defensively handled)."""
    border_pixels = np.concatenate(
        [component_map[0, :], component_map[-1, :], component_map[:, 0], component_map[:, -1]]
    ).astype(np.int64)
    border_len = np.bincount(border_pixels, minlength=n)
    if not border_len.any():
        return None
    return int(np.argmax(border_len))


def _rim_core_split(mask2d: np.ndarray, rim_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Erode ``rim_px`` from ``mask2d``'s own boundary: the rim stays an
    un-partitioned strip tracing the region's real silhouette (protects
    subject boundaries -- a face/ear outline must never be eaten into by an
    organic cell), only the eroded core is organic-partitioned."""
    dist_to_bg = ndimage.distance_transform_edt(mask2d)
    core2d = dist_to_bg > rim_px
    rim2d = mask2d & ~core2d
    return core2d, rim2d


def _voronoi_organic_labels(
    core2d: np.ndarray,
    *,
    k: int,
    warp_px: float,
    noise_scale_px: float,
    seed: int,
) -> np.ndarray:
    """Local cell-id array over ``core2d``'s True pixels (flatnonzero order):
    domain-warped Voronoi assignment -- flowing-but-compact cells."""
    seeds = farthest_point_seeds(core2d, k) if k > 0 else []
    if not seeds:
        return np.zeros(int(core2d.sum()), dtype=np.int64)
    h, w = core2d.shape
    warp_dx, warp_dy = flow_field(h, w, strength_px=warp_px, scale_px=noise_scale_px, seed=seed)
    return voronoi_labels(core2d, seeds, warp_dx=warp_dx, warp_dy=warp_dy)


def _streamline_organic_labels(
    core2d: np.ndarray,
    *,
    target_cell_area: float,
    warp_px: float,
    noise_scale_px: float,
    ribbon_elongation: float,
    seed: int,
) -> np.ndarray:
    """Local cell-id array over ``core2d``'s True pixels: curl-noise
    streamline pockets -- the more organic/ribbon-prone of the two modes.
    ``ribbon_elongation`` biases strokes toward longer and thinner, which
    biases the enclosed pockets toward branching ribbon shapes."""
    stroke_canvas = streamline_labels(
        core2d,
        target_cell_area=target_cell_area,
        warp_px=warp_px,
        noise_scale_px=noise_scale_px,
        seed=seed,
        max_steps_gain=1.0 + _RIBBON_MAX_STEPS_GAIN * ribbon_elongation,
        width_shrink=1.0 - _RIBBON_WIDTH_SHRINK * ribbon_elongation,
    )
    pockets2d = core2d & ~stroke_canvas
    pocket_ids2d = label_components(to_component_input(pockets2d))
    if stroke_canvas.any() and pockets2d.any():
        _, (pr, pc) = ndimage.distance_transform_edt(~pockets2d, return_indices=True)
        cell_local2d = np.where(pockets2d, pocket_ids2d, pocket_ids2d[pr, pc])
    else:
        cell_local2d = pocket_ids2d
    return cell_local2d[core2d]


def _carve_one_island(
    core2d: np.ndarray,
    flat_idx: np.ndarray,
    *,
    area: int,
    seed_density_px: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Try to carve one enclosed island sub-cell out of a single host cell's
    pixel set (``flat_idx``, positions into ``core2d.ravel()``). Returns the
    island's flat pixel indices (a subset of ``flat_idx``), or ``None`` if no
    genuine interior enclosure resulted (e.g. the island seed cluster touched
    the cell's own boundary and "won" too much of the cell)."""
    cell_mask = np.zeros(core2d.size, dtype=bool)
    cell_mask[flat_idx] = True
    cell_mask2d = cell_mask.reshape(core2d.shape)
    # A small island seed cluster (1-3 seeds), grown against a denser outer
    # seed set covering the rest of the host cell -- the island is whichever
    # component ends up touching only the island seeds, which is a genuine
    # interior enclosure rather than a near-50/50 split.
    island_k = 1 + int(rng.integers(0, 3))
    island_target = min(area * 0.25, seed_density_px)
    k_outer = max(1, round(area / max(island_target, 1.0)))
    outer_seeds = farthest_point_seeds(cell_mask2d, max(k_outer, island_k + 1))
    if len(outer_seeds) <= island_k:
        return None
    local = voronoi_labels(cell_mask2d, outer_seeds)
    is_island = np.isin(local, list(range(island_k)))
    if not is_island.any() or is_island.all():
        return None
    return flat_idx[is_island]


def _carve_islands(
    core2d: np.ndarray,
    cell_local: np.ndarray,
    *,
    island_probability: float,
    island_min_area_px: float,
    seed_density_px: float,
    seed: int,
) -> np.ndarray:
    """For a deterministic subset of cells above ``island_min_area_px``,
    carve a small secondary seed cluster within that cell's own mask,
    producing a genuinely enclosed sub-cell as its own new cell id. Capped at
    one nesting level (an island cannot itself contain a sub-island) --
    islands are only ever carved out of the input ``cell_local`` ids, never
    out of a just-created island."""
    if island_probability <= 0.0:
        return cell_local
    core_pixels = np.flatnonzero(core2d.ravel())
    n_cells = int(cell_local.max()) + 1 if cell_local.size else 0
    rng = np.random.default_rng(seed)
    next_id = n_cells
    updated = cell_local.copy()
    for cell in range(n_cells):
        sel = cell_local == cell
        area = int(sel.sum())
        if area < island_min_area_px or rng.random() >= island_probability:
            continue
        flat_idx = core_pixels[sel]
        island_flat = _carve_one_island(
            core2d, flat_idx, area=area, seed_density_px=seed_density_px, rng=rng
        )
        if island_flat is None:
            continue
        island_pixel_mask = np.zeros(core2d.size, dtype=bool)
        island_pixel_mask[island_flat] = True
        island_pixel_mask2d = island_pixel_mask.reshape(core2d.shape)
        island_components = label_components(to_component_input(island_pixel_mask2d))
        island_ids_present = sorted(set(island_components[island_pixel_mask2d].tolist()))
        for local_island_id in island_ids_present:
            comp_mask = island_pixel_mask2d & (island_components == local_island_id)
            comp_pixels = np.flatnonzero(comp_mask.ravel())
            core_positions = np.searchsorted(core_pixels, comp_pixels)
            updated[core_positions] = next_id
            next_id += 1
    return updated


def organic_partition_regions(
    graph: RegionGraph,
    palette: Palette,
    *,
    mode: OrganicMode = "streamline",
    min_area_px: float,
    seed_density_px: float,
    rim_px: float,
    warp_px: float,
    noise_scale_px: float,
    ribbon_elongation: float = 0.0,
    island_probability: float = 0.0,
    island_min_area_px: float = 0.0,
    fold_a_min_px: float,
    fold_min_inradius_px: float = 0.0,
    dark_fold_max_inradius_px: float = 0.0,
    skip_background: bool = SKIP_BACKGROUND_DEFAULT,
    skip_dark_lab_l_threshold: float = SKIP_DARK_LAB_L_THRESHOLD,
    warp_seed: int = 0,
    config_hash: str = _UNSET_HASH,
) -> tuple[RegionGraph, frozenset[int], frozenset[int]]:
    """Full organic-partition pass (ADR-003). Regions with
    ``area_px >= min_area_px`` get their core (interior, beyond ``rim_px`` of
    their own boundary) subdivided into organic cells per ``mode``; smaller
    regions and each region's own rim pass through untouched. If
    ``skip_background`` (default), the single region most likely to be the
    page background (touches the page border, largest border contact among
    those that do -- see ``_background_region_id``) is also left untouched
    regardless of size, so a large flat backdrop is not shattered into
    hundreds of small organic cells. Any region whose palette color's LAB L*
    is at or below ``skip_dark_lab_l_threshold`` is folded into a neighboring
    region *before* partitioning (not merely left untouched -- see
    ``fold_regions_where``) -- targets a source image's own pre-drawn
    near-black outline stroke, which has real width and so quantizes into
    its own thin ring-shaped region; left standalone, organic-partitioning
    its neighbors still traces both of the ring's edges as real silhouette
    boundaries, reading as a doubled outline (see
    ``SKIP_DARK_LAB_L_THRESHOLD``). Every product cell inherits its parent
    region's palette label (the folded-away dark region's own label is
    discarded, replaced by whichever neighbor absorbed it). Returns
    the rebuilt, fully re-derived ``RegionGraph``, the frozenset of *filler*
    region ids (core + rim of every organic-partitioned region -- exempt
    from the printability readable-size floor, same contract as
    ``split_large_regions``), and the frozenset of *render-filler* ids (core
    only, for bold vs. fine stroke weight)."""
    labels_of_region = [r.label for r in graph.regions]
    component_map = graph.component_map

    # A source image's own pre-drawn near-black outline stroke has real
    # width, so it quantizes into its own thin, ring-shaped region (encloses
    # the subject, touching both the background and the subject's fill on
    # its two opposite edges). Left as a standalone region, organic-
    # partitioning its *neighbors* still traces both of that ring's edges as
    # real silhouette boundaries -- two near-parallel lines that read as a
    # doubled outline (confirmed via rendered preview on a real cartoon
    # photo). Folding it into an adjacent region *before* partitioning
    # removes the ring as a distinct shape entirely, so only one silhouette
    # edge remains.
    dark_label_ids = {c.index for c in palette.colors if c.lab[0] <= skip_dark_lab_l_threshold}
    if dark_label_ids:
        dark_list = list(dark_label_ids)

        def _dark_and_thin(
            _areas: np.ndarray, cur_labels: list[int], cmap: np.ndarray
        ) -> np.ndarray:
            dark = np.isin(np.asarray(cur_labels, dtype=np.int64), dark_list)
            if dark_fold_max_inradius_px <= 0.0:
                return dark
            # Only a THIN dark region is an outline stroke; a large solid
            # dark mass (black hair, a navy dress) is real subject art whose
            # color must survive -- folding it recolors the whole feature
            # into whichever neighbour absorbs it (observed: near-black hair
            # coming out brown).
            return dark & (region_inradius_px(cmap) < dark_fold_max_inradius_px)

        component_map, labels_of_region = fold_regions_where(
            component_map, labels_of_region, should_fold=_dark_and_thin
        )

    h, w = component_map.shape
    flat = component_map.ravel()
    n = len(labels_of_region)
    areas = np.bincount(flat, minlength=n)
    background_rid = _background_region_id(component_map, n) if skip_background else None

    new_flat = np.empty_like(flat)
    new_labels: list[int] = []
    filler_pixel = np.zeros(flat.size, dtype=bool)
    rim_pixel = np.zeros(flat.size, dtype=bool)
    next_id = 0

    for rid in range(n):
        pixels = np.flatnonzero(flat == rid)
        area = int(areas[rid])
        if area < min_area_px or rid == background_rid:
            new_flat[pixels] = next_id
            new_labels.append(labels_of_region[rid])
            next_id += 1
            continue

        mask = np.zeros(h * w, dtype=bool)
        mask[pixels] = True
        mask2d = mask.reshape(h, w)
        core2d, rim2d = _rim_core_split(mask2d, rim_px)
        core_area = int(core2d.sum())

        if core_area < min_area_px:
            new_flat[pixels] = next_id
            new_labels.append(labels_of_region[rid])
            next_id += 1
            continue

        region_seed = warp_seed + rid * 1009  # large odd stride keeps per-region streams disjoint
        k = max(1, round(core_area / max(seed_density_px, 1.0)))
        target_cell_area = max(seed_density_px, 1.0)

        if mode == "voronoi":
            cell_local = _voronoi_organic_labels(
                core2d, k=k, warp_px=warp_px, noise_scale_px=noise_scale_px, seed=region_seed
            )
        elif mode == "streamline":
            cell_local = _streamline_organic_labels(
                core2d,
                target_cell_area=target_cell_area,
                warp_px=warp_px,
                noise_scale_px=noise_scale_px,
                ribbon_elongation=ribbon_elongation,
                seed=region_seed,
            )
        else:  # "mixed": streamline first, then re-split any oversized pocket
            cell_local = _streamline_organic_labels(
                core2d,
                target_cell_area=target_cell_area,
                warp_px=warp_px,
                noise_scale_px=noise_scale_px,
                ribbon_elongation=ribbon_elongation,
                seed=region_seed,
            )
            sizes = np.bincount(cell_local) if cell_local.size else np.array([])
            if sizes.size and sizes.max() > target_cell_area * 2:
                # Escalate: Voronoi-split the whole core once more on top of
                # the streamline result's seed density, biasing toward
                # compact cells for whatever the streamline pass left large.
                extra_k = max(1, round(core_area / target_cell_area))
                voronoi_overlay = _voronoi_organic_labels(
                    core2d,
                    k=extra_k,
                    warp_px=warp_px,
                    noise_scale_px=noise_scale_px,
                    seed=region_seed + 1,
                )
                # Combine: pair (streamline_cell, voronoi_cell) -> new dense id.
                pairs = list(zip(cell_local.tolist(), voronoi_overlay.tolist(), strict=True))
                uniq = {p: i for i, p in enumerate(dict.fromkeys(pairs))}
                cell_local = np.array([uniq[p] for p in pairs], dtype=np.int64)

        if island_probability > 0.0:
            cell_local = _carve_islands(
                core2d,
                cell_local,
                island_probability=island_probability,
                island_min_area_px=island_min_area_px,
                seed_density_px=seed_density_px,
                seed=region_seed + 2,
            )

        core_pixels = np.flatnonzero(core2d.ravel())
        used = sorted(set(cell_local.tolist()))
        remap = {c: i for i, c in enumerate(used)}
        for local in used:
            sel = core_pixels[cell_local == local]
            new_flat[sel] = next_id + remap[local]
            new_labels.append(labels_of_region[rid])
        filler_pixel[core_pixels] = True
        next_id += len(used)

        # The rim stays one un-split cell tracing the region's real
        # silhouette -- shares filler cells' printability-floor exemption
        # (see OrganicPartitionStage/split_large's shared contract), but is
        # tracked separately (rim_pixel) so render-time bold/fine stroke
        # weight can still treat it as a real boundary, not a filler seam.
        rim_pixels = np.flatnonzero(rim2d.ravel())
        if rim_pixels.size:
            new_flat[rim_pixels] = next_id
            new_labels.append(labels_of_region[rid])
            filler_pixel[rim_pixels] = True
            rim_pixel[rim_pixels] = True
            next_id += 1

    provisional = new_flat.reshape(h, w).astype(np.int32)
    per_pixel_label = np.array(new_labels, dtype=np.int64)[provisional]
    final_map, final_labels = connected_labels(provisional, per_pixel_label)
    final_map, final_labels = fold_subfloor_regions(final_map, final_labels, a_min=fold_a_min_px)
    if fold_min_inradius_px > 0.0:
        # Width floor, FILLER CELLS ONLY: a cell can clear the area floor yet
        # still be a ribbon too narrow to carry a printed number (commercial
        # sheets have no such slivers); fold those into a neighbour. Real
        # subject regions are exempt -- a character's eye or a thin garment
        # band is legitimately narrow, and folding it erases the feature
        # (observed: eyes/eyebrows disappearing from character art). The
        # filler test uses the per-PIXEL mask (stable across fold passes,
        # unlike region ids, which renumber every pass).
        filler_weights = filler_pixel.astype(np.float64)
        # The floor is clamped per cell to its PARENT region's own inradius:
        # a cell carved from a thick outline band (a ribbon a few mm wide,
        # e.g. a cartoon's drawn stroke too wide for the dark fold) can never
        # be wider than the band itself. Folding those cells for missing the
        # global floor re-merges the whole band into one giant region with a
        # single number at its widest pocket -- the rest of the stroke then
        # prints unnumbered. With the clamp, such a band still segments into
        # numberable chunks, while cells inside wide parents keep the full
        # floor.
        parent_flat = component_map.ravel()
        parent_inradius = region_inradius_px(component_map)

        def _narrow_filler(
            _areas: np.ndarray, _labels: list[int], cmap: np.ndarray
        ) -> np.ndarray:
            flat_cur = cmap.ravel()
            n_cur = int(flat_cur.max()) + 1
            total_cur = np.bincount(flat_cur, minlength=n_cur)
            filler_cur = np.bincount(flat_cur, weights=filler_weights, minlength=n_cur)
            is_filler = (filler_cur > 0) & (filler_cur * 2 >= total_cur)
            uniq_ids, first_idx = np.unique(flat_cur, return_index=True)
            cell_parent = np.zeros(n_cur, dtype=np.int64)
            cell_parent[uniq_ids] = parent_flat[first_idx]
            floor_per_cell = np.minimum(
                fold_min_inradius_px, 0.9 * parent_inradius[cell_parent]
            )
            return is_filler & (region_inradius_px(cmap) < floor_per_cell)

        final_map, final_labels = fold_regions_where(
            final_map, final_labels, should_fold=_narrow_filler
        )

    # A final region is "filler" if a majority of its pixels came from an
    # organic-partitioned cell (robust to the fold occasionally absorbing a
    # sliver of a non-partitioned region into a filler cell or vice versa);
    # same rule for "rim" (mirrors split_large_regions' identical logic).
    ff = final_map.ravel()
    n_final = int(ff.max()) + 1
    total = np.bincount(ff, minlength=n_final)
    filler_count = np.bincount(ff, weights=filler_pixel.astype(np.float64), minlength=n_final)
    filler_ids = {
        i for i in range(n_final) if filler_count[i] > 0 and filler_count[i] * 2 >= total[i]
    }
    rim_count = np.bincount(ff, weights=rim_pixel.astype(np.float64), minlength=n_final)
    rim_ids = {i for i in range(n_final) if rim_count[i] > 0 and rim_count[i] * 2 >= total[i]}
    render_filler_ids = filler_ids - rim_ids

    new_graph = rebuild_region_graph(
        final_map,
        final_labels,
        palette,
        stage_name=STAGE_NAME,
        stage_version=STAGE_VERSION,
        config_hash=config_hash,
        source_hash=graph.provenance.source_hash,
    )
    return new_graph, frozenset(filler_ids), frozenset(render_filler_ids)


class OrganicPartitionStage:
    """Stage wrapper: (``region_graph``, ``palette``) → ``region_graph`` (organic)."""

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
            raise ConfigError(f"organic config: enabled must be a bool, got {enabled!r}")

        mode = section.get("mode", "streamline")
        if mode not in _MODES:
            raise ConfigError(f"organic config: mode must be one of {_MODES}, got {mode!r}")

        skip_background = section.get("skip_background", SKIP_BACKGROUND_DEFAULT)
        if not isinstance(skip_background, bool):
            raise ConfigError(
                f"organic config: skip_background must be a bool, got {skip_background!r}"
            )

        skip_dark_l = section.get("skip_dark_lab_l_threshold", SKIP_DARK_LAB_L_THRESHOLD)
        if not isinstance(skip_dark_l, (int, float)) or not (0.0 <= float(skip_dark_l) <= 100.0):
            raise ConfigError(
                "organic config: skip_dark_lab_l_threshold must be in [0.0, 100.0], "
                f"got {skip_dark_l!r}"
            )

        def _bounded(key: str, default: float, lo: float, hi: float) -> float:
            value = section.get(key, default)
            if not isinstance(value, (int, float)) or not (lo <= float(value) <= hi):
                raise ConfigError(f"organic config: {key} must be in [{lo}, {hi}], got {value!r}")
            return float(value)

        self._enabled = enabled
        self._mode: OrganicMode = mode
        self._skip_background = skip_background
        self._skip_dark_lab_l_threshold = float(skip_dark_l)
        self._min_area_mm2 = _bounded(
            "min_area_mm2", MIN_AREA_MM2_DEFAULT, _MIN_AREA_MM2_MIN, _MIN_AREA_MM2_MAX
        )
        self._seed_density_mm2 = _bounded(
            "seed_density_mm2",
            SEED_DENSITY_MM2_DEFAULT,
            _SEED_DENSITY_MM2_MIN,
            _SEED_DENSITY_MM2_MAX,
        )
        self._rim_mm = _bounded("rim_mm", RIM_MM_DEFAULT, _RIM_MM_MIN, _RIM_MM_MAX)
        self._warp_mm = _bounded(
            "warp_strength_mm",
            WARP_STRENGTH_MM_DEFAULT,
            _WARP_STRENGTH_MM_MIN,
            _WARP_STRENGTH_MM_MAX,
        )
        self._noise_scale_mm = _bounded(
            "noise_scale_mm", NOISE_SCALE_MM_DEFAULT, _NOISE_SCALE_MM_MIN, _NOISE_SCALE_MM_MAX
        )
        self._ribbon_elongation = _bounded(
            "ribbon_elongation",
            RIBBON_ELONGATION_DEFAULT,
            _RIBBON_ELONGATION_MIN,
            _RIBBON_ELONGATION_MAX,
        )
        self._island_probability = _bounded(
            "island_probability",
            ISLAND_PROBABILITY_DEFAULT,
            _ISLAND_PROBABILITY_MIN,
            _ISLAND_PROBABILITY_MAX,
        )
        self._island_min_area_mm2 = _bounded(
            "island_min_area_mm2",
            ISLAND_MIN_AREA_MM2_DEFAULT,
            _ISLAND_MIN_AREA_MM2_MIN,
            _ISLAND_MIN_AREA_MM2_MAX,
        )
        self._min_inner_diameter_mm = _bounded("min_inner_diameter_mm", 0.0, 0.0, 20.0)
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
        return ("region_graph", "palette", "raster_working")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("region_graph", "filler_region_ids", "render_filler_region_ids")

    @property
    def config_section(self) -> str:
        return "organic"

    def run(self, ctx: PipelineContext) -> None:
        graph = ctx.get("region_graph")
        palette = ctx.get("palette")
        raster = ctx.get("raster_working")
        if not isinstance(graph, RegionGraph) or not isinstance(palette, Palette):
            raise ConfigError("organic_partition requires RegionGraph + Palette artifacts")
        if not self._enabled:
            ctx.put("region_graph", graph)
            ctx.put("filler_region_ids", frozenset())
            ctx.put("render_filler_region_ids", frozenset())
            return

        work_scale = getattr(raster, "work_scale", 0.0)
        ppmm = 1.0 / (work_scale * MM_PER_INCH / PT_PER_INCH) if work_scale > 0 else 1.0

        from mysterycbn.stages.graph.merge import area_floor_px

        fold_a_min_px = area_floor_px(self._d_min_mm, work_scale)
        new_graph, filler_ids, render_filler_ids = organic_partition_regions(
            graph,
            palette,
            mode=self._mode,
            min_area_px=self._min_area_mm2 * ppmm * ppmm,
            seed_density_px=self._seed_density_mm2 * ppmm * ppmm,
            rim_px=self._rim_mm * ppmm,
            warp_px=self._warp_mm * ppmm,
            noise_scale_px=self._noise_scale_mm * ppmm,
            ribbon_elongation=self._ribbon_elongation,
            island_probability=self._island_probability,
            island_min_area_px=self._island_min_area_mm2 * ppmm * ppmm,
            fold_a_min_px=fold_a_min_px,
            fold_min_inradius_px=(self._min_inner_diameter_mm / 2.0) * ppmm,
            dark_fold_max_inradius_px=DARK_FOLD_MAX_INRADIUS_MM * ppmm,
            skip_background=self._skip_background,
            skip_dark_lab_l_threshold=self._skip_dark_lab_l_threshold,
            warp_seed=stage_seed(ctx.seed),
            config_hash=self._config_hash,
        )
        ctx.put("region_graph", new_graph)
        ctx.put("filler_region_ids", filler_ids)
        ctx.put("render_filler_region_ids", render_filler_ids)
