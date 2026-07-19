"""Shared organic-texture primitives for the graph-domain filler stages.

Extracted from ``split_large.py`` (ENGINE_SPEC §12) so both it and
``organic_partition.py`` share one implementation of value noise, domain
warping, streamline tracing, Voronoi seeding/assignment, and the
connect/fold/rebuild machinery that turns a mutated component map back into
a valid ``RegionGraph``. No behavior change versus the original
``split_large.py`` implementations -- this is a pure extraction.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import ndimage

from mysterycbn.model.records import Palette, Provenance, Region, RegionGraph
from mysterycbn.stages.graph.components import label_components

# Farthest-point sampling is O(k*N); above this many seeds for one region we
# switch to a deterministic uniform-grid seeding (still O(N) Voronoi), which
# keeps split-heavy "dense" runs from becoming pathologically slow on a large
# flat background broken into thousands of cells.
FPS_SEED_CAP = 256


def value_noise2d(h: int, w: int, *, cell_px: float, seed: int) -> np.ndarray:
    """Smooth, deterministic 2-D value noise in [-1, 1] at shape ``(h, w)``.

    Random values are drawn on a coarse lattice (spacing ``cell_px``,
    ``>= 2px`` so there is always something to interpolate) and bilinearly
    upsampled -- this is the standard "value noise" construction (simpler
    than gradient/Perlin noise, but visually equivalent for a flow field:
    both are just a smooth scalar field). No external noise library is a
    project dependency, so this is a small, self-contained NumPy
    implementation rather than pulling one in for a single texture.
    """
    cell_px = max(2.0, cell_px)
    lattice_h = max(2, math.ceil(h / cell_px) + 2)
    lattice_w = max(2, math.ceil(w / cell_px) + 2)
    rng = np.random.default_rng(seed)
    lattice = rng.uniform(-1.0, 1.0, size=(lattice_h, lattice_w))
    # Sample the lattice at each output pixel's fractional lattice coordinate
    # via map_coordinates (order=1 -> bilinear), which is exactly "upsample
    # with smooth interpolation" without hand-rolling the 4-tap blend.
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
    lattice_r = rows / cell_px
    lattice_c = cols / cell_px
    return np.asarray(
        ndimage.map_coordinates(lattice, [lattice_r, lattice_c], order=1, mode="nearest"),
        dtype=np.float64,
    )


def flow_field(
    h: int, w: int, *, strength_px: float, scale_px: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A smooth 2-D displacement field ``(dx, dy)``, each in
    ``[-strength_px, strength_px]``, for domain-warping the Voronoi split
    (see ``voronoi_labels``). Two independent noise fields (different seeds)
    drive the x/y components so the flow is not simply radial/separable."""
    if strength_px <= 0.0:
        zero = np.zeros((h, w), dtype=np.float64)
        return zero, zero
    nx = value_noise2d(h, w, cell_px=scale_px, seed=seed)
    ny = value_noise2d(h, w, cell_px=scale_px, seed=seed + 1)
    return nx * strength_px, ny * strength_px


def flow_angle_field(h: int, w: int, *, scale_px: float, seed: int) -> np.ndarray:
    """A smooth angle field in radians, for tracing streamlines (see
    ``streamline_labels``): value noise directly gives a smoothly-varying
    scalar, scaled to a full turn so streamlines curl and cross freely
    instead of following a single dominant direction."""
    return np.asarray(
        value_noise2d(h, w, cell_px=scale_px, seed=seed) * np.pi * 2.0, dtype=np.float64
    )


def trace_streamlines(
    mask: np.ndarray,
    *,
    n_lines: int,
    step_px: float,
    max_steps: int,
    line_width_px: int,
    scale_px: float,
    seed: int,
) -> np.ndarray:
    """Rasterize ``n_lines`` short curved strokes onto a boolean canvas the
    shape of ``mask``: each stroke starts at a deterministic pseudo-random
    in-mask pixel and walks ``max_steps`` fixed-length steps, at each step
    turning to follow a smooth angle field (see ``flow_angle_field``) --
    the standard "curl noise" streamline construction used for organic,
    hand-drawn-looking line textures (wood grain, marbling). Distinct from
    ``voronoi_labels``: strokes are drawn independently and can cross or
    run parallel to each other freely; nothing here partitions the plane by
    construction the way a distance-transform Voronoi does. Deterministic:
    the RNG is seeded, start points are drawn without replacement in a fixed
    order, and every arithmetic step is a pure function of position + the
    (deterministic) angle field."""
    h, w = mask.shape
    if n_lines <= 0 or not mask.any():
        return np.zeros((h, w), dtype=bool)
    angle = flow_angle_field(h, w, scale_px=scale_px, seed=seed)
    rng = np.random.default_rng(seed)
    canvas = np.zeros((h, w), dtype=bool)

    inside_idx = np.flatnonzero(mask.ravel())
    n_start = min(n_lines, inside_idx.size)
    starts = rng.choice(inside_idx, size=n_start, replace=False)

    for flat_idx in starts:
        y0, x0 = divmod(int(flat_idx), w)
        py, px = float(y0), float(x0)
        for _ in range(max_steps):
            iy, ix = round(py), round(px)
            if not (0 <= iy < h and 0 <= ix < w) or not mask[iy, ix]:
                break
            canvas[iy, ix] = True
            a = float(angle[iy, ix])
            py += step_px * math.sin(a)
            px += step_px * math.cos(a)

    if line_width_px > 1:
        canvas = ndimage.binary_dilation(canvas, iterations=line_width_px // 2)
    return canvas & mask


def to_component_input(free_mask: np.ndarray) -> np.ndarray:
    """A per-pixel int array ``label_components`` can partition: in-mask free
    pixels get one shared placeholder id, everything else (stroke pixels and
    outside-mask background) gets a different id, so ``label_components``'s
    4-connected labelling only ever splits the free pixels into their
    enclosed pockets."""
    return np.where(free_mask, 1, 0).astype(np.int32)


def streamline_labels(
    mask: np.ndarray,
    *,
    target_cell_area: float,
    warp_px: float,
    noise_scale_px: float,
    seed: int,
    max_steps_gain: float = 1.0,
    width_shrink: float = 1.0,
) -> np.ndarray:
    """Split ``mask`` into cells bounded by organic, freely-crossing curved
    strokes rather than a Voronoi diagram's straight/warped-straight edges
    (see ``trace_streamlines``): every pixel not covered by a stroke keeps
    its original provisional-cell id (0, i.e. still just ``mask``); strokes
    themselves become the boundary. The stroke density is targeted at
    roughly ``target_cell_area`` per enclosed pocket via a small number of
    fixed passes with increasing stroke counts, since (unlike Voronoi) there
    is no closed-form seed-count -> cell-count relationship for streamline
    strokes. Returns the stroke canvas (boolean); the caller re-labels
    connected components of ``mask & ~stroke_canvas`` directly."""
    h, w = mask.shape
    area = int(mask.sum())
    if area <= 0:
        return np.zeros((h, w), dtype=bool)
    target_n_pockets = max(1, round(area / target_cell_area))
    # A closed pocket needs roughly its own perimeter's worth of stroke
    # coverage; scale line count with target pocket count directly (each
    # stroke contributes to several pocket walls as it winds through).
    canvas = np.zeros((h, w), dtype=bool)
    n_lines = max(8, target_n_pockets * 4)
    max_steps = max(20, round(math.sqrt(target_cell_area) / max(0.5, 1.0) * max_steps_gain))
    line_width_px = max(1, round((2 + warp_px / 6.0) * width_shrink))
    for attempt in range(6):
        canvas = canvas | trace_streamlines(
            mask,
            n_lines=n_lines,
            step_px=1.3,
            max_steps=max_steps,
            line_width_px=line_width_px,
            scale_px=max(4.0, noise_scale_px),
            seed=seed + attempt * 97,
        )
        free = mask & ~canvas
        labeled = label_components(to_component_input(free))
        sizes = np.bincount(labeled[free].ravel()) if free.any() else np.array([])
        oversized = int((sizes > target_cell_area * 2).sum())
        if oversized == 0:
            break
        # Still-oversized pockets: trace more strokes next pass, weighted
        # toward the same target density (a fixed, bounded number of passes
        # keeps this O(passes * N) rather than an open-ended loop).
        n_lines = max(8, oversized * 4)
    return canvas


def farthest_point_seeds(mask: np.ndarray, k: int) -> list[int]:
    """``k`` seed flat-indices inside ``mask`` by farthest-point sampling on
    the mask's Euclidean distance transform. Deterministic: the first seed
    is the distance-transform argmax (ties -> lowest flat index), each next
    seed maximises distance-to-nearest-existing-seed (same tie rule)."""
    w = mask.shape[1]
    inside = np.flatnonzero(mask.ravel())
    if k >= inside.size:
        return inside.tolist()
    if k > FPS_SEED_CAP:
        return grid_seeds(mask, k)

    dt = ndimage.distance_transform_edt(mask).ravel()
    first = int(inside[np.argmax(dt[inside])])
    seeds = [first]

    ys, xs = np.divmod(inside, w)
    seed_r, seed_c = divmod(first, w)
    nearest = np.hypot(ys - seed_r, xs - seed_c)
    while len(seeds) < k:
        idx_local = int(np.argmax(nearest))
        seeds.append(int(inside[idx_local]))
        sr, sc = divmod(int(inside[idx_local]), w)
        nearest = np.minimum(nearest, np.hypot(ys - sr, xs - sc))
        nearest[idx_local] = -1.0  # never re-pick a seed
    return seeds


def grid_seeds(mask: np.ndarray, k: int) -> list[int]:
    """Deterministic O(N) seeding for large ``k``: lay a uniform grid over the
    mask's bounding box, snap each grid point to the nearest in-mask pixel,
    and dedupe. Returns roughly ``k`` seed flat-indices (never more)."""
    w = mask.shape[1]
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    # Grid dimensions proportional to the bbox aspect ratio, product ~ k.
    bh, bw = (r1 - r0 + 1), (c1 - c0 + 1)
    gr = max(1, round(math.sqrt(k * bh / bw)))
    gc = max(1, math.ceil(k / gr))
    inside_flat = np.flatnonzero(mask.ravel())
    _, (ir, ic) = ndimage.distance_transform_edt(~mask, return_indices=True)
    seeds: list[int] = []
    seen: set[int] = set()
    for gy in range(gr):
        ry = r0 + int((gy + 0.5) * bh / gr)
        for gx in range(gc):
            cx = c0 + int((gx + 0.5) * bw / gc)
            # snap (ry, cx) to the nearest in-mask pixel
            sr, sc = int(ir[ry, cx]), int(ic[ry, cx])
            flat = sr * w + sc
            if flat not in seen:
                seen.add(flat)
                seeds.append(flat)
    return seeds if seeds else inside_flat[:1].tolist()


def voronoi_labels(
    mask: np.ndarray,
    seeds: list[int],
    *,
    warp_dx: np.ndarray | None = None,
    warp_dy: np.ndarray | None = None,
) -> np.ndarray:
    """Assign every ``mask`` pixel to its nearest seed (compact Voronoi
    cells). Returns an int array over the mask's pixels (in ``np.flatnonzero``
    order) giving the local cell index [0, len(seeds)).

    Uses one Euclidean distance-transform pass with ``return_indices`` (each
    background pixel resolved to its nearest feature pixel) rather than a
    dense ``pixels x seeds`` distance matrix, so cost is O(N) regardless of
    the seed count -- essential when a large flat region gets thousands of
    seeds and materialising N*k floats would exhaust memory.

    ``warp_dx``/``warp_dy`` (see ``flow_field``), when given, implement
    domain warping (a standard procedural-texture technique): every pixel's
    *query point* into the nearest-seed field is displaced by the flow field
    before lookup, so straight Voronoi cell edges become the flowing, organic
    curves the offset produces."""
    h, w = mask.shape
    inside = np.flatnonzero(mask.ravel())
    # A feature map where each seed pixel carries its 1-based cell id, all
    # else 0; distance_transform_edt(return_indices) then hands every pixel
    # the coordinates of its nearest seed, from which we read the cell id.
    feat = np.zeros(h * w, dtype=np.int64)
    for cell, s in enumerate(seeds):
        feat[s] = cell + 1
    feat2d = feat.reshape(h, w)
    _, (ir, ic) = ndimage.distance_transform_edt(feat2d == 0, return_indices=True)
    nearest_cell2d = feat2d[ir, ic] - 1  # (H, W): cell id of the nearest seed

    if warp_dx is not None and warp_dy is not None:
        rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
        query_r = np.clip(rows + warp_dy, 0, h - 1)
        query_c = np.clip(cols + warp_dx, 0, w - 1)
        nearest_cell2d = ndimage.map_coordinates(
            nearest_cell2d.astype(np.float64),
            [query_r, query_c],
            order=0,  # nearest-neighbour: cell ids are not interpolable
            mode="nearest",
        ).astype(np.int64)

    return np.asarray(nearest_cell2d.ravel()[inside], dtype=np.int64)


def connected_labels(
    provisional: np.ndarray, per_pixel_label: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Relabel ``provisional`` into dense 4-connected component ids and return
    (map, per-region palette label)."""
    final_map = label_components(provisional)
    ff = final_map.ravel()
    n_final = int(ff.max()) + 1
    first_pixel = np.full(n_final, ff.size, dtype=np.int64)
    np.minimum.at(first_pixel, ff, np.arange(ff.size))
    labels = per_pixel_label.ravel()[first_pixel].tolist()
    return final_map, labels


# fold_subfloor_regions can need more than one pass: a chain of tiny
# same-label regions with no larger neighbour folds into *each other* on a
# single pass, and the merged result can itself still be below a_min (no
# region in the chain was ever compared against the *combined* area). This
# caps the fixpoint loop so a pathological all-tiny component map can't spin
# forever -- in practice real inputs converge in 1-3 passes.
_FOLD_MAX_PASSES = 8


def fold_subfloor_regions(
    component_map: np.ndarray, labels: list[int], *, a_min: float
) -> tuple[np.ndarray, list[int]]:
    """Absorb every region below ``a_min`` into a 4-adjacent region -- see
    ``fold_regions_where`` for the shared fold mechanics this specializes."""
    return fold_regions_where(
        component_map, labels, should_fold=lambda areas, _labels, _cmap: areas < a_min
    )


def region_inradius_px(component_map: np.ndarray) -> np.ndarray:
    """Per-region inradius in px: the max distance from any of the region's
    pixels to the nearest pixel of a *different* region (or the page edge).
    This is the radius of the largest disk that fits inside the region — the
    printable clearance a label actually needs, which per-region *area*
    cannot capture (a long 1px ribbon has plenty of area and zero room)."""
    from scipy.ndimage import distance_transform_edt

    padded = np.pad(component_map, 1, constant_values=-1)
    boundary = np.zeros(padded.shape, dtype=bool)
    boundary[1:, :] |= padded[1:, :] != padded[:-1, :]
    boundary[:-1, :] |= padded[:-1, :] != padded[1:, :]
    boundary[:, 1:] |= padded[:, 1:] != padded[:, :-1]
    boundary[:, :-1] |= padded[:, :-1] != padded[:, 1:]
    dist = np.asarray(distance_transform_edt(~boundary))[1:-1, 1:-1]
    out = np.zeros(int(component_map.max()) + 1)
    np.maximum.at(out, component_map.ravel(), dist.ravel())
    return out


def fold_narrow_regions(
    component_map: np.ndarray, labels: list[int], *, min_inradius_px: float
) -> tuple[np.ndarray, list[int]]:
    """Absorb every region too *narrow* to carry a printed label (inradius
    below ``min_inradius_px``) into a 4-adjacent region, preferring a
    same-label neighbour -- the width-based complement of
    ``fold_subfloor_regions``'s area floor."""
    return fold_regions_where(
        component_map,
        labels,
        should_fold=lambda _areas, _labels, cmap: region_inradius_px(cmap) < min_inradius_px,
    )


def fold_regions_where(
    component_map: np.ndarray,
    labels: list[int],
    *,
    should_fold: Callable[[np.ndarray, list[int], np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Absorb every region for which ``should_fold(areas, labels, cmap)`` is
    True into a 4-adjacent region, preferring a neighbour of the same palette
    label (so paint-by-number colour stays correct), falling back to any
    neighbour. ``should_fold`` receives the *current pass's* per-region-id
    area array (``np.bincount`` over that pass's component map), the
    *current pass's* per-id palette-label list, and that pass's component
    map itself -- all re-derived fresh each
    pass, since ids are renumbered by every fold (a predicate that closes
    over the *original* labels/ids instead of using the ones passed in will
    silently fold the wrong regions on the second pass onward). Deterministic:
    regions processed by ascending id, ties in neighbour choice broken by
    lowest neighbour id. Repeats until no region matches ``should_fold`` or a
    pass makes no further progress (a single pass can leave a freshly-merged
    region still matching -- see ``_FOLD_MAX_PASSES``). Re-runs component
    labelling after every pass so ids stay dense and connected throughout."""
    cmap, region_labels = component_map, labels
    for _ in range(_FOLD_MAX_PASSES):
        new_cmap, new_labels = _fold_regions_where_once(
            cmap, region_labels, should_fold=should_fold
        )
        if new_cmap.shape == cmap.shape and np.array_equal(new_cmap, cmap):
            break
        cmap, region_labels = new_cmap, new_labels
    return cmap, region_labels


def _fold_regions_where_once(
    component_map: np.ndarray,
    labels: list[int],
    *,
    should_fold: Callable[[np.ndarray, list[int], np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """One fold pass -- may leave a freshly-merged region still matching
    ``should_fold``; see ``fold_regions_where``, which loops this to a
    fixpoint."""
    n = len(labels)
    areas = np.bincount(component_map.ravel(), minlength=n)
    mask = np.asarray(should_fold(areas, labels, component_map), dtype=bool)
    to_fold = [i for i in range(n) if bool(mask[i])]
    if not to_fold:
        return component_map, labels

    cmap = component_map.copy()
    label_arr = np.array(labels, dtype=np.int64)

    # Region adjacency (4-neighbour differing-id pixel pairs), computed once.
    from mysterycbn.stages.graph.components import _adjacency

    boundary, _ = _adjacency(cmap)
    adj: dict[int, set[int]] = {}
    for a, b in boundary:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # Union-find so a chain of folding regions folds transitively to one target.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rid in to_fold:
        neighbours = sorted(adj.get(rid, set()))
        if not neighbours:
            continue  # isolated (should not happen on a connected map)
        same = [n_ for n_ in neighbours if label_arr[n_] == label_arr[rid]]
        target = (same or neighbours)[0]
        parent[find(rid)] = find(target)

    if all(find(i) == i for i in range(n)):
        return component_map, labels  # nothing actually folded

    remap = np.array([find(i) for i in range(n)], dtype=np.int64)
    cmap = remap[cmap]
    per_pixel_label = label_arr[cmap]
    # Re-densify: absorbed ids leave gaps and may connect same-id blobs.
    return connected_labels(cmap.astype(np.int32), per_pixel_label)


def rebuild_region_graph(
    component_map: np.ndarray,
    labels_of_region: list[int],
    palette: Palette,
    *,
    stage_name: str,
    stage_version: str,
    config_hash: str,
    source_hash: str,
) -> RegionGraph:
    """Re-derive a full ``RegionGraph`` (records, adjacency, ΔE00 edges) from a
    mutated component map + per-region palette labels, reusing the same
    closed-form record/adjacency math as ``components.py`` (imported lazily to
    avoid a circular import between the graph stages)."""
    from mysterycbn.stages.graph.components import _adjacency, _region_records

    # _region_records expects a per-pixel *label* array to read each region's
    # label from its first pixel; synthesise one from labels_of_region.
    label_lut = np.array(labels_of_region, dtype=np.int64)
    per_pixel_label = label_lut[component_map]
    records = _region_records(per_pixel_label, component_map)
    boundary, border_len = _adjacency(component_map)

    perimeter = border_len.astype(np.int64).copy()
    for (a, b), w_len in boundary.items():
        perimeter[a] += w_len
        perimeter[b] += w_len

    table = palette.delta_e_table
    regions = tuple(
        Region(
            region_id=i,
            label=rec["label"],  # type: ignore[arg-type]
            area_px=rec["area_px"],  # type: ignore[arg-type]
            bbox=rec["bbox"],  # type: ignore[arg-type]
            seed_px=rec["seed_px"],  # type: ignore[arg-type]
            perimeter_px=int(perimeter[i]),
            centroid=rec["centroid"],  # type: ignore[arg-type]
        )
        for i, rec in enumerate(records)
    )
    edges = tuple(
        (a, b, float(table[regions[a].label, regions[b].label]), w_len)
        for (a, b), w_len in sorted(boundary.items())
    )
    return RegionGraph(
        regions=regions,
        component_map=component_map,
        edges=edges,
        provenance=Provenance(
            stage_name=stage_name,
            stage_version=stage_version,
            config_hash=config_hash,
            source_hash=source_hash,
        ),
    )
