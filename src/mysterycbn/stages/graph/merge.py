"""Tiny Region Merge stage: enforce the printability area floor
(ENGINE_SPEC.md §11; cost function MATH_SPEC §11).

Smallest-first greedy merge with perceptual cost:

1. **Area floor** — ``A_min = π·(d_min_mm/2)²·ppmm²`` (working px² of the
   smallest physically colorable disc).
2. **Priority queue** — min-heap of sub-floor regions keyed by
   ``(area, region_id)``; stale entries are invalidated lazily (an entry is
   dead if its region was absorbed or its recorded area is outdated).
3. **Merge cost** — fold region ``r`` into ``argmin_n C(r, n)`` with
   ``C(r, n) = ΔE00(r, n) − λ · w_len(r, n) / perim(r)``: color similarity
   decides clear cases, the boundary-hug term breaks near-ties toward the
   neighbor the sliver hugs. Ties → larger neighbor, then lower id.
4. **Update** — the absorber inherits edges (boundary lengths summed),
   ``perim(n′) = perim(n) + perim(r) − 2·w_len(r, n)`` (inclusion–exclusion);
   if it is still sub-floor it re-enters the heap.
5. **Compaction** — surviving regions keep their relative order under new
   dense ids; palette entries with zero coverage are dropped and renumbered
   (renumber map returned for provenance). Records and adjacency are
   re-derived from the merged component map — the double-entry boundary
   identity re-asserts itself on the output.

Degenerate page (every region sub-floor, or fewer than two colors survive):
merging stops at R = 1 legally; palette compaction is skipped when it would
leave < 2 colors (the model's palette floor), recorded as an identity map.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping

import numpy as np

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import (
    Palette,
    PaletteColor,
    Provenance,
    Region,
    RegionGraph,
)
from mysterycbn.stages.graph.components import _adjacency, _region_records

STAGE_NAME = "merge_tiny"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

LAMBDA_BOUNDARY_DEFAULT = 15.0
_LAMBDA_MAX = 50.0


def area_floor_px(d_min_mm: float, work_scale: float) -> float:
    """Printability area floor ``A_min = π (d_min/2)² ppmm²`` in working px²
    (ENGINE_SPEC §11.1; ppmm from ``work_scale``, MATH_SPEC §2)."""
    if d_min_mm <= 0 or work_scale <= 0:
        raise ConfigError(f"invalid floor parameters: d_min_mm={d_min_mm}, work_scale={work_scale}")
    ppmm = 1.0 / (work_scale * MM_PER_INCH / PT_PER_INCH)
    return math.pi * (d_min_mm / 2.0) ** 2 * ppmm**2


def merge_cost(delta_e: float, boundary_len: int, perimeter: int, lambda_boundary: float) -> float:
    """``C(r, n) = ΔE00 − λ · w_len(r, n) / perim(r)`` (MATH_SPEC §11)."""
    return delta_e - lambda_boundary * boundary_len / perimeter


def merge_tiny_regions(
    graph: RegionGraph,
    palette: Palette,
    *,
    a_min: float,
    lambda_boundary: float = LAMBDA_BOUNDARY_DEFAULT,
    protect_dark_l: float = 0.0,
    protect_dark_delta_l: float = 0.0,
    config_hash: str = _UNSET_HASH,
) -> tuple[RegionGraph, Palette, tuple[int, ...]]:
    """Full §11 merge. Returns (new graph, new palette, palette renumber map).

    ``renumber[old_index]`` is the compacted palette index, or −1 for a
    dropped color; it is the identity when compaction was skipped.

    ``protect_dark_l`` > 0 keeps a sub-floor region OUT of the merge when it is
    a semantic dark dot -- its palette L* is below ``protect_dark_l`` AND every
    neighbour is at least ``protect_dark_delta_l`` lighter (a dark pupil/nostril
    on a light surround). Such features are smaller than the printability floor
    yet visually essential and asymmetric to lose; the ink layer can outline
    them but only a surviving region carries the dark FILL. Downstream they are
    micro-labelled / left unnumbered like any other sub-floor region.
    """
    if a_min >= graph.component_map.size:
        raise ConfigError(f"area floor {a_min:.0f} px² exceeds the content area")
    n = len(graph.regions)
    area = np.array([r.area_px for r in graph.regions], dtype=np.int64)
    perim = np.array([r.perimeter_px for r in graph.regions], dtype=np.int64)
    label = np.array([r.label for r in graph.regions], dtype=np.int64)
    table = palette.delta_e_table
    palette_l = np.array([c.lab[0] for c in palette.colors], dtype=np.float64)
    neighbors: list[dict[int, int]] = [{} for _ in range(n)]
    for a, b, _, w_len in graph.edges:
        neighbors[a][b] = w_len
        neighbors[b][a] = w_len

    parent = np.arange(n, dtype=np.int64)  # absorbed region → absorber
    alive = np.ones(n, dtype=bool)
    heap = [(int(area[i]), i) for i in range(n) if area[i] < a_min]
    heapq.heapify(heap)

    while heap:
        popped_area, r = heapq.heappop(heap)
        if not alive[r] or popped_area != area[r] or area[r] >= a_min:
            continue  # lazily invalidated: absorbed, grown, or now legal
        if not neighbors[r]:
            continue  # R = 1: degenerate page, legal (validator warns)
        if protect_dark_l > 0.0:
            r_l = palette_l[label[r]]
            if r_l < protect_dark_l and all(
                palette_l[label[m]] - r_l >= protect_dark_delta_l for m in neighbors[r]
            ):
                continue  # protected dark dot (pupil/nostril): keep, never merge

        def cost_key(item: tuple[int, int], r: int = r) -> tuple[float, int, int]:
            m, w_len = item
            cost = merge_cost(
                float(table[label[r], label[m]]), w_len, int(perim[r]), lambda_boundary
            )
            return (cost, -int(area[m]), m)

        target, w_rn = min(neighbors[r].items(), key=cost_key)
        area[target] += area[r]
        perim[target] = perim[target] + perim[r] - 2 * w_rn
        del neighbors[target][r]
        for m, w_len in neighbors[r].items():
            if m == target:
                continue
            neighbors[target][m] = neighbors[target].get(m, 0) + w_len
            del neighbors[m][r]
            neighbors[m][target] = neighbors[target][m]
        neighbors[r] = {}
        alive[r] = False
        parent[r] = target
        if area[target] < a_min:
            heapq.heappush(heap, (int(area[target]), target))

    # Resolve absorption chains, then compact surviving ids order-preservingly.
    root = np.arange(n, dtype=np.int64)
    for i in range(n):
        while parent[root[i]] != root[i]:
            root[i] = parent[root[i]]
    survivors = np.flatnonzero(alive)
    new_id = np.full(n, -1, dtype=np.int64)
    new_id[survivors] = np.arange(survivors.size)
    component_map = np.asarray(new_id[root[graph.component_map]], dtype=np.int32)

    # Palette compaction by final coverage; skipped below the K ≥ 2 floor.
    region_label = label[survivors]
    coverage = np.bincount(region_label, weights=area[survivors], minlength=palette.size)
    kept = np.flatnonzero(coverage > 0)
    renumber = np.full(palette.size, -1, dtype=np.int64)
    if kept.size >= 2:
        renumber[kept] = np.arange(kept.size)
        new_palette = Palette(
            colors=tuple(
                PaletteColor.from_lab(int(renumber[i]), palette.colors[i].lab, int(coverage[i]))
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
    else:
        renumber = np.arange(palette.size, dtype=np.int64)
        new_palette = palette

    # Re-derive records and adjacency from the merged component map. Note two
    # surviving same-label regions may now touch orthogonally — the component
    # map, not the label partition, is authoritative here.
    merged_labels = np.asarray(renumber[region_label][component_map], dtype=np.int32)
    records = _region_records(merged_labels, component_map)
    boundary, border_len = _adjacency(component_map)
    perimeter_out = border_len.astype(np.int64).copy()
    for (a, b), w_len in boundary.items():
        perimeter_out[a] += w_len
        perimeter_out[b] += w_len

    out_table = new_palette.delta_e_table
    regions = tuple(
        Region(
            region_id=i,
            label=rec["label"],  # type: ignore[arg-type]
            area_px=rec["area_px"],  # type: ignore[arg-type]
            bbox=rec["bbox"],  # type: ignore[arg-type]
            seed_px=rec["seed_px"],  # type: ignore[arg-type]
            perimeter_px=int(perimeter_out[i]),
            centroid=rec["centroid"],  # type: ignore[arg-type]
        )
        for i, rec in enumerate(records)
    )
    edges = tuple(
        (a, b, float(out_table[regions[a].label, regions[b].label]), w_len)
        for (a, b), w_len in sorted(boundary.items())
    )
    new_graph = RegionGraph(
        regions=regions,
        component_map=component_map,
        edges=edges,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=graph.provenance.source_hash,
        ),
    )
    return new_graph, new_palette, tuple(int(v) for v in renumber)


class MergeTinyStage:
    """Stage wrapper: (``region_graph``, ``palette``) → both replaced."""

    def __init__(
        self,
        section: Mapping[str, object],
        *,
        d_min_mm: float = 3.5,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        lam = section.get("lambda_boundary", LAMBDA_BOUNDARY_DEFAULT)
        if not isinstance(lam, (int, float)) or not 0.0 <= float(lam) <= _LAMBDA_MAX:
            raise ConfigError(
                f"merge config: lambda_boundary must be in [0, {_LAMBDA_MAX}], got {lam!r}"
            )
        enabled = section.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"merge config: enabled must be a bool, got {enabled!r}")

        def _num(key: str) -> float:
            v = section.get(key, 0.0)
            if not isinstance(v, (int, float)) or float(v) < 0.0:
                raise ConfigError(f"merge config: {key} must be a number ≥ 0, got {v!r}")
            return float(v)

        self._enabled = enabled
        self._lambda = float(lam)
        self._protect_dark_l = _num("protect_dark_l")
        self._protect_dark_delta_l = _num("protect_dark_delta_l")
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
        return ("region_graph", "palette")

    @property
    def config_section(self) -> str:
        return "merge"

    def run(self, ctx: PipelineContext) -> None:
        graph = ctx.get("region_graph")
        palette = ctx.get("palette")
        raster = ctx.get("raster_working")
        if not isinstance(graph, RegionGraph) or not isinstance(palette, Palette):
            raise ConfigError("merge_tiny requires RegionGraph + Palette artifacts")
        if not self._enabled:
            # Dense/decorative mode: keep every region, however small, so the
            # page can be tiled into many small numbered cells (the opposite
            # of the default region-minimizing goal). The artifacts are
            # already bound in ctx; re-put them unchanged to satisfy the
            # stage's provides-contract without merging anything.
            ctx.put("region_graph", graph)
            ctx.put("palette", palette)
            return
        work_scale = getattr(raster, "work_scale", 0.0)
        new_graph, new_palette, _ = merge_tiny_regions(
            graph,
            palette,
            a_min=area_floor_px(self._d_min_mm, work_scale),
            lambda_boundary=self._lambda,
            protect_dark_l=self._protect_dark_l,
            protect_dark_delta_l=self._protect_dark_delta_l,
            config_hash=self._config_hash,
        )
        ctx.put("region_graph", new_graph)
        ctx.put("palette", new_palette)
