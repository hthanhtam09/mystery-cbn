"""Connected Components stage: label map → ``RegionGraph``
(ENGINE_SPEC.md §9; adjacency per §10 / MATH_SPEC §5.1).

Two steps, both exact and deterministic:

1. **Component labeling** — maximal 4-connected sets of equal-label pixels
   (union-find via ``skimage.measure.label``, connectivity non-configurable),
   renumbered to raster-scan first-occurrence order so region ids are dense
   and top-left-first. Per region: palette label, ``area_px``, tight ``bbox``,
   ``seed_px`` (first pixel in raster order), ``centroid`` (mean pixel center).
2. **Adjacency sweep** — every 4-adjacent pixel pair with differing region
   ids contributes one crack edge to ``w_len(a, b)``; pairs with the page
   border accumulate into the region's border length. Edge color weight is
   ``ΔE00(palette[label(a)], palette[label(b)])``; region perimeter is
   ``Σ_b w_len(a, b) + border_len(a)``.

The double-entry identity ``Σ_e w_len + Σ_r border = B`` (total crack-edge
count of the label map) holds by construction and is asserted here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from skimage.measure import label as cc_label

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import LabelMap, Palette, Provenance, Region, RegionGraph

STAGE_NAME = "regions"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64


def label_components(labels: np.ndarray) -> np.ndarray:
    """4-connected component map with dense int32 ids in raster-scan
    first-occurrence order (ENGINE_SPEC §9)."""
    raw = cc_label(labels, connectivity=1, background=-1)  # type: ignore[no-untyped-call]
    comps = np.asarray(raw, dtype=np.int64) - 1
    ids, first = np.unique(comps.ravel(), return_index=True)
    order = np.argsort(first, kind="stable")  # provisional id → rank by first pixel
    remap = np.empty(ids.size, dtype=np.int64)
    remap[order] = np.arange(ids.size)
    return np.asarray(remap[comps], dtype=np.int32)


def _region_records(labels: np.ndarray, component_map: np.ndarray) -> list[dict[str, object]]:
    """Per-region statistics (label, area, bbox, seed, centroid) via bincounts."""
    w = component_map.shape[1]
    flat = component_map.ravel().astype(np.int64)
    n = int(flat.max()) + 1
    areas = np.bincount(flat, minlength=n)

    # Run-length reduce: horizontal runs of one region id never cross rows
    # (row starts are forced run breaks), so per-run stats are closed-form and
    # the scatter reductions touch O(runs) elements, not O(N).
    breaks = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    run_start = np.unique(np.concatenate([np.arange(0, flat.size, w), breaks, [0]]))
    run_end = np.append(run_start[1:], flat.size)
    run_len = run_end - run_start
    run_id = flat[run_start]
    run_row = run_start // w
    run_c0 = run_start % w
    run_c1 = (run_end - 1) % w

    row_max = np.full(n, -1, dtype=np.int64)
    col_min = np.full(n, w, dtype=np.int64)
    col_max = np.full(n, -1, dtype=np.int64)
    first = np.full(n, flat.size, dtype=np.int64)
    np.maximum.at(row_max, run_id, run_row)
    np.minimum.at(col_min, run_id, run_c0)
    np.maximum.at(col_max, run_id, run_c1)
    np.minimum.at(first, run_id, run_start)
    row_min = first // w  # raster order: the first pixel sits in the minimal row

    centroid_r = np.bincount(run_id, weights=run_len * run_row, minlength=n) / areas
    centroid_c = np.bincount(run_id, weights=run_len * (run_c0 + run_c1) / 2.0, minlength=n) / areas
    return [
        {
            "label": int(labels.ravel()[first[i]]),
            "area_px": int(areas[i]),
            "bbox": (int(row_min[i]), int(col_min[i]), int(row_max[i]), int(col_max[i])),
            "seed_px": (int(first[i] // w), int(first[i] % w)),
            "centroid": (float(centroid_r[i]), float(centroid_c[i])),
        }
        for i in range(n)
    ]


def _adjacency(
    component_map: np.ndarray,
) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """Pixel-pair sweep: shared boundary lengths per region pair + per-region
    page-border lengths (ENGINE_SPEC §10)."""
    n = int(component_map.max()) + 1
    boundary: dict[tuple[int, int], int] = {}
    for a, b in (
        (component_map[:, :-1].ravel(), component_map[:, 1:].ravel()),
        (component_map[:-1, :].ravel(), component_map[1:, :].ravel()),
    ):
        diff = a != b
        lo = np.minimum(a[diff], b[diff]).astype(np.int64)
        hi = np.maximum(a[diff], b[diff]).astype(np.int64)
        keys, counts = np.unique(lo * n + hi, return_counts=True)  # scalar-encoded pairs
        for packed, cnt in zip(keys, counts, strict=True):
            key = (int(packed // n), int(packed % n))
            boundary[key] = boundary.get(key, 0) + int(cnt)

    border_pixels = np.concatenate(
        [component_map[0, :], component_map[-1, :], component_map[:, 0], component_map[:, -1]]
    ).astype(np.int64)
    border_len = np.bincount(border_pixels, minlength=n)
    return boundary, border_len


def build_region_graph(
    label_map: LabelMap,
    palette: Palette,
    *,
    config_hash: str = _UNSET_HASH,
) -> RegionGraph:
    """Full §9+§10 construction: components, records, adjacency, ΔE00 edges."""
    label_map.validate_against(palette)
    labels = label_map.labels
    component_map = label_components(labels)
    records = _region_records(labels, component_map)
    boundary, border_len = _adjacency(component_map)

    perimeter = border_len.astype(np.int64).copy()
    for (a, b), w_len in boundary.items():
        perimeter[a] += w_len
        perimeter[b] += w_len

    h, w = labels.shape
    total_cracks = (
        int((labels[:, :-1] != labels[:, 1:]).sum())
        + int((labels[:-1, :] != labels[1:, :]).sum())
        + 2 * (h + w)
    )
    # Components refine the label partition, so label cracks == region cracks.
    assert sum(boundary.values()) + int(border_len.sum()) == total_cracks, (
        "boundary double-entry identity violated (MATH_SPEC §5.1)"
    )

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
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=label_map.provenance.source_hash,
        ),
    )


class ConnectedComponentsStage:
    """Stage wrapper: (``label_map``, ``palette``) → ``region_graph``."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        # ENGINE_SPEC §9: no configuration parameters; connectivity is an
        # invariant, not a knob.
        del section
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("label_map", "palette")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("region_graph",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        label_map = ctx.get("label_map")
        palette = ctx.get("palette")
        if not isinstance(label_map, LabelMap) or not isinstance(palette, Palette):
            raise ConfigError("regions requires LabelMap + Palette artifacts")
        ctx.put(
            "region_graph",
            build_region_graph(label_map, palette, config_hash=self._config_hash),
        )
