"""Noise Removal stage: label-map cleanup before component analysis
(ENGINE_SPEC.md §8).

Two label-domain-safe operations, in order:

1. **Modal filter** — iterated 3×3 majority (the label-space analogue of a
   morphological rank/median cleanup): a pixel takes the most frequent label
   in its 8-neighborhood including itself. Ties break by smallest ΔE00 to the
   pixel's current palette color, then lowest label id. Runs to fixpoint or
   ``max_modal_iters``.
2. **Area opening** — 4-connected components smaller than the speck threshold
   are relabeled to a neighbor: non-speck neighbors are always preferred
   (a speck must not vanish into another vanishing speck when a surviving
   region touches it), then longest shared boundary, then smallest ΔE00,
   then lowest label id. Specks are processed in ascending
   (area, min-pixel-index) order; speck→speck chains resolve to the final
   surviving region's label.

Neither step can invent labels or move a boundary by more than one modal
radius. Per-label morphological open/close was rejected (ENGINE_SPEC §8
alternatives): overlapping per-label operations create label conflicts; the
modal + area-opening pair is the standard conflict-free formulation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from skimage.measure import label as cc_label

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import LabelMap, Palette, Provenance

STAGE_NAME = "denoise"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

# Score packing for the vectorized modal argmax: counts (integer ≤ 9) are
# primary, ΔE00 (≤ ~150) secondary, label id tertiary.
_COUNT_WEIGHT = 1.0e6
_ID_WEIGHT = 1.0e-6


def speck_threshold(d_min_mm: float, work_scale: float, divisor: int = 16) -> int:
    """Speck area floor: ``max(4, ⌊A_min / divisor⌋)`` px (ENGINE_SPEC §8.2).

    ``A_min = π (d_min/2)² ppmm²`` with ppmm derived from ``work_scale``
    (pt/px → px per printed mm, MATH_SPEC §2).
    """
    if d_min_mm <= 0 or work_scale <= 0 or divisor < 1:
        raise ConfigError(
            f"invalid speck parameters: d_min_mm={d_min_mm}, "
            f"work_scale={work_scale}, divisor={divisor}"
        )
    ppmm = 1.0 / (work_scale * MM_PER_INCH / PT_PER_INCH)
    a_min = math.pi * (d_min_mm / 2.0) ** 2 * ppmm**2
    return max(4, int(a_min / divisor))


def _neighborhood_counts(padded: np.ndarray, label_value: int) -> np.ndarray:
    """Occurrences of ``label_value`` in each pixel's 3×3 neighborhood."""
    h, w = padded.shape[0] - 2, padded.shape[1] - 2
    counts = np.zeros((h, w), dtype=np.uint8)
    mask = (padded == label_value).astype(np.uint8)
    for dy in range(3):
        for dx in range(3):
            counts += mask[dy : dy + h, dx : dx + w]
    return counts


def modal_filter(labels: np.ndarray, palette: Palette, *, max_iters: int = 3) -> np.ndarray:
    """Iterated 3×3 majority filter with the §8.1 deterministic tie rules.

    Two-pass per iteration: a cheap uint8 count/argmax decides every pixel
    with a unique majority; the ΔE00 tie-break (then lowest label id) runs
    only on the sparse count-tie pixels.
    """
    current = labels.astype(np.int32)
    table = palette.delta_e_table
    for _ in range(max_iters):
        padded = np.pad(current, 1, constant_values=-1)
        present = [int(v) for v in np.unique(current)]
        count_stack = np.stack(
            [_neighborhood_counts(padded, lab) for lab in present], axis=-1
        )  # (H, W, P) uint8
        best_idx = np.argmax(count_stack, axis=-1)  # first max → lowest label id
        best_count = np.take_along_axis(count_stack, best_idx[..., None], axis=-1)[..., 0]
        ties = (count_stack == best_count[..., None]).sum(axis=-1) > 1

        present_arr = np.asarray(present, dtype=np.int32)
        best_label = present_arr[best_idx]
        if np.any(ties):
            idx = np.nonzero(ties.ravel())[0]
            tie_counts = count_stack.reshape(-1, len(present))[idx].astype(np.float64)
            delta = table[np.ix_(current.ravel()[idx], present_arr)]
            score = tie_counts * _COUNT_WEIGHT - delta - present_arr[None, :] * _ID_WEIGHT
            best_label.ravel()[idx] = present_arr[np.argmax(score, axis=1)]
        if np.array_equal(best_label, current):
            break
        current = best_label.astype(np.int32)
    return current


def area_opening(labels: np.ndarray, palette: Palette, *, speck_px: int) -> np.ndarray:
    """Absorb 4-connected components with area < ``speck_px`` into their
    longest-shared-boundary neighbor (§8.2)."""
    out = labels.astype(np.int32).copy()
    raw = cc_label(out, connectivity=1, background=-1)  # type: ignore[no-untyped-call]
    comps = np.asarray(raw, dtype=np.int64) - 1
    n_comps = int(comps.max()) + 1
    areas = np.bincount(comps.ravel(), minlength=n_comps)

    # Component adjacency with shared crack-edge counts (pixel-pair sweep).
    pairs: dict[tuple[int, int], int] = {}
    for a, b in (
        (comps[:, :-1].ravel(), comps[:, 1:].ravel()),
        (comps[:-1, :].ravel(), comps[1:, :].ravel()),
    ):
        diff = a != b
        lo = np.minimum(a[diff], b[diff])
        hi = np.maximum(a[diff], b[diff])
        keys, counts = np.unique(np.stack([lo, hi]), axis=1, return_counts=True)
        for (ca, cb), n in zip(keys.T, counts, strict=True):
            pairs[(int(ca), int(cb))] = pairs.get((int(ca), int(cb)), 0) + int(n)

    neighbors: dict[int, list[tuple[int, int]]] = {}
    for (ca, cb), n in pairs.items():
        neighbors.setdefault(ca, []).append((cb, n))
        neighbors.setdefault(cb, []).append((ca, n))

    comp_label = np.zeros(n_comps, dtype=np.int32)
    first_index = np.full(n_comps, comps.size, dtype=np.int64)
    flat_comps = comps.ravel()
    flat_labels = out.ravel()
    seen_first = np.unique(flat_comps, return_index=True)
    comp_label[seen_first[0]] = flat_labels[seen_first[1]]
    first_index[seen_first[0]] = seen_first[1]

    table = palette.delta_e_table
    is_speck = areas < speck_px
    specks = [c for c in range(n_comps) if is_speck[c]]
    specks.sort(key=lambda c: (int(areas[c]), int(first_index[c])))
    # Each speck points at its chosen absorber; chains (speck → speck → region)
    # are resolved after the pass, so a speck absorbed into another speck ends
    # up with the FINAL label of the chain's surviving region.
    target = np.arange(n_comps, dtype=np.int64)
    for comp in specks:
        if comp not in neighbors:
            continue  # isolated single-component map
        own = int(comp_label[comp])

        def choice_key(item: tuple[int, int], own: int = own) -> tuple[bool, int, float, int]:
            neighbor, boundary = item
            return (
                bool(is_speck[neighbor]),  # surviving regions first
                -boundary,
                float(table[own, int(comp_label[neighbor])]),
                int(comp_label[neighbor]),
            )

        target[comp] = min(neighbors[comp], key=choice_key)[0]

    final_label = comp_label.copy()
    for comp in range(n_comps):
        root, hops = comp, 0
        while target[root] != root and hops <= n_comps:
            root, hops = int(target[root]), hops + 1
        if hops > n_comps:  # cycle of mutually-absorbing specks: keep original
            root = comp
        final_label[comp] = comp_label[root]
    return np.asarray(final_label[comps])


def denoise_label_map(
    label_map: LabelMap,
    palette: Palette,
    *,
    max_modal_iters: int = 3,
    speck_px: int = 4,
    config_hash: str = _UNSET_HASH,
) -> LabelMap:
    """Full §8 cleanup: modal filter to fixpoint, then area opening."""
    if max_modal_iters < 0:
        raise ConfigError(f"max_modal_iters must be ≥ 0, got {max_modal_iters}")
    label_map.validate_against(palette)
    labels = label_map.labels
    if max_modal_iters > 0:
        labels = modal_filter(labels, palette, max_iters=max_modal_iters)
    labels = area_opening(labels, palette, speck_px=speck_px)
    return LabelMap(
        labels=labels,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=label_map.provenance.source_hash,
        ),
    )


class DenoiseStage:
    """Stage wrapper: (``label_map``, ``palette``) → ``label_map`` (replaced)."""

    def __init__(
        self,
        section: Mapping[str, object],
        *,
        d_min_mm: float = 3.5,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        max_iters = section.get("max_modal_iters", 3)
        divisor = section.get("speck_divisor", 16)
        if not isinstance(max_iters, int) or not isinstance(divisor, int):
            raise ConfigError("denoise config: max_modal_iters and speck_divisor must be int")
        self._max_iters = max_iters
        self._divisor = divisor
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
        return ("label_map", "palette", "raster_working")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("label_map",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        label_map = ctx.get("label_map")
        palette = ctx.get("palette")
        raster = ctx.get("raster_working")
        if not isinstance(label_map, LabelMap) or not isinstance(palette, Palette):
            raise ConfigError("denoise requires LabelMap + Palette artifacts")
        work_scale = getattr(raster, "work_scale", 0.0)
        ctx.put(
            "label_map",
            denoise_label_map(
                label_map,
                palette,
                max_modal_iters=self._max_iters,
                speck_px=speck_threshold(self._d_min_mm, work_scale, self._divisor),
                config_hash=self._config_hash,
            ),
        )
