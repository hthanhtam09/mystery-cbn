"""Hole-fill stage: absorb small, fully-enclosed label components into their
surrounding region (ENGINE_SPEC denoise addendum).

Quantization can split a tiny feature into a ring + an enclosed core of a
different label -- e.g. a small eye pupil whose white catchlight becomes a
disconnected island of the eye-white label sitting *inside* the dark pupil.
That island is a spatially-disconnected component of a larger region, so the
region-level ``merge_tiny`` can never absorb it; it renders as a pale hole,
making the pupil read as a broken donut.

This stage works at the connected-component level: a component that (a) does
not touch the image border, (b) is bordered by exactly ONE other component,
and (c) is smaller than ``max_hole_mm2`` is relabelled to that single
enclosing component's label. Small enclosed holes are almost always such
artifacts; the size gate keeps legitimate large shapes untouched.

Disabled (``enabled=False``, the default outside dense/partial) is a no-op.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from skimage.measure import label as cc_label

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import LabelMap, Provenance

STAGE_NAME = "fill_holes"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64
MAX_HOLE_MM2_DEFAULT = 1.5


def fill_small_holes(labels: np.ndarray, max_hole_px: int) -> np.ndarray:
    """Relabel small, single-neighbour, non-border components to their
    enclosing component's label. Returns a new label array (input unchanged)."""
    if max_hole_px <= 0:
        return labels
    comps = cc_label(np.asarray(labels), background=-1, connectivity=1)
    n = int(comps.max()) + 1
    if n <= 1:
        return labels

    # Per-component: set of neighbouring component ids, and whether on border.
    neighbors: list[set[int]] = [set() for _ in range(n)]

    def _accumulate(a: np.ndarray, b: np.ndarray) -> None:
        m = a != b
        for x, y in zip(a[m].tolist(), b[m].tolist(), strict=True):
            neighbors[x].add(y)
            neighbors[y].add(x)

    _accumulate(comps[:-1, :], comps[1:, :])
    _accumulate(comps[:, :-1], comps[:, 1:])

    on_border = np.zeros(n, dtype=bool)
    for edge in (comps[0, :], comps[-1, :], comps[:, 0], comps[:, -1]):
        on_border[np.unique(edge)] = True

    areas = np.bincount(comps.ravel(), minlength=n)
    labels_flat = np.asarray(labels)
    out = labels_flat.copy()
    # A representative pixel per component to read its enclosing label from.
    changed = False
    for c in range(n):
        if on_border[c] or areas[c] > max_hole_px or len(neighbors[c]) != 1:
            continue
        (enclosing,) = tuple(neighbors[c])
        mask = comps == c
        enc_label = int(labels_flat[comps == enclosing][0])
        if enc_label == int(labels_flat[mask][0]):
            continue  # already same label value (no visible change)
        out[mask] = enc_label
        changed = True
    return out if changed else labels


def _ppmm(work_scale: float) -> float:
    if work_scale <= 0.0:
        return 0.0
    return 1.0 / (work_scale * MM_PER_INCH / PT_PER_INCH)


class FillHolesStage:
    """Stage wrapper: (``label_map``, ``raster_working``) → ``label_map`` (replaced)."""

    def __init__(
        self,
        section: Mapping[str, object],
        *,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        enabled = section.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"fill_holes config: enabled must be a bool, got {enabled!r}")
        max_hole = section.get("max_hole_mm2", MAX_HOLE_MM2_DEFAULT)
        if not isinstance(max_hole, (int, float)) or float(max_hole) < 0.0:
            raise ConfigError(f"fill_holes config: max_hole_mm2 must be ≥ 0, got {max_hole!r}")
        self._enabled = enabled
        self._max_hole_mm2 = float(max_hole)
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("label_map", "raster_working")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("label_map",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        label_map = ctx.get("label_map")
        raster = ctx.get("raster_working")
        if not isinstance(label_map, LabelMap):
            raise ConfigError("fill_holes requires a LabelMap artifact")
        if not self._enabled:
            return  # no-op; label_map already bound
        ppmm = _ppmm(getattr(raster, "work_scale", 0.0))
        max_hole_px = int(math.floor(self._max_hole_mm2 * ppmm * ppmm)) if ppmm > 0 else 0
        filled = fill_small_holes(label_map.labels, max_hole_px)
        if filled is label_map.labels:
            return  # nothing changed
        ctx.put(
            "label_map",
            LabelMap(
                labels=filled.astype(np.int32),
                provenance=Provenance(
                    stage_name=STAGE_NAME,
                    stage_version=STAGE_VERSION,
                    config_hash=self._config_hash,
                    source_hash=label_map.provenance.source_hash,
                ),
            ),
        )
