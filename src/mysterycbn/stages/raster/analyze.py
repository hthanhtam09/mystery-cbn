"""Color Analysis stage: global statistics → advisory auto-tune proposals
(ENGINE_SPEC.md §6; see docs/modules/analyze.md).

All statistics are closed-form and O(N): deterministic, explainable, and
rotation/mirror-invariant. Proposals land in the AUTO_TUNE config layer,
which may only fill values the user left unset (ARCHITECTURE.md §7) —
explicit human intent always wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import cv2
import numpy as np

from mysterycbn.foundation.color import DefaultColorScience
from mysterycbn.foundation.errors import ConfigError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import ImageStats, Provenance, RasterImage

STAGE_NAME = "analyze"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

_COLOR = DefaultColorScience()
_HIST_BINS = 64
_EDGE_THRESHOLD = 0.1  # on L*/100 gradient magnitude (ENGINE_SPEC §6.2)


@dataclass(frozen=True)
class AutoTuneProposal:
    """Advisory config fragment for the AUTO_TUNE layer (fill-only)."""

    fragment: Mapping[str, Mapping[str, object]]
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fragment",
            MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.fragment.items()}),
        )


def compute_stats(raster: RasterImage, *, config_hash: str = _UNSET_HASH) -> ImageStats:
    """Compute all §6 statistics in one pass over the working raster."""
    lab = _COLOR.srgb_to_lab(raster.pixels)
    lightness = lab[..., 0]

    # Colorfulness (Hasler–Süsstrunk, §6.1) — from foundation, one implementation.
    colorfulness = _COLOR.colorfulness(raster.pixels)

    # Edge density (§6.2): Sobel magnitude on L*/100, fraction above threshold.
    l_norm = (lightness / 100.0).astype(np.float32)
    gx = cv2.Sobel(l_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_norm, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = float(np.count_nonzero(np.hypot(gx, gy) > _EDGE_THRESHOLD) / l_norm.size)

    # Luminance histogram + entropy (§6.3). Clamp: conversion noise can land
    # an epsilon outside [0, 100], which np.histogram would silently drop.
    hist, _ = np.histogram(np.clip(lightness, 0.0, 100.0), bins=_HIST_BINS, range=(0.0, 100.0))
    hist_norm = hist.astype(np.float64) / lightness.size
    nonzero = hist_norm[hist_norm > 0.0]
    entropy_bits = float(-np.sum(nonzero * np.log2(nonzero)))

    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return ImageStats(
        colorfulness=colorfulness,
        edge_density=edge_density,
        luminance_histogram=hist_norm,
        lab_mean=tuple(float(v) for v in lab.reshape(-1, 3).mean(axis=0)),  # type: ignore[arg-type]
        lab_std=tuple(float(v) for v in lab.reshape(-1, 3).std(axis=0)),  # type: ignore[arg-type]
        brightness=float(lightness.mean()),
        contrast=float(lightness.std()),
        saturation=float(chroma.mean()),
        entropy_bits=entropy_bits,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=raster.provenance.source_hash,
        ),
    )


def propose_overrides(stats: ImageStats, *, k_min: int = 8, k_max: int = 30) -> AutoTuneProposal:
    """Translate statistics into the §6.4–6.5 advisory config fragment.

    ``k* = clip(round(6 + 0.12·M + 6·ρ + 0.8·H_L), k_min, k_max)`` — busy,
    colorful, tonally rich images earn more palette entries. Smoothing:
    ρ > 0.25 proposes 3 passes; ρ < 0.05 proposes 1; otherwise no proposal
    (the built-in default stands).
    """
    if not 2 <= k_min <= k_max <= 64:
        raise ConfigError(f"k bounds must satisfy 2 ≤ k_min ≤ k_max ≤ 64, got {k_min}, {k_max}")
    k_star = int(
        np.clip(
            round(
                6.0
                + 0.12 * stats.colorfulness
                + 6.0 * stats.edge_density
                + 0.8 * stats.entropy_bits
            ),
            k_min,
            k_max,
        )
    )
    fragment: dict[str, dict[str, object]] = {"quantize": {"n_colors": k_star}}
    if stats.edge_density > 0.25:
        fragment["preprocess"] = {"smooth_passes": 3}
    elif stats.edge_density < 0.05:
        fragment["preprocess"] = {"smooth_passes": 1}
    return AutoTuneProposal(fragment=fragment, provenance=stats.provenance)


class AnalyzeStage:
    """Stage wrapper: ``raster_working`` → ``image_stats`` + ``auto_tune``."""

    def __init__(self, section: Mapping[str, object], config_hash: str = _UNSET_HASH) -> None:
        enabled = section.get("enabled", True)
        k_min = section.get("k_min", 8)
        k_max = section.get("k_max", 30)
        if (
            not isinstance(enabled, bool)
            or not isinstance(k_min, int)
            or not isinstance(k_max, int)
        ):
            raise ConfigError("analyze config: enabled must be bool, k_min/k_max int")
        self._enabled = enabled
        self._k_min = k_min
        self._k_max = k_max
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("raster_working",)

    @property
    def provides(self) -> tuple[str, ...]:
        return ("image_stats", "auto_tune")

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        raster = ctx.get("raster_working")
        if not isinstance(raster, RasterImage):
            raise ConfigError(f"artifact 'raster_working' has wrong type {type(raster).__name__}")
        stats = compute_stats(raster, config_hash=self._config_hash)
        ctx.put("image_stats", stats)
        fragment: AutoTuneProposal
        if self._enabled:
            fragment = propose_overrides(stats, k_min=self._k_min, k_max=self._k_max)
        else:
            fragment = AutoTuneProposal(fragment={}, provenance=stats.provenance)
        ctx.put("auto_tune", fragment)
