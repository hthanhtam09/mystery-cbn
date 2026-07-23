"""Ink-line detection stage: recover thin dark line work that quantization
loses (ENGINE_SPEC ink-line addendum).

A large detailed subject (e.g. a cartoon lion's face) carries low-contrast
thin lines -- whiskers, fine muzzle/eye line work -- that color quantization
maps into the surrounding fill, so they vanish from the trace. High-contrast
black lines (eyebrows, mouth outlines) survive as their own dark regions;
low-contrast thin ones do not. This stage detects the thin dark lines in the
*pre-quantize* working raster (where the information still exists) and emits a
boolean ``InkMask``; a later vector stage turns it into a render-only black
overlay (see ``stages/vector/ink_overlay.py`` / ``model/ink.py``).

Detection is a black-hat ridge extractor with strict false-positive gates so
it inks genuine thin dark lines only, never whole dark masses or texture:

1. **Black-hat on L\*** (morphological closing − L*): responds only to dark
   structures *narrower* than the structuring element, so wide dark shapes
   (hair masses, filled pupils) produce no response -- the primary width gate.
2. **Contrast + darkness thresholds**: keep pixels whose black-hat response
   clears ``contrast_l`` and whose L* is below ``darkness_l`` (reject light
   thin texture).
3. **Lost-by-quantize gate**: drop candidates whose assigned palette color is
   already near-black (L* < ``survived_l``) -- those lines survived quantize
   as dark regions and are drawn by the normal region stroke; re-inking them
   would double-stroke. Only lines mapped to a lighter fill become ink.
4. **Width upper bound**: drop anything that survives an opening with a disc
   wider than ``max_width_mm`` (belongs to a thick blob, not a line).
5. **Length gate**: drop connected components shorter than ``min_length_mm``
   (speckle / compression texture).

Disabled (``enabled=False``, the default in every preset except dense/partial)
emits an all-False mask, a no-op that keeps the downstream stage valid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from skimage.measure import label as cc_label
from skimage.morphology import closing, disk, opening

from mysterycbn.foundation.color import DefaultColorScience
from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.ink import InkMask
from mysterycbn.model.records import LabelMap, Palette, Provenance

STAGE_NAME = "ink_detect"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64
_COLOR = DefaultColorScience()

MAX_WIDTH_MM_DEFAULT = 0.6
CONTRAST_L_DEFAULT = 8.0
DARKNESS_L_DEFAULT = 55.0
SURVIVED_L_DEFAULT = 25.0
MIN_LENGTH_MM_DEFAULT = 2.0


def _ppmm(work_scale: float) -> float:
    """Working px per printed mm (pt/mm ÷ pt/px), like denoise.speck_threshold."""
    if work_scale <= 0.0:
        return 0.0
    return 1.0 / (work_scale * MM_PER_INCH / PT_PER_INCH)


def detect_ink_mask(
    pixels: np.ndarray,
    label_map: np.ndarray,
    palette_l: np.ndarray,
    *,
    ppmm: float,
    max_width_mm: float,
    contrast_l: float,
    darkness_l: float,
    survived_l: float,
    min_length_mm: float,
) -> np.ndarray:
    """Boolean ink mask (see module docstring). ``palette_l`` is the L* of each
    palette color, indexed by label. ``ppmm`` ≤ 0 disables the mm-scaled gates
    and returns an all-False mask (nothing detectable without a physical scale)."""
    if ppmm <= 0.0:
        return np.zeros(label_map.shape, dtype=bool)
    lab = _COLOR.srgb_to_lab(np.array(pixels, dtype=np.float32))
    # skimage morphology mutates its input buffer, so hand it a fresh writable
    # copy -- raster pixels and derived views are read-only frozen artifacts,
    # and ascontiguousarray would alias (not copy) an already-contiguous one.
    lightness = np.array(lab[:, :, 0], dtype=np.float64)

    r_line = max(1, int(math.ceil(max_width_mm / 2.0 * ppmm)))
    se_line = disk(r_line)
    blackhat = closing(lightness, se_line) - lightness

    candidate = (blackhat > contrast_l) & (lightness < darkness_l)

    # Lost-by-quantize gate: keep only lines whose quantized fill is NOT already
    # near-black (those are drawn by the region stroke already).
    assigned_l = palette_l[label_map]
    candidate &= assigned_l >= survived_l

    # Width upper bound: an opening with a slightly larger disc keeps only wide
    # blobs; subtract them so only genuinely thin structures remain.
    r_wide = max(1, int(math.ceil(max_width_mm * ppmm)))
    wide = opening(np.array(candidate), disk(r_wide))
    candidate = candidate & ~wide

    # Length gate: drop connected components smaller than the length floor
    # (speckle / compression texture). Manual size filter over the CC labels.
    min_px = max(1, int(round(min_length_mm * ppmm)))
    labeled = cc_label(candidate, connectivity=2)
    if labeled.max() == 0:
        return np.zeros(label_map.shape, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_px
    keep[0] = False  # background
    return np.asarray(keep[labeled], dtype=bool)


class InkDetectStage:
    """Stage wrapper: (``raster_working``, ``label_map``, ``palette``) → ``ink_mask``."""

    def __init__(
        self,
        section: Mapping[str, object],
        *,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        enabled = section.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"ink config: enabled must be a bool, got {enabled!r}")

        def _num(key: str, default: float) -> float:
            v = section.get(key, default)
            if not isinstance(v, (int, float)) or float(v) < 0.0:
                raise ConfigError(f"ink config: {key} must be a number ≥ 0, got {v!r}")
            return float(v)

        self._enabled = enabled
        self._max_width_mm = _num("max_width_mm", MAX_WIDTH_MM_DEFAULT)
        self._contrast_l = _num("contrast_l", CONTRAST_L_DEFAULT)
        self._darkness_l = _num("darkness_l", DARKNESS_L_DEFAULT)
        self._survived_l = _num("survived_l", SURVIVED_L_DEFAULT)
        self._min_length_mm = _num("min_length_mm", MIN_LENGTH_MM_DEFAULT)
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("raster_working", "label_map", "palette")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("ink_mask",)

    @property
    def config_section(self) -> str:
        return "ink"

    def run(self, ctx: PipelineContext) -> None:
        label_map = ctx.get("label_map")
        palette = ctx.get("palette")
        raster = ctx.get("raster_working")
        if not isinstance(label_map, LabelMap) or not isinstance(palette, Palette):
            raise ConfigError("ink_detect requires LabelMap + Palette artifacts")
        provenance = Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=self._config_hash,
            source_hash=label_map.provenance.source_hash,
        )
        if not self._enabled:
            mask = np.zeros(label_map.labels.shape, dtype=bool)
            ctx.put("ink_mask", InkMask(mask=mask, provenance=provenance))
            return
        palette_l = np.array([c.lab[0] for c in palette.colors], dtype=np.float64)
        mask = detect_ink_mask(
            raster.pixels,
            label_map.labels,
            palette_l,
            ppmm=_ppmm(getattr(raster, "work_scale", 0.0)),
            max_width_mm=self._max_width_mm,
            contrast_l=self._contrast_l,
            darkness_l=self._darkness_l,
            survived_l=self._survived_l,
            min_length_mm=self._min_length_mm,
        )
        ctx.put("ink_mask", InkMask(mask=mask, provenance=provenance))
