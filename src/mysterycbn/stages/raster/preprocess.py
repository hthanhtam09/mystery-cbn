"""Preprocessing stage: working resolution + edge-preserving smoothing
(ENGINE_SPEC.md §5).

Algorithm choice (normative, ENGINE_SPEC §5): the default smoother is the
**iterated bilateral filter** — strongest color flattening per millisecond
among deterministic options, creates no colors outside local mixtures, and
its two parameters map directly to user-meaningful knobs. The **guided
filter** ships as the registered alternative (``impl = "guided"``): ~2–4×
faster at working resolution with slightly weaker flattening at strong edges
— the right trade for a fast preset, the wrong default for print quality.
Mean-shift (10–30× slower) and ML denoisers (nondeterministic across
hardware) were rejected; see ENGINE_SPEC §5 "Algorithm alternatives".

Optional CLAHE operates on the L channel only, after smoothing.
OpenCV is an implementation detail and never appears in the interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import cv2
import numpy as np

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Provenance, RasterImage

STAGE_NAME = "preprocess"
STAGE_VERSION = "1.0.0"

# US Letter content box at default margins, in points (MATH_SPEC §1.3).
_DEFAULT_CONTENT_PT = (540.0, 720.0)
_UNSET_HASH = "0" * 64

SmootherImpl = Literal["bilateral", "guided"]


def resize_to_working(pixels: np.ndarray, max_working_px: int) -> tuple[np.ndarray, float]:
    """Area-average downscale so max(H, W) ≤ ``max_working_px``; never upscales.

    Returns ``(working pixels, resize factor f ≤ 1)``.
    """
    h, w = pixels.shape[:2]
    longest = max(h, w)
    if longest <= max_working_px:
        return pixels, 1.0
    f = max_working_px / longest
    new_w = max(1, round(w * f))
    new_h = max(1, round(h * f))
    out = cv2.resize(pixels, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return np.asarray(out, dtype=np.float32), f


def smooth_bilateral(
    pixels: np.ndarray, *, passes: int, sigma_color: float, sigma_space: float
) -> np.ndarray:
    """Iterated bilateral filter on float32 sRGB (default smoother).

    ``sigma_color`` is in [0, 1] RGB units; kernel diameter is derived from
    ``sigma_space`` (radius ⌈2σ⌉, ENGINE_SPEC §5.2).
    """
    out = np.asarray(pixels, dtype=np.float32)
    diameter = 2 * math.ceil(2.0 * sigma_space) + 1
    for _ in range(passes):
        out = cv2.bilateralFilter(out, diameter, sigma_color, sigma_space)
    return np.asarray(out, dtype=np.float32)


def smooth_guided(pixels: np.ndarray, *, passes: int, radius: int, eps: float) -> np.ndarray:
    """Self-guided filter (He et al.), gray guide, per-channel — the fast
    alternative smoother. ``eps`` is in squared [0, 1] intensity units."""
    out = np.asarray(pixels, dtype=np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(img: np.ndarray) -> np.ndarray:
        return np.asarray(cv2.blur(img, ksize, borderType=cv2.BORDER_REFLECT))

    for _ in range(passes):
        guide = out.mean(axis=2)
        mean_i = box(guide)
        var_i = box(guide * guide) - mean_i * mean_i
        channels = []
        for c in range(3):
            p = out[:, :, c]
            mean_p = box(p)
            cov_ip = box(guide * p) - mean_i * mean_p
            a = cov_ip / (var_i + eps)
            b = mean_p - a * mean_i
            channels.append(box(a) * guide + box(b))
        out = np.clip(np.stack(channels, axis=2), 0.0, 1.0).astype(np.float32)
    return out


def apply_clahe(pixels: np.ndarray, clip: float) -> np.ndarray:
    """CLAHE on the L channel of LAB only (8×8 tiles, ENGINE_SPEC §5.3)."""
    lab = cv2.cvtColor((pixels * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2Lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def preprocess_raster(
    raster: RasterImage,
    *,
    max_working_px: int = 1600,
    smooth_passes: int = 2,
    impl: SmootherImpl = "bilateral",
    bilateral_sigma_color: float = 0.08,
    bilateral_sigma_space: float = 5.0,
    guided_radius: int = 8,
    guided_eps: float = 1e-3,
    clahe: bool = False,
    clahe_clip: float = 2.0,
    content_size_pt: tuple[float, float] = _DEFAULT_CONTENT_PT,
    config_hash: str = _UNSET_HASH,
) -> RasterImage:
    """Produce the working raster: bounded resolution, flattened colors.

    ``work_scale`` (pt/px) is set here — the aspect-preserving letterbox
    scale ``s = min(C_w/W, C_h/H)`` of MATH_SPEC §1.3 — and applied to
    geometry exactly once, later, by the Arc Graph stage.
    """
    if smooth_passes < 0:
        raise ConfigError(f"smooth_passes must be ≥ 0, got {smooth_passes}")
    working, f = resize_to_working(raster.pixels, max_working_px)
    if smooth_passes > 0:
        if impl == "bilateral":
            working = smooth_bilateral(
                working,
                passes=smooth_passes,
                sigma_color=bilateral_sigma_color,
                sigma_space=bilateral_sigma_space,
            )
        elif impl == "guided":
            working = smooth_guided(
                working, passes=smooth_passes, radius=guided_radius, eps=guided_eps
            )
        else:
            raise ConfigError(f"unknown smoother impl {impl!r} (bilateral | guided)")
    if clahe:
        working = apply_clahe(working, clahe_clip)

    h, w = working.shape[:2]
    work_scale = min(content_size_pt[0] / w, content_size_pt[1] / h)
    return RasterImage(
        pixels=np.clip(working, 0.0, 1.0),
        work_scale=work_scale,
        resize_factor=raster.resize_factor * f,
        icc_applied=raster.icc_applied,
        exif_orientation=raster.exif_orientation,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=raster.provenance.source_hash,
        ),
    )


class PreprocessStage:
    """Stage wrapper: ``raster_source`` → ``raster_working``."""

    def __init__(self, section: Mapping[str, object], config_hash: str = _UNSET_HASH) -> None:
        self._section = dict(section)
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("raster_source",)

    @property
    def provides(self) -> tuple[str, ...]:
        return ("raster_working",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        raster = ctx.get("raster_source")
        if not isinstance(raster, RasterImage):
            raise ConfigError(f"artifact 'raster_source' has wrong type {type(raster).__name__}")
        section = self._section
        try:
            result = preprocess_raster(
                raster,
                max_working_px=int(section.get("max_working_px", 1600)),  # type: ignore[call-overload]
                smooth_passes=int(section.get("smooth_passes", 2)),  # type: ignore[call-overload]
                impl=section.get("impl", "bilateral"),  # type: ignore[arg-type]
                bilateral_sigma_color=float(section.get("bilateral_sigma_color", 0.08)),  # type: ignore[arg-type]
                bilateral_sigma_space=float(section.get("bilateral_sigma_space", 5.0)),  # type: ignore[arg-type]
                clahe=bool(section.get("clahe", False)),
                clahe_clip=float(section.get("clahe_clip", 2.0)),  # type: ignore[arg-type]
                config_hash=self._config_hash,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid preprocess config: {exc}") from exc
        ctx.put("raster_working", result)
