"""Preprocessing: resize to working resolution + edge-preserving smoothing.

Bilateral filtering flattens texture and gradients into paintable plateaus
while keeping object boundaries crisp — exactly the structure the quantizer
needs to produce clean, book-quality regions instead of speckle.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.config import PreprocessConfig
from ..core.pipeline import FunctionStage
from ..core.types import PipelineContext


def resize_to_working(image: np.ndarray, max_px: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is ``max_px``. Never upscales.

    Returns the resized image and ``scale`` = original px per working px.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_px:
        return image, 1.0
    factor = max_px / longest
    resized = cv2.resize(
        image,
        (max(1, round(w * factor)), max(1, round(h * factor))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, 1.0 / factor


def smooth_edge_preserving(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Iterated bilateral filter on float32 RGB in [0,1]."""
    out = image
    for _ in range(cfg.smooth_passes):
        out = cv2.bilateralFilter(
            out,
            d=0,  # derive neighborhood from sigma_space
            sigmaColor=cfg.bilateral_sigma_color,
            sigmaSpace=cfg.bilateral_sigma_space,
        )
    return np.clip(out, 0.0, 1.0)


def apply_clahe(image: np.ndarray, clip: float) -> np.ndarray:
    """CLAHE on the L channel only, preserving chroma."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l_scaled = (lab[..., 0] / 100.0 * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(l_scaled).astype(np.float32) / 255.0 * 100.0
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0)


def preprocess(image: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, float]:
    working, scale = resize_to_working(image, cfg.max_working_px)
    if cfg.clahe:
        working = apply_clahe(working, cfg.clahe_clip)
    working = smooth_edge_preserving(working, cfg)
    return working, scale


def make_preprocess_stage() -> FunctionStage:
    def _run(ctx: PipelineContext) -> None:
        assert ctx.image is not None
        ctx.image, ctx.work_scale = preprocess(ctx.image, ctx.config.preprocess)

    return FunctionStage("preprocess", _run, requires=("image",), provides=("image",))
