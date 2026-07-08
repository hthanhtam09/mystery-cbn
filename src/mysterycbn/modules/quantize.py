"""Perceptual color quantization.

Clustering runs in CIELAB (with optional chroma weighting) so cluster
distances approximate perceived color difference. k-means is seeded for
determinism (invariant I2). After clustering, palette entries closer than a
ΔE threshold are merged so the printed legend never contains two colors a
reader cannot tell apart.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.config import QuantizeConfig
from ..core.errors import StageError
from ..core.pipeline import FunctionStage
from ..core.types import Palette, PaletteColor, PipelineContext


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """float32 sRGB [0,1] → CIELAB (L in [0,100], a/b roughly [-128,127])."""
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2Lab)


def lab_to_rgb255(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _kmeans_lab(samples: np.ndarray, k: int, cfg: QuantizeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Seeded k-means over N×3 weighted-LAB samples → (centers k×3, labels N)."""
    cv2.setRNGSeed(cfg.seed)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        cfg.max_iter,
        0.25,
    )
    _, labels, centers = cv2.kmeans(
        samples,
        k,
        None,  # type: ignore[arg-type]
        criteria,
        cfg.attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers, labels.ravel()


def _merge_close_centers(
    centers: np.ndarray, labels: np.ndarray, min_delta_e: float
) -> tuple[np.ndarray, np.ndarray]:
    """Union palette entries whose ΔE76 distance is below threshold.

    Merged center = area-weighted mean of its members, so dominant colors
    stay put and rare near-duplicates fold into them.
    """
    counts = np.bincount(labels, minlength=len(centers)).astype(np.float64)
    order = np.argsort(-counts)  # dominant first: rare colors merge into common
    remap = np.arange(len(centers))
    kept: list[int] = []
    for idx in order:
        target = -1
        for k_idx in kept:
            if np.linalg.norm(centers[idx] - centers[k_idx]) < min_delta_e:
                target = k_idx
                break
        if target == -1:
            kept.append(int(idx))
        else:
            merged_count = counts[target] + counts[idx]
            centers[target] = (
                centers[target] * counts[target] + centers[idx] * counts[idx]
            ) / merged_count
            counts[target] = merged_count
            remap[idx] = target
    # Resolve chains (a→b where b itself merged) — single pass suffices since
    # targets are always kept representatives.
    remap = remap[remap]
    new_index = {old: new for new, old in enumerate(kept)}
    final_map = np.array([new_index[int(remap[i])] for i in range(len(centers))])
    return centers[kept], final_map[labels]


def quantize(image: np.ndarray, cfg: QuantizeConfig) -> tuple[np.ndarray, Palette]:
    """Quantize an H×W×3 float32 sRGB image.

    Returns ``(label_map H×W int32, palette)`` where label values index the
    palette and palette numbers are assigned by descending pixel coverage
    (color 1 is the most-used color, matching book convention).
    """
    h, w = image.shape[:2]
    lab = rgb_to_lab(image)
    samples = lab.reshape(-1, 3).astype(np.float32)
    if cfg.chroma_weight != 1.0:
        samples = samples.copy()
        samples[:, 1:] *= cfg.chroma_weight

    centers, labels = _kmeans_lab(samples, cfg.n_colors, cfg)
    if cfg.chroma_weight != 1.0:
        centers = centers.copy()
        centers[:, 1:] /= cfg.chroma_weight
    if cfg.min_delta_e > 0:
        centers, labels = _merge_close_centers(centers, labels, cfg.min_delta_e)
    if len(centers) < 2:
        raise StageError("quantize", "image collapsed to a single color")

    # Renumber by coverage: most-common color gets number 1.
    counts = np.bincount(labels, minlength=len(centers))
    coverage_order = np.argsort(-counts)
    rank = np.empty(len(centers), dtype=np.int32)
    rank[coverage_order] = np.arange(len(centers), dtype=np.int32)
    label_map = rank[labels].reshape(h, w).astype(np.int32)
    centers = centers[coverage_order]

    rgb255 = lab_to_rgb255(centers.reshape(1, -1, 3)).reshape(-1, 3)
    palette = Palette(
        colors=tuple(
            PaletteColor(
                number=i + 1,
                lab=tuple(float(v) for v in centers[i]),
                rgb=tuple(int(v) for v in rgb255[i]),
            )
            for i in range(len(centers))
        )
    )
    return label_map, palette


def make_quantize_stage() -> FunctionStage:
    def _run(ctx: PipelineContext) -> None:
        assert ctx.image is not None
        ctx.label_map, ctx.palette = quantize(ctx.image, ctx.config.quantize)

    return FunctionStage("quantize", _run, requires=("image",), provides=("label_map", "palette"))
