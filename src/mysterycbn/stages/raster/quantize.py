"""Quantization stage: raster → K perceptual palette colors + label map
(ENGINE_SPEC.md §7; implementation-ready design in docs/modules/quantize.md).

Algorithm comparison (normative decision, ENGINE_SPEC §7 + design doc §5):

============  =======================================================
LAB k-means   **DEFAULT** (``impl="labkmeans"``). Directly minimizes
              perceptual within-class variance, exact K control,
              trivially determinized (seeded k-means++, 4 restarts).
Median cut    Registered alternative (``impl="mediancut"``): 5–10×
              faster palette fit, weaker in smooth gradients (splits
              by extent, not variance). The fast-preset trade.
Octree        Rejected as default: favors dominant hues, K control is
              approximate (leaf folding); deferred (design doc §16.1).
Wu            Rejected: variance minimization is solid but operates
              in RGB histogram space (no perceptual metric) and has
              no natural seed-stream control for I2.
NeuQuant      Rejected: SOM training is order- and rate-sensitive —
              nondeterministic across implementations, and perceptual
              quality on flat-art fixtures is poor.
============  =======================================================

Both shipped implementations share the same assignment → ΔE00-merge →
exact-mean finalize path, so the QM-16 separation invariant holds for any
``impl`` by construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Literal

import numpy as np

from mysterycbn.foundation.color import DefaultColorScience
from mysterycbn.foundation.errors import ConfigError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance, RasterImage

STAGE_NAME = "quantize"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

_COLOR = DefaultColorScience()
_BLOCK = 65_536  # blocked-argmin row block (design doc §8)
_N_INIT = 4
_CONVERGENCE = 0.05

QuantizeImpl = Literal["labkmeans", "mediancut"]


def stage_seed(seed: int) -> int:
    """``SHA-256(seed ‖ stage_name)[:8]`` as uint64 (ENGINE_SPEC §1.3)."""
    digest = hashlib.sha256(f"{seed}\x1f{STAGE_NAME}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _stride_sample(flat_lab: np.ndarray, sample_px: int) -> np.ndarray:
    """Deterministic RNG-free stride sample of exactly ``sample_px`` rows
    (design doc §5.2); all rows when N ≤ sample_px."""
    n = flat_lab.shape[0]
    if n <= sample_px:
        return flat_lab
    stride = n // sample_px
    return flat_lab[: stride * sample_px : stride][:sample_px]


def _kmeanspp_init(x: np.ndarray, k: int, base: int, restart: int) -> np.ndarray:
    """k-means++ D² seeding on an explicit PCG64 stream (design doc §5.3)."""
    rng = np.random.Generator(np.random.PCG64(base + restart))
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = x[(base + restart) % len(x)]
    d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:  # all remaining points coincide with a center
            centers[i:] = centers[0]
            break
        idx = int(rng.choice(len(x), p=d2 / total))
        centers[i] = x[idx]
        d2 = np.minimum(d2, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def _pairwise_d2(x: np.ndarray, centers: np.ndarray, x2: np.ndarray | None = None) -> np.ndarray:
    """Squared ΔE76 distances via the matmul identity ‖x−c‖² = ‖x‖² + ‖c‖² − 2x·c
    (an order of magnitude faster than broadcast subtraction at S = 10⁵).

    ``x2`` (‖x‖² per row) is invariant across Lloyd iterations and restarts;
    callers that loop may pass a precomputed value to skip recomputing it."""
    if x2 is None:
        x2 = np.einsum("ij,ij->i", x, x)
    c2 = np.einsum("ij,ij->i", centers, centers)
    return np.asarray(np.maximum(x2[:, None] + c2[None, :] - 2.0 * (x @ centers.T), 0.0))


def _class_means(x: np.ndarray, assign: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-class means via bincount; empty classes yield zeros."""
    counts = np.bincount(assign, minlength=k).astype(np.float64)
    means = np.empty((k, 3), dtype=np.float64)
    for ch in range(3):
        sums = np.bincount(assign, weights=x[:, ch], minlength=k)
        means[:, ch] = np.divide(sums, counts, out=np.zeros(k), where=counts > 0)
    return means, counts


def _fix_empty(centers: np.ndarray, x: np.ndarray, assign: np.ndarray) -> bool:
    """Re-seed empty clusters at the farthest sample point (deterministic)."""
    counts = np.bincount(assign, minlength=len(centers))
    if np.all(counts > 0):  # common case: skip the O(N) distance pass entirely
        return False
    moved = False
    d2 = np.sum((x - centers[assign]) ** 2, axis=1)
    for c in range(len(centers)):
        if counts[c] == 0:
            centers[c] = x[int(np.argmax(d2))]
            d2[int(np.argmax(d2))] = 0.0
            moved = True
    return moved


def _lloyd(x: np.ndarray, centers: np.ndarray, max_iter: int) -> tuple[np.ndarray, float]:
    """Lloyd iterations with ΔE76 assignment (sanctioned inner loop)."""
    k = len(centers)
    x2 = np.einsum("ij,ij->i", x, x)  # invariant across iterations; hoisted out of the loop
    for _ in range(max_iter):
        assign = np.argmin(_pairwise_d2(x, centers, x2), axis=1)
        new, counts = _class_means(x, assign, k)
        new[counts == 0] = centers[counts == 0]  # placeholder before re-seed
        moved = _fix_empty(new, x, assign)
        shift = float(np.linalg.norm(new - centers, axis=1).max())
        centers = new
        if shift < _CONVERGENCE and not moved:
            break
    inertia = float(_pairwise_d2(x, centers, x2).min(axis=1).sum())
    return centers, inertia


def _fit_labkmeans(x: np.ndarray, k: int, seed_base: int, max_iter: int) -> np.ndarray:
    """Default fitter: 4-restart seeded LAB k-means; lowest inertia wins."""
    best: np.ndarray | None = None
    best_inertia = np.inf
    for restart in range(_N_INIT):
        centers, inertia = _lloyd(x, _kmeanspp_init(x, k, seed_base, restart), max_iter)
        if inertia < best_inertia:
            best, best_inertia = centers, inertia
    assert best is not None
    return best


def _fit_mediancut(x: np.ndarray, k: int) -> np.ndarray:
    """Alternative fitter: deterministic median cut in LAB.

    Repeatedly splits the box with the largest axis extent at its median
    along that axis; centers are box means. RNG-free.
    """
    boxes: list[np.ndarray] = [x]
    while len(boxes) < k:
        extents = [float(np.ptp(b, axis=0).max()) if len(b) > 1 else -1.0 for b in boxes]
        widest = int(np.argmax(extents))
        if extents[widest] <= 0.0:
            break  # nothing splittable left (fewer distinct colors than k)
        box = boxes.pop(widest)
        axis = int(np.argmax(np.ptp(box, axis=0)))
        order = np.argsort(box[:, axis], kind="stable")
        half = len(order) // 2
        boxes.insert(widest, box[order[:half]])
        boxes.insert(widest + 1, box[order[half:]])
    return np.array([b.mean(axis=0) for b in boxes], dtype=np.float64)


def _assign_blocked(flat_lab: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Full-raster nearest-center assignment in 64 K-pixel blocks (§8)."""
    out = np.empty(flat_lab.shape[0], dtype=np.int32)
    c = centers.astype(np.float32)
    for start in range(0, flat_lab.shape[0], _BLOCK):
        block = flat_lab[start : start + _BLOCK]
        out[start : start + _BLOCK] = np.argmin(_pairwise_d2(block, c), axis=1)
    return out


def _merge_close(
    centers: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge argmin-ΔE00 center pairs until separation ≥ threshold or K = 2
    (design doc §5.7; ties resolve lexicographically via row-major argmin)."""
    centers = centers.copy()
    while len(centers) > 2:
        table = _COLOR.delta_e_2000(centers[:, None, :], centers[None, :, :])
        np.fill_diagonal(table, np.inf)
        flat_idx = int(np.argmin(table))
        i, j = divmod(flat_idx, len(centers))
        if i > j:
            i, j = j, i
        if float(table[i, j]) >= threshold:
            break
        counts = np.bincount(labels, minlength=len(centers)).astype(np.float64)
        weight = counts[i] + counts[j]
        centers[i] = (
            (centers[i] * counts[i] + centers[j] * counts[j]) / weight if weight > 0 else centers[i]
        )
        centers = np.delete(centers, j, axis=0)
        remap = np.arange(len(centers) + 1)
        remap[j] = i
        remap[j + 1 :] = np.arange(j, len(centers))
        labels = remap[labels].astype(np.int32)
    return centers, labels


def _finalize(
    lab_flat: np.ndarray, labels: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact per-class means, then coverage-descending renumber
    (ties: LAB-lexicographic — design doc §5.8)."""
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    means = np.empty((k, 3), dtype=np.float64)
    for ch in range(3):
        sums = np.bincount(labels, weights=lab_flat[:, ch].astype(np.float64), minlength=k)
        means[:, ch] = np.divide(sums, counts, out=np.zeros(k), where=counts > 0)
    order = sorted(range(k), key=lambda c: (-counts[c], means[c, 0], means[c, 1], means[c, 2]))
    rank = np.empty(k, dtype=np.int32)
    rank[np.array(order)] = np.arange(k, dtype=np.int32)
    return means[np.array(order)], counts[np.array(order)], rank[labels]


def quantize_raster(
    raster: RasterImage,
    *,
    n_colors: int = 16,
    merge_delta_e: float = 7.0,
    sample_px: int = 100_000,
    max_iter: int = 50,
    impl: QuantizeImpl = "labkmeans",
    seed: int = 0,
    config_hash: str = _UNSET_HASH,
) -> tuple[LabelMap, Palette]:
    """Quantize the working raster (full algorithm, design doc §5–§6)."""
    if not 2 <= n_colors <= 64:
        raise ConfigError(f"n_colors must be in [2, 64], got {n_colors}")
    h, w = raster.pixels.shape[:2]
    lab = _COLOR.srgb_to_lab(raster.pixels).astype(np.float32)
    flat = lab.reshape(-1, 3)
    effective_sample = max(sample_px, 100 * n_colors)
    x = _stride_sample(flat, effective_sample).astype(np.float64)

    if impl == "labkmeans":
        centers = _fit_labkmeans(x, n_colors, stage_seed(seed), max_iter)
    elif impl == "mediancut":
        centers = _fit_mediancut(x, n_colors)
    else:
        raise ConfigError(f"unknown quantizer impl {impl!r} (labkmeans | mediancut)")

    labels = _assign_blocked(flat, centers)
    centers, labels = _merge_close(centers, labels, merge_delta_e)
    if (
        len(centers) < 2 or float(np.ptp(centers, axis=0).max()) == 0.0
    ):  # flat-input clamp (design doc §9)
        base = centers[0]
        offset = np.array([0.5, 0.0, 0.0])
        centers = np.array([base + offset, base - offset])
        labels = np.zeros_like(labels)
        labels[flat[:, 0] < base[0]] = 1

    means, counts, labels = _finalize(flat, labels, len(centers))
    table = _COLOR.delta_e_2000(means[:, None, :], means[None, :, :])
    np.fill_diagonal(table, np.inf)
    guaranteed = min(merge_delta_e, float(table.min()))

    provenance = Provenance(
        stage_name=STAGE_NAME,
        stage_version=STAGE_VERSION,
        config_hash=config_hash,
        source_hash=raster.provenance.source_hash,
    )
    palette = Palette(
        colors=tuple(
            PaletteColor.from_lab(i, (float(m[0]), float(m[1]), float(m[2])), int(counts[i]))
            for i, m in enumerate(means)
        ),
        provenance=provenance,
        min_delta_e=guaranteed,
    )
    label_map = LabelMap(labels=labels.reshape(h, w), provenance=provenance)
    label_map.validate_against(palette)
    return label_map, palette


class QuantizeStage:
    """Stage wrapper: ``raster_working`` → ``label_map`` + ``palette``."""

    def __init__(
        self, section: Mapping[str, object], *, seed: int = 0, config_hash: str = _UNSET_HASH
    ) -> None:
        self._section = dict(section)
        self._seed = seed
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
        return ("label_map", "palette")

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        raster = ctx.get("raster_working")
        if not isinstance(raster, RasterImage):
            raise ConfigError(f"artifact 'raster_working' has wrong type {type(raster).__name__}")
        section = self._section
        try:
            label_map, palette = quantize_raster(
                raster,
                n_colors=int(section.get("n_colors", 16)),  # type: ignore[call-overload]
                merge_delta_e=float(section.get("merge_delta_e", 7.0)),  # type: ignore[arg-type]
                sample_px=int(section.get("sample_px", 100_000)),  # type: ignore[call-overload]
                max_iter=int(section.get("max_iter", 50)),  # type: ignore[call-overload]
                impl=section.get("impl", "labkmeans"),  # type: ignore[arg-type]
                seed=ctx.seed,
                config_hash=self._config_hash,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid quantize config: {exc}") from exc
        ctx.put("label_map", label_map)
        ctx.put("palette", palette)
