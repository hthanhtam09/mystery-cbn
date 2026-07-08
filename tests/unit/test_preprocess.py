"""Unit tests for the Preprocessing stage (ENGINE_SPEC §5)."""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import Provenance, RasterImage
from mysterycbn.stages.raster.preprocess import (
    PreprocessStage,
    apply_clahe,
    preprocess_raster,
    resize_to_working,
    smooth_bilateral,
    smooth_guided,
)

PROV = Provenance("load", "1.0.0", "0" * 64, "1" * 64)
RNG = np.random.default_rng(0)


def _raster(pixels: np.ndarray) -> RasterImage:
    return RasterImage(pixels.astype(np.float32), 0.0, 1.0, False, 1, PROV)


def _checkerboard(h: int = 128, w: int = 128, cell: int = 16) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    board = ((y // cell + x // cell) % 2).astype(np.float32)
    return np.repeat(board[:, :, None], 3, axis=2)


# ------------------------------------------------------------------- resize


def test_resize_bounds_longest_side_and_reports_factor() -> None:
    pixels = RNG.random((200, 300, 3)).astype(np.float32)
    out, f = resize_to_working(pixels, 128)
    assert out.shape == (85, 128, 3)
    assert f == pytest.approx(128 / 300)


def test_resize_never_upscales() -> None:
    pixels = RNG.random((100, 100, 3)).astype(np.float32)
    out, f = resize_to_working(pixels, 1600)
    assert out is pixels
    assert f == 1.0


# --------------------------------------------------------------- smoothers


@pytest.mark.parametrize("smoother", ["bilateral", "guided"])
def test_flat_field_is_a_fixpoint(smoother: str) -> None:
    flat = np.full((64, 64, 3), 0.42, dtype=np.float32)
    if smoother == "bilateral":
        out = smooth_bilateral(flat, passes=2, sigma_color=0.08, sigma_space=5.0)
    else:
        out = smooth_guided(flat, passes=2, radius=8, eps=1e-3)
    np.testing.assert_allclose(out, flat, atol=1e-5)


@pytest.mark.parametrize("smoother", ["bilateral", "guided"])
def test_noise_reduced_on_noisy_flat_field(smoother: str) -> None:
    noisy = (0.5 + 0.05 * RNG.standard_normal((128, 128, 3))).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    if smoother == "bilateral":
        out = smooth_bilateral(noisy, passes=2, sigma_color=0.08, sigma_space=5.0)
    else:
        out = smooth_guided(noisy, passes=2, radius=8, eps=1e-3)
    assert float(out.std()) < 0.5 * float(noisy.std())  # ≥ 2× noise reduction


def test_bilateral_preserves_strong_edges() -> None:
    board = _checkerboard()
    out = smooth_bilateral(board, passes=2, sigma_color=0.08, sigma_space=5.0)
    # Edge amplitude at cell boundaries survives nearly intact.
    edge_in = float(np.abs(np.diff(board[:, :, 0], axis=1)).max())
    edge_out = float(np.abs(np.diff(out[:, :, 0], axis=1)).max())
    assert edge_out >= 0.9 * edge_in
    # Interiors stay flat (no ringing).
    assert float(out[8, 8, 0]) == pytest.approx(0.0, abs=0.02)


def test_guided_softens_edges_more_than_bilateral() -> None:
    """The documented trade-off (ENGINE_SPEC §5): guided is the fast preset,
    bilateral keeps strong edges sharper — this is WHY bilateral is default."""
    board = _checkerboard()
    bil = smooth_bilateral(board, passes=2, sigma_color=0.08, sigma_space=5.0)
    gui = smooth_guided(board, passes=2, radius=8, eps=1e-3)
    edge = lambda img: float(np.abs(np.diff(img[:, :, 0], axis=1)).max())  # noqa: E731
    assert edge(bil) > edge(gui)


# ------------------------------------------------------------------- CLAHE


def test_clahe_keeps_gray_gray_and_raises_contrast() -> None:
    y = np.linspace(0.45, 0.55, 128, dtype=np.float32)
    gray = np.repeat(np.repeat(y[:, None], 128, axis=1)[:, :, None], 3, axis=2)
    out = apply_clahe(gray, clip=2.0)
    np.testing.assert_allclose(out[..., 0], out[..., 1], atol=0.02)  # L-only op
    assert float(out.std()) > float(gray.std())  # contrast expanded


# ------------------------------------------------------------ full stage


def test_preprocess_sets_work_scale_and_provenance() -> None:
    src = _raster(RNG.random((400, 300, 3)))
    out = preprocess_raster(src, max_working_px=200, smooth_passes=1)
    assert out.pixels.shape[0] == 200  # portrait: height is longest
    # s = min(540/W, 720/H) with W=150, H=200.
    assert out.work_scale == pytest.approx(min(540.0 / 150, 720.0 / 200))
    assert out.resize_factor == pytest.approx(0.5)
    assert out.provenance.stage_name == "preprocess"
    assert out.provenance.source_hash == src.provenance.source_hash


def test_zero_passes_is_resize_only() -> None:
    src = _raster(RNG.random((128, 128, 3)))
    out = preprocess_raster(src, max_working_px=128, smooth_passes=0)
    np.testing.assert_array_equal(out.pixels, src.pixels)


def test_invalid_config_rejected() -> None:
    src = _raster(RNG.random((64, 64, 3)))
    with pytest.raises(ConfigError, match="impl"):
        preprocess_raster(src, impl="meanshift")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="passes"):
        preprocess_raster(src, smooth_passes=-1)


def test_stage_via_context() -> None:
    stage = PreprocessStage({"max_working_px": 96, "smooth_passes": 1})
    ctx = InMemoryContext(seed=0)
    ctx.put("raster_source", _raster(RNG.random((200, 200, 3))))
    stage.run(ctx)
    working = ctx.get("raster_working")
    assert isinstance(working, RasterImage)
    assert max(working.pixels.shape[:2]) == 96
    assert working.work_scale > 0.0
