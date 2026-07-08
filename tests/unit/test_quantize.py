"""Unit tests for the Quantization stage (ENGINE_SPEC §7, docs/modules/quantize.md §11)."""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.color import DefaultColorScience
from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import LabelMap, Palette, Provenance, RasterImage
from mysterycbn.stages.raster.quantize import QuantizeStage, quantize_raster, stage_seed

PROV = Provenance("preprocess", "1.0.0", "0" * 64, "1" * 64)
CS = DefaultColorScience()
RNG = np.random.default_rng(0)


def _raster(pixels: np.ndarray) -> RasterImage:
    return RasterImage(pixels.astype(np.float32), 0.4, 1.0, False, 1, PROV)


def _four_color() -> RasterImage:
    px = np.zeros((64, 64, 3), dtype=np.float32)
    px[:32, :32] = [1, 0, 0]
    px[:32, 32:] = [0, 1, 0]
    px[32:, :32] = [0, 0, 1]
    px[32:, 32:] = [1, 1, 0]
    return _raster(px)


def _gradient() -> RasterImage:
    h, w = 96, 128
    y, x = np.mgrid[0:h, 0:w]
    px = np.stack([x / (w - 1), y / (h - 1), 0.5 * np.ones((h, w))], axis=2)
    return _raster(px)


# ---------------------------------------------------------- exact recovery


@pytest.mark.parametrize("impl", ["labkmeans", "mediancut"])
def test_four_color_exact_recovery(impl: str) -> None:
    lm, pal = quantize_raster(_four_color(), n_colors=4, impl=impl)  # type: ignore[arg-type]
    assert pal.size == 4
    assert [c.coverage_px for c in pal.colors] == [1024] * 4
    # Every pixel's assigned palette color is within 0.5 ΔE00 of its true color.
    true_lab = CS.srgb_to_lab(_four_color().pixels)
    pal_lab = np.array([c.lab for c in pal.colors])
    err = CS.delta_e_2000(true_lab, pal_lab[lm.labels])
    assert float(err.max()) < 0.5


def test_determinism_and_seed_sensitivity() -> None:
    a, _ = quantize_raster(_gradient(), n_colors=8, seed=0)
    b, _ = quantize_raster(_gradient(), n_colors=8, seed=0)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert stage_seed(0) != stage_seed(1)  # seed streams are isolated


# ------------------------------------------------------------------- merge


def test_merge_collapses_near_duplicates() -> None:
    px = np.zeros((64, 64, 3), dtype=np.float32)
    px[:, :32] = 0.50
    px[:, 32:] = 0.52  # ~1.5 ΔE00 from the left half
    lm, pal = quantize_raster(_raster(px), n_colors=8, merge_delta_e=7.0)
    assert pal.size == 2  # merged to the K=2 floor
    assert int(lm.labels.max()) <= 1


def test_separation_invariant_recorded() -> None:
    _, pal = quantize_raster(_gradient(), n_colors=8, merge_delta_e=7.0)
    off = pal.delta_e_table[~np.eye(pal.size, dtype=bool)]
    assert float(off.min()) >= pal.min_delta_e > 0.0


# --------------------------------------------------------------- ordering


def test_coverage_ordering_descending() -> None:
    px = np.zeros((64, 64, 3), dtype=np.float32)
    px[:, :48] = [1.0, 1.0, 1.0]  # 75 % white
    px[:, 48:] = [0.8, 0.1, 0.1]  # 25 % red
    lm, pal = quantize_raster(_raster(px), n_colors=2)
    coverages = [c.coverage_px for c in pal.colors]
    assert coverages == sorted(coverages, reverse=True)
    assert pal.colors[0].lab[0] > 90.0  # label 0 = dominant white
    assert int(lm.labels[0, 0]) == 0


# -------------------------------------------------------------- edge cases


def test_flat_input_clamps_to_two_colors() -> None:
    _, pal = quantize_raster(_raster(np.full((64, 64, 3), 0.5)), n_colors=8)
    assert pal.size == 2  # design doc §9 clamp


def test_fewer_distinct_colors_than_k() -> None:
    lm, pal = quantize_raster(_four_color(), n_colors=12)
    assert pal.size == 4  # duplicates collapsed in merge
    lm.validate_against(pal)


def test_sampling_boundaries() -> None:
    # N < sample_px (all pixels) and N > sample_px (stride path) both work.
    small, _ = quantize_raster(_gradient(), n_colors=4, sample_px=1_000_000)
    big, _ = quantize_raster(_gradient(), n_colors=4, sample_px=1_000)
    assert small.labels.shape == big.labels.shape


def test_invalid_config_rejected() -> None:
    with pytest.raises(ConfigError, match="n_colors"):
        quantize_raster(_gradient(), n_colors=1)
    with pytest.raises(ConfigError, match="impl"):
        quantize_raster(_gradient(), impl="neuquant")  # type: ignore[arg-type]


# ---------------------------------------------------- artifacts & contract


@pytest.mark.parametrize("impl", ["labkmeans", "mediancut"])
def test_artifact_contract(impl: str) -> None:
    """Shared quantizer contract (design doc §14): dense labels, valid
    palette, provenance, determinism — for every registered implementation."""
    lm, pal = quantize_raster(_gradient(), n_colors=6, impl=impl)  # type: ignore[arg-type]
    assert isinstance(lm, LabelMap) and isinstance(pal, Palette)
    assert set(np.unique(lm.labels)) == set(range(pal.size))  # dense
    assert sum(c.coverage_px for c in pal.colors) == lm.labels.size
    assert lm.provenance.stage_name == "quantize"
    lm2, _ = quantize_raster(_gradient(), n_colors=6, impl=impl)  # type: ignore[arg-type]
    np.testing.assert_array_equal(lm.labels, lm2.labels)


def test_stage_via_context() -> None:
    ctx = InMemoryContext(seed=3)
    ctx.put("raster_working", _gradient())
    QuantizeStage({"n_colors": 6}).run(ctx)
    pal = ctx.get("palette")
    lm = ctx.get("label_map")
    assert isinstance(pal, Palette) and isinstance(lm, LabelMap)
    with pytest.raises(ConfigError, match="invalid quantize config"):
        QuantizeStage({"n_colors": "many"}).run(ctx)
