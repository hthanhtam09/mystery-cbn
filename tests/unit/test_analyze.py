"""Unit tests for the Color Analysis stage (ENGINE_SPEC §6)."""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import ImageStats, Provenance, RasterImage
from mysterycbn.stages.raster.analyze import (
    AnalyzeStage,
    AutoTuneProposal,
    compute_stats,
    propose_overrides,
)

PROV = Provenance("preprocess", "1.0.0", "0" * 64, "1" * 64)
RNG = np.random.default_rng(0)


def _raster(pixels: np.ndarray) -> RasterImage:
    return RasterImage(pixels.astype(np.float32), 0.4, 1.0, False, 1, PROV)


def _flat(value: float = 0.5) -> RasterImage:
    return _raster(np.full((64, 64, 3), value))


def test_flat_field_statistics_are_degenerate() -> None:
    stats = compute_stats(_flat())
    assert stats.colorfulness == pytest.approx(0.0)
    assert stats.edge_density == 0.0
    assert stats.contrast == pytest.approx(0.0, abs=1e-3)
    assert stats.saturation == pytest.approx(0.0, abs=0.2)  # gray ⇒ chroma ≈ 0
    assert stats.entropy_bits == pytest.approx(0.0)  # single occupied bin
    assert stats.luminance_histogram.sum() == pytest.approx(1.0)


def test_two_tone_image_has_one_bit_entropy() -> None:
    pixels = np.zeros((64, 64, 3), dtype=np.float32)
    pixels[:, 32:] = 1.0
    stats = compute_stats(_raster(pixels))
    assert stats.entropy_bits == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < stats.edge_density < 0.1  # one vertical edge column band
    assert stats.contrast > 40.0


def test_brightness_tracks_lightness() -> None:
    dark = compute_stats(_flat(0.1))
    bright = compute_stats(_flat(0.9))
    assert dark.brightness < 40.0 < 90.0 < bright.brightness


def test_saturation_separates_gray_from_color() -> None:
    red = np.zeros((64, 64, 3), dtype=np.float32)
    red[..., 0] = 1.0
    assert compute_stats(_raster(red)).saturation > 50.0
    assert compute_stats(_flat()).saturation < 1.0


def test_lab_moments_shape_and_consistency() -> None:
    stats = compute_stats(_raster(RNG.random((64, 64, 3))))
    assert stats.lab_mean[0] == pytest.approx(stats.brightness)
    assert stats.lab_std[0] == pytest.approx(stats.contrast)


# ---------------------------------------------------------------- proposals


def test_flat_image_proposes_k_floor_and_light_smoothing() -> None:
    proposal = propose_overrides(compute_stats(_flat()))
    assert proposal.fragment["quantize"]["n_colors"] == 8  # floor
    assert proposal.fragment["preprocess"]["smooth_passes"] == 1  # ρ < 0.05


def test_busy_image_proposes_more_colors_and_heavier_smoothing() -> None:
    noisy = np.clip(RNG.random((128, 128, 3)), 0, 1)
    stats = compute_stats(_raster(noisy))
    assert stats.edge_density > 0.25
    proposal = propose_overrides(stats)
    assert proposal.fragment["preprocess"]["smooth_passes"] == 3
    k = proposal.fragment["quantize"]["n_colors"]
    assert isinstance(k, int) and 8 <= k <= 30


def test_mid_edge_density_makes_no_smoothing_proposal() -> None:
    stats = compute_stats(_flat())
    object.__setattr__(stats, "edge_density", 0.15)  # inside the no-proposal band
    proposal = propose_overrides(stats)
    assert "preprocess" not in proposal.fragment


def test_k_star_rotation_and_mirror_invariance() -> None:
    pixels = RNG.random((96, 64, 3)).astype(np.float32)
    variants = [pixels, np.rot90(pixels), np.rot90(pixels, 2), pixels[:, ::-1], pixels[::-1]]
    proposals = {
        propose_overrides(compute_stats(_raster(np.ascontiguousarray(v)))).fragment["quantize"][
            "n_colors"
        ]
        for v in variants
    }
    assert len(proposals) == 1  # identical k* under the dihedral group


def test_proposal_fragment_is_frozen_and_bounds_checked() -> None:
    proposal = propose_overrides(compute_stats(_flat()))
    with pytest.raises(TypeError):
        proposal.fragment["quantize"]["n_colors"] = 64  # type: ignore[index]
    with pytest.raises(ConfigError, match="k bounds"):
        propose_overrides(compute_stats(_flat()), k_min=10, k_max=5)


def test_determinism() -> None:
    pixels = RNG.random((80, 80, 3)).astype(np.float32)
    a = compute_stats(_raster(pixels))
    b = compute_stats(_raster(pixels))
    assert a.to_dict() == b.to_dict()


# -------------------------------------------------------------------- stage


def test_stage_via_context_and_disabled_mode() -> None:
    ctx = InMemoryContext(seed=0)
    ctx.put("raster_working", _flat())
    AnalyzeStage({}).run(ctx)
    assert isinstance(ctx.get("image_stats"), ImageStats)
    proposal = ctx.get("auto_tune")
    assert isinstance(proposal, AutoTuneProposal) and "quantize" in proposal.fragment

    ctx2 = InMemoryContext(seed=0)
    ctx2.put("raster_working", _flat())
    AnalyzeStage({"enabled": False}).run(ctx2)
    disabled = ctx2.get("auto_tune")
    assert isinstance(disabled, AutoTuneProposal) and dict(disabled.fragment) == {}
    with pytest.raises(ConfigError, match="analyze config"):
        AnalyzeStage({"k_min": "eight"})
