"""Unit tests for the Noise Removal stage (ENGINE_SPEC §8)."""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import (
    LabelMap,
    Palette,
    PaletteColor,
    Provenance,
    RasterImage,
)
from mysterycbn.stages.raster.denoise import (
    DenoiseStage,
    area_opening,
    denoise_label_map,
    modal_filter,
    speck_threshold,
)

PROV = Provenance("quantize", "1.0.0", "0" * 64, "1" * 64)


def _palette(labs: list[tuple[float, float, float]]) -> Palette:
    colors = tuple(PaletteColor.from_lab(i, lab, 100) for i, lab in enumerate(labs))
    return Palette(colors=colors, provenance=PROV)


# Well-separated 3-color palette: 0 dark, 1 mid, 2 light.
PAL3 = _palette([(20.0, 0.0, 0.0), (55.0, 0.0, 0.0), (90.0, 0.0, 0.0)])


def _label_map(labels: np.ndarray) -> LabelMap:
    return LabelMap(labels=labels.astype(np.int32), provenance=PROV)


# ------------------------------------------------------------- modal filter


def test_modal_removes_isolated_pixel() -> None:
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[3, 3] = 1
    out = modal_filter(labels, PAL3)
    assert int(out[3, 3]) == 0
    assert np.all(out == 0)


def test_modal_preserves_solid_edge() -> None:
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[:, 4:] = 1
    out = modal_filter(labels, PAL3)
    np.testing.assert_array_equal(out, labels)  # a straight edge is a fixpoint


def test_modal_tie_breaks_by_delta_e_to_current_color() -> None:
    # Center pixel 1 (mid): 4 dark + 4 light neighbors tie at count 4;
    # candidate scores: dark/light tie on count, but *including itself* the
    # center's own label has count 1 — majority is a 4:4 tie between 0 and 2.
    # Palette: mid (55) is closer to light (90 → ΔE ~26) than… make it closer
    # to dark instead by using an asymmetric palette.
    pal = _palette([(50.0, 0.0, 0.0), (55.0, 0.0, 0.0), (95.0, 0.0, 0.0)])
    labels = np.array(
        [
            [0, 2, 0],
            [2, 1, 2],
            [0, 2, 0],
        ],
        dtype=np.int32,
    )
    out = modal_filter(labels, pal, max_iters=1)
    # counts around center: 0×4, 2×4, 1×1 → tie between 0 and 2; current color
    # is 1 (L=55): ΔE(1,0) ≪ ΔE(1,2) ⇒ label 0 wins.
    assert int(out[1, 1]) == 0


def test_modal_fixpoint_termination_and_determinism() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, (32, 32)).astype(np.int32)
    a = modal_filter(labels, PAL3, max_iters=10)
    b = modal_filter(labels, PAL3, max_iters=10)
    np.testing.assert_array_equal(a, b)
    # Fixpoint: one more application changes nothing.
    np.testing.assert_array_equal(modal_filter(a, PAL3, max_iters=1), a)


# ------------------------------------------------------------- area opening


def test_area_opening_absorbs_small_component() -> None:
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:5, 3:5] = 1  # 4-px island
    out = area_opening(labels, PAL3, speck_px=5)
    assert np.all(out == 0)


def test_area_opening_threshold_is_strict() -> None:
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:5, 3:5] = 1  # exactly 4 px
    out = area_opening(labels, PAL3, speck_px=4)  # area == threshold → kept
    assert int(out[3, 3]) == 1


def test_area_opening_prefers_longest_boundary() -> None:
    # Speck (label 2) sits in a corner between label 0 (long contact) and
    # label 1 (short contact): boundary length must win over color proximity.
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[:, 6:] = 1
    labels[3:5, 5] = 2  # 2-px speck: 3 edges to label 0, 2 edges to label 1
    pal = _palette([(20.0, 0.0, 0.0), (88.0, 0.0, 0.0), (90.0, 0.0, 0.0)])
    out = area_opening(labels, pal, speck_px=3)
    assert int(out[3, 5]) == 0  # longest boundary, despite 2≈1 in color


def test_area_opening_chain_resolves_to_surviving_region() -> None:
    # Speck A (label 2) touches only speck B (label 1); B touches region 0.
    # A → B → 0 must chain so BOTH end at label 0 (no stale intermediate).
    pal = _palette([(20.0, 0.0, 0.0), (55.0, 0.0, 0.0), (56.0, 0.0, 0.0)])
    labels = np.zeros((3, 12), dtype=np.int32)
    labels[1, 10] = 2  # A: 1 px, only 4-neighbor with a boundary majority is B
    labels[0:3, 9] = 1  # B: 3-px bar between A and the big 0 field
    labels[0, 10:] = 1
    labels[2, 10:] = 1
    labels[1, 11] = 1  # enclose A so its only neighbors are B-labeled pixels
    out = area_opening(labels, pal, speck_px=9)  # B is exactly 8 px — strict <
    assert set(np.unique(out)) == {0}  # everything chains down to region 0


def test_single_region_map_is_untouched() -> None:
    labels = np.zeros((6, 6), dtype=np.int32)
    out = area_opening(labels, PAL3, speck_px=100)
    np.testing.assert_array_equal(out, labels)


# ------------------------------------------------------------- full cleanup


def test_denoise_end_to_end_and_provenance() -> None:
    rng = np.random.default_rng(1)
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[:, 16:] = 1
    speckle = rng.random((32, 32)) < 0.05
    labels[speckle] = 2  # sprinkle isolated specks
    out = denoise_label_map(_label_map(labels), PAL3, speck_px=4)
    # No isolated single pixels remain.
    padded = np.pad(out.labels, 1, constant_values=-1)
    isolated = 0
    for y in range(32):
        for x in range(32):
            window = padded[y : y + 3, x : x + 3]
            if np.count_nonzero(window == out.labels[y, x]) == 1:
                isolated += 1
    assert isolated == 0
    assert out.provenance.stage_name == "denoise"
    with pytest.raises(ConfigError):
        denoise_label_map(_label_map(labels), PAL3, max_modal_iters=-1)


def test_speck_threshold_math() -> None:
    # d_min 3.5 mm at work_scale 0.35 pt/px: ppmm = 72/(0.35·25.4) ≈ 8.1 px/mm,
    # A_min = π·1.75²·8.1² ≈ 631 px → /16 ≈ 39.
    value = speck_threshold(3.5, 0.35, 16)
    assert 30 < value < 50
    assert speck_threshold(0.1, 10.0, 16) == 4  # floor
    with pytest.raises(ConfigError):
        speck_threshold(3.5, 0.0)


def test_stage_via_context() -> None:
    ctx = InMemoryContext(seed=0)
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[10, 10] = 1
    ctx.put("label_map", _label_map(labels))
    pal2 = _palette([(20.0, 0.0, 0.0), (80.0, 0.0, 0.0)])
    ctx.put("palette", pal2)
    ctx.put(
        "raster_working",
        RasterImage(np.zeros((64, 64, 3), dtype=np.float32), 0.35, 1.0, False, 1, PROV),
    )
    DenoiseStage({}).run(ctx)
    cleaned = ctx.get("label_map")
    assert isinstance(cleaned, LabelMap)
    assert int(cleaned.labels[10, 10]) == 0  # speck gone
    with pytest.raises(ConfigError, match="denoise config"):
        DenoiseStage({"speck_divisor": "big"})
