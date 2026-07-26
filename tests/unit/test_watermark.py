"""Unit tests for Gemini sparkle-watermark removal (stages/raster/watermark.py).

The tests stamp the watermark synthetically using the same blend the remover
inverts, so "does it come back off cleanly" is measurable rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mysterycbn.stages.raster.watermark import (
    WATERMARK_ALPHA,
    WATERMARK_CENTER_X_FRAC,
    WATERMARK_CENTER_Y_FRAC,
    WATERMARK_EXPONENT,
    WATERMARK_RADIUS_X_FRAC,
    WATERMARK_RADIUS_Y_FRAC,
    _star_coverage,
    detect_watermark_alpha,
    remove_gemini_watermark,
)

# Matches the real Gemini output the geometry was fitted against.
GEMINI_H, GEMINI_W = 2336, 1824
RNG = np.random.default_rng(0)


def _coverage(h: int = GEMINI_H, w: int = GEMINI_W) -> np.ndarray:
    return _star_coverage(
        h,
        w,
        w * WATERMARK_CENTER_X_FRAC,
        h * WATERMARK_CENTER_Y_FRAC,
        w * WATERMARK_RADIUS_X_FRAC,
        w * WATERMARK_RADIUS_Y_FRAC,
        WATERMARK_EXPONENT,
    )


def _stamp(pixels: np.ndarray, alpha: float = WATERMARK_ALPHA) -> np.ndarray:
    """Alpha-blend the sparkle over ``pixels`` exactly as Gemini does."""
    a = (_coverage(*pixels.shape[:2]) * alpha)[:, :, None]
    return (pixels * (1.0 - a) + a).astype(np.float32)


def _artwork(h: int = GEMINI_H, w: int = GEMINI_W) -> np.ndarray:
    """Flat mid-tone with a black line running under the watermark, so the
    tests cover the case inpainting got wrong (content beneath the glyph)."""
    art = np.full((h, w, 3), 0.62, dtype=np.float32)
    art[int(h * 0.897) - 2 : int(h * 0.897) + 3, :] = 0.05  # line through the glyph
    return art


# ---------------------------------------------------------------- geometry


def test_coverage_is_local_and_star_shaped() -> None:
    cov = _coverage()
    assert cov.shape == (GEMINI_H, GEMINI_W)
    # Fully inside at the center, untouched far away.
    cy, cx = int(GEMINI_H * WATERMARK_CENTER_Y_FRAC), int(GEMINI_W * WATERMARK_CENTER_X_FRAC)
    assert cov[cy, cx] == pytest.approx(1.0)
    assert cov[:2000, :1200].max() == 0.0
    # Concave sides: the diagonal at 45% of the radius is already outside,
    # while the same distance along the axis is still inside (a four-pointed
    # sparkle, not an ellipse).
    r = GEMINI_W * WATERMARK_RADIUS_X_FRAC
    assert cov[cy, cx + int(r * 0.45)] > 0.0
    assert cov[cy + int(r * 0.45), cx + int(r * 0.45)] == 0.0
    # The glyph occupies a tiny fraction of the page.
    assert 0.0002 < cov.mean() < 0.002


def test_coverage_is_antialiased() -> None:
    cov = _coverage()
    partial = cov[(cov > 0.0) & (cov < 1.0)]
    assert partial.size > 100  # a real soft edge, not a hard binary cut


# ---------------------------------------------------------------- detection


def test_detects_stamped_watermark() -> None:
    art = _artwork()
    assert detect_watermark_alpha(_stamp(art), *_probe_args()) > 0.10


def test_does_not_detect_on_clean_artwork() -> None:
    assert detect_watermark_alpha(_artwork(), *_probe_args()) < 0.10


def test_does_not_detect_on_noise() -> None:
    """High-variance content must not trigger removal. The statistic is
    one-sided by construction — only a consistent *positive* step means a
    white overlay — so a negative median is a non-detection, not a miss."""
    noisy = RNG.random((GEMINI_H // 4, GEMINI_W // 4, 3)).astype(np.float32)
    h, w = noisy.shape[:2]
    args = (
        w * WATERMARK_CENTER_X_FRAC,
        h * WATERMARK_CENTER_Y_FRAC,
        w * WATERMARK_RADIUS_X_FRAC,
        w * WATERMARK_RADIUS_Y_FRAC,
        WATERMARK_EXPONENT,
    )
    assert detect_watermark_alpha(noisy, *args) < 0.10
    assert remove_gemini_watermark(noisy) is noisy


def _probe_args() -> tuple[float, float, float, float, float]:
    return (
        GEMINI_W * WATERMARK_CENTER_X_FRAC,
        GEMINI_H * WATERMARK_CENTER_Y_FRAC,
        GEMINI_W * WATERMARK_RADIUS_X_FRAC,
        GEMINI_W * WATERMARK_RADIUS_Y_FRAC,
        WATERMARK_EXPONENT,
    )


# ----------------------------------------------------------------- removal


def test_unblend_recovers_the_original() -> None:
    """The whole point: content under the glyph comes back, not a smear."""
    art = _artwork()
    recovered = remove_gemini_watermark(_stamp(art))
    np.testing.assert_allclose(recovered, art, atol=2.0 / 255.0)


def test_black_line_under_the_glyph_survives() -> None:
    art = _artwork()
    recovered = remove_gemini_watermark(_stamp(art))
    row = int(GEMINI_H * 0.897)
    col = int(GEMINI_W * WATERMARK_CENTER_X_FRAC)
    assert recovered[row, col, 0] == pytest.approx(0.05, abs=2.0 / 255.0)


def test_clean_artwork_is_returned_untouched() -> None:
    art = _artwork()
    out = remove_gemini_watermark(art)
    assert out is art  # identity, not merely equal: detection short-circuits


def test_only_the_glyph_region_changes() -> None:
    art = _artwork()
    recovered = remove_gemini_watermark(_stamp(art))
    changed = (np.abs(recovered - art) > 1.0 / 255.0).any(axis=2)
    assert changed.sum() < 0.001 * art.shape[0] * art.shape[1]
    # Every changed pixel lies inside the glyph's own footprint.
    assert not changed[_coverage() == 0.0].any()


def test_output_stays_in_range() -> None:
    # A near-white background is where un-blending could overshoot past 1.0.
    art = np.full((GEMINI_H, GEMINI_W, 3), 0.985, dtype=np.float32)
    out = remove_gemini_watermark(_stamp(art))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_tiny_image_is_a_noop() -> None:
    tiny = RNG.random((12, 10, 3)).astype(np.float32)
    assert remove_gemini_watermark(tiny) is tiny


# ------------------------------------------------ drifted (resized) glyph


def _stamp_at(pixels: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Stamp the sparkle at an arbitrary center (not the fitted fraction),
    emulating a frame resized to a different aspect ratio."""
    h, w = pixels.shape[:2]
    a = (
        _star_coverage(
            h, w, cx, cy, w * WATERMARK_RADIUS_X_FRAC, w * WATERMARK_RADIUS_Y_FRAC, WATERMARK_EXPONENT
        )
        * WATERMARK_ALPHA
    )[:, :, None]
    return (pixels * (1.0 - a) + a).astype(np.float32)


def test_stronger_overlay_is_estimated_and_removed() -> None:
    """The fitted ``WATERMARK_ALPHA`` (0.30) is only one batch's opacity; newer
    output stamps the sparkle at ~0.6. With a fixed alpha the removal barely
    touches a 0.6 glyph and it survives onto the page. The per-image estimate
    must recover the true opacity and un-blend it. Regression for the reported
    'watermark still there' case."""
    art = _artwork()
    strong = 0.60
    stamped = _stamp(art, alpha=strong)

    # A fixed-0.30 removal leaves most of the glyph energy behind...
    fixed = remove_gemini_watermark(stamped, alpha=WATERMARK_ALPHA)
    fixed_residual = np.abs(fixed - art).sum()
    stamped_energy = np.abs(stamped - art).sum()
    assert fixed_residual > 0.4 * stamped_energy  # under-removed

    # ...while the default (estimate per image) recovers the original closely.
    recovered = remove_gemini_watermark(stamped)
    np.testing.assert_allclose(recovered, art, atol=3.0 / 255.0)


def test_drifted_glyph_is_still_removed() -> None:
    """A watermark shifted off the fitted center -- what happens when the
    source was resized to a different aspect ratio -- must still be found and
    removed. The fixed-position probe misses it; the windowed search recovers
    it. Regression for sparkles surviving onto the finished page."""
    # Aspect ratio deliberately unlike the fitted 1824x2336 (~0.781).
    h, w = 1400, 900
    art = np.full((h, w, 3), 0.55, dtype=np.float32)
    cx = w * WATERMARK_CENTER_X_FRAC - w * 0.045  # drift left ~4.5% of width
    cy = h * WATERMARK_CENTER_Y_FRAC + h * 0.02  # drift down ~2% of height
    stamped = _stamp_at(art, cx, cy)

    # Precondition: the old single fixed probe would have missed this glyph.
    fixed_probe = detect_watermark_alpha(
        stamped,
        w * WATERMARK_CENTER_X_FRAC,
        h * WATERMARK_CENTER_Y_FRAC,
        w * WATERMARK_RADIUS_X_FRAC,
        w * WATERMARK_RADIUS_Y_FRAC,
        WATERMARK_EXPONENT,
    )
    assert fixed_probe < 0.10

    recovered = remove_gemini_watermark(stamped)
    assert recovered is not stamped  # detected and acted on

    stamped_err = np.abs(stamped - art).max(axis=2)
    recovered_err = np.abs(recovered - art).max(axis=2)
    # The overlay is knocked down by the large majority of its total energy;
    # the sub-pixel centroid localization leaves only a thin low-amplitude edge
    # ring rather than a solid star.
    assert recovered_err.sum() < 0.20 * stamped_err.sum()
    # And what remains is not a coherent glyph: the count of strongly-lit
    # pixels (> 0.10) collapses, so nothing survives to quantize into its own
    # numbered region -- which is the whole point of the removal.
    strong = 0.10
    assert (recovered_err > strong).sum() < 0.15 * (stamped_err > strong).sum()
