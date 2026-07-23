"""Unit tests for the ink-line stages (detection + vectorization)."""

from __future__ import annotations

import numpy as np

from mysterycbn.stages.raster.ink_detect import detect_ink_mask
from mysterycbn.stages.vector.ink_overlay import vectorize_ink

_KW = dict(
    ppmm=10.0,
    max_width_mm=0.6,
    contrast_l=8.0,
    darkness_l=55.0,
    survived_l=25.0,
    min_length_mm=2.0,
)


def _field(light: float = 0.75) -> np.ndarray:
    return np.full((80, 80, 3), light, np.float32)


def test_thin_dark_line_detected() -> None:
    px = _field()
    px[40, 10:70, :] = 0.15  # 1-px dark horizontal line
    m = detect_ink_mask(px, np.zeros((80, 80), np.int32), np.array([80.0]), **_KW)
    assert m[40, 10:70].all()
    assert m.sum() == 60


def test_thick_blob_not_inked() -> None:
    px = _field()
    px[30:60, 30:60, :] = 0.15  # wide dark blob
    m = detect_ink_mask(px, np.zeros((80, 80), np.int32), np.array([80.0]), **_KW)
    assert m.sum() == 0


def test_line_already_dark_after_quantize_skipped() -> None:
    px = _field()
    px[40, 10:70, :] = 0.15
    # palette fill is near-black -> lost-by-quantize gate drops the line.
    m = detect_ink_mask(px, np.zeros((80, 80), np.int32), np.array([10.0]), **_KW)
    assert m.sum() == 0


def test_short_speckle_dropped() -> None:
    px = _field()
    px[40, 40:43, :] = 0.15  # 3-px < min_length (2mm * 10ppmm = 20px)
    m = detect_ink_mask(px, np.zeros((80, 80), np.int32), np.array([80.0]), **_KW)
    assert m.sum() == 0


def test_disabled_scale_returns_empty() -> None:
    px = _field()
    px[40, 10:70, :] = 0.15
    m = detect_ink_mask(
        px, np.zeros((80, 80), np.int32), np.array([80.0]), **{**_KW, "ppmm": 0.0}
    )
    assert m.sum() == 0


def test_vectorize_empty_mask() -> None:
    assert vectorize_ink(np.zeros((10, 10), bool), scale=2.0, origin_xy=(0.0, 0.0)) == ()


def test_vectorize_maps_phi_and_is_deterministic() -> None:
    mask = np.zeros((10, 10), bool)
    mask[5, 2:7] = True
    a = vectorize_ink(mask, scale=2.0, origin_xy=(3.0, 4.0))
    b = vectorize_ink(mask, scale=2.0, origin_xy=(3.0, 4.0))
    assert len(a) >= 1
    # determinism: identical polylines, same order.
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
    # Φ: normal pixel (r, c) -> (m_x + (c+1)*scale, m_y + (r+1)*scale).
    xs = np.concatenate([p[:, 0] for p in a])
    ys = np.concatenate([p[:, 1] for p in a])
    assert xs.min() == 3.0 + (2 + 1) * 2.0
    assert (ys == 4.0 + (5 + 1) * 2.0).all()
