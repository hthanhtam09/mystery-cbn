"""Unit tests for the fill_holes stage."""
from __future__ import annotations

import numpy as np

from mysterycbn.stages.raster.fill_holes import fill_small_holes


def test_small_enclosed_hole_filled() -> None:
    # A field of label 1 with a single enclosed pixel of label 2 -> filled to 1.
    a = np.ones((7, 7), np.int32)
    a[3, 3] = 2
    out = fill_small_holes(a, max_hole_px=5)
    assert out[3, 3] == 1
    assert (out == 1).all()


def test_hole_above_size_not_filled() -> None:
    a = np.ones((9, 9), np.int32)
    a[3:6, 3:6] = 2  # 9-px enclosed block
    out = fill_small_holes(a, max_hole_px=5)  # 9 > 5 -> keep
    assert (out[3:6, 3:6] == 2).all()


def test_border_hole_not_filled() -> None:
    a = np.ones((7, 7), np.int32)
    a[0, 3] = 2  # touches top border -> not enclosed
    out = fill_small_holes(a, max_hole_px=5)
    assert out[0, 3] == 2


def test_two_neighbor_region_not_filled() -> None:
    # A small region bordering two different labels is not a single-enclosed
    # hole -> left alone.
    a = np.ones((6, 8), np.int32)
    a[:, 4:] = 3
    a[2, 3:5] = 2  # straddles the 1|3 seam -> borders both
    out = fill_small_holes(a, max_hole_px=10)
    assert (out[2, 3:5] == 2).all()


def test_disabled_threshold_noop() -> None:
    a = np.ones((7, 7), np.int32)
    a[3, 3] = 2
    out = fill_small_holes(a, max_hole_px=0)
    assert out[3, 3] == 2
