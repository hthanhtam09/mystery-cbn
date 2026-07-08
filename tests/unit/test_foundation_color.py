"""Unit tests for foundation/color against known values and skimage as oracle."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.color import deltaE_ciede2000

from mysterycbn.foundation.color import DefaultColorScience

CS = DefaultColorScience()


def test_white_black_lab() -> None:
    lab = CS.srgb_to_lab(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(lab[0], [100.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-3)


def test_srgb_lab_roundtrip_bound() -> None:
    rng = np.random.default_rng(0)
    srgb = rng.random((256, 3))
    back = CS.lab_to_srgb(CS.srgb_to_lab(srgb))
    assert float(np.abs(back - srgb).max()) <= 1e-4  # MATH_SPEC §3.2 bound


def test_delta_e_76_is_euclidean() -> None:
    a = np.array([50.0, 10.0, -10.0])
    b = np.array([50.0, 13.0, -6.0])
    assert CS.delta_e_76(a, b) == pytest.approx(5.0)


def test_delta_e_2000_sharma_pair_one() -> None:
    # Sharma, Wu & Dalal (2005) test pair #1.
    a = np.array([50.0, 2.6772, -79.7751])
    b = np.array([50.0, 0.0, -82.7485])
    assert float(CS.delta_e_2000(a, b)) == pytest.approx(2.0425, abs=1e-4)


def test_delta_e_2000_matches_skimage_oracle() -> None:
    rng = np.random.default_rng(1)
    a = np.column_stack(
        [rng.uniform(0, 100, 500), rng.uniform(-90, 90, 500), rng.uniform(-90, 90, 500)]
    )
    b = np.column_stack(
        [rng.uniform(0, 100, 500), rng.uniform(-90, 90, 500), rng.uniform(-90, 90, 500)]
    )
    ours = CS.delta_e_2000(a, b)
    reference = deltaE_ciede2000(a, b)
    np.testing.assert_allclose(ours, reference, atol=1e-6)


def test_delta_e_2000_symmetry_and_identity() -> None:
    rng = np.random.default_rng(2)
    a = rng.uniform(-50, 100, (100, 3))
    b = rng.uniform(-50, 100, (100, 3))
    np.testing.assert_allclose(CS.delta_e_2000(a, b), CS.delta_e_2000(b, a), atol=1e-12)
    np.testing.assert_allclose(CS.delta_e_2000(a, a), 0.0, atol=1e-12)


def test_colorfulness_gray_is_zero_and_saturated_is_large() -> None:
    gray = np.full((8, 8, 3), 0.5)
    assert CS.colorfulness(gray) == pytest.approx(0.0)
    saturated = np.zeros((8, 8, 3))
    saturated[:, :4, 0] = 1.0  # half pure red
    saturated[:, 4:, 2] = 1.0  # half pure blue
    assert CS.colorfulness(saturated) > 100.0
