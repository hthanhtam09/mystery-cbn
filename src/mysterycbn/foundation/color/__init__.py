"""Color science API: all color conversion and difference math, one place (ARCHITECTURE.md §6)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ColorScience(ABC):
    """sRGB↔LAB conversion, ΔE metrics, and colorfulness — the only color math in the engine."""

    @abstractmethod
    def srgb_to_lab(self, srgb: np.ndarray) -> np.ndarray:
        """Convert (..., 3) float sRGB in [0, 1] to CIELAB."""

    @abstractmethod
    def lab_to_srgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert (..., 3) CIELAB to float sRGB clipped to [0, 1]."""

    @abstractmethod
    def delta_e_76(self, lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
        """CIE76 color difference, broadcast over leading dimensions."""

    @abstractmethod
    def delta_e_2000(self, lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
        """CIEDE2000 color difference, broadcast over leading dimensions."""

    @abstractmethod
    def colorfulness(self, srgb: np.ndarray) -> float:
        """Hasler–Süsstrunk colorfulness metric of an (H, W, 3) sRGB image."""


# sRGB D65 constants, pinned to the exact digits in MATH_SPEC §3.2 — library
# defaults differ in the 7th digit and would break byte-determinism (I2).
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0  # CIELAB nonlinearity joint


class DefaultColorScience(ColorScience):
    """Reference implementation of MATH_SPEC §3–§4 in vectorized float64.

    All arrays broadcast over leading dimensions; the trailing axis is the
    3-channel color. ΔE00 conforms to Sharma, Wu & Dalal (2005), including the
    ``h′ = 0`` convention at zero chroma.
    """

    def srgb_to_lab(self, srgb: np.ndarray) -> np.ndarray:
        c = np.clip(np.asarray(srgb, dtype=np.float64), 0.0, 1.0)
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        xyz = lin @ _SRGB_TO_XYZ.T / _WHITE_D65
        f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3.0 * _DELTA**2) + 4.0 / 29.0)
        lab = np.empty_like(f)
        lab[..., 0] = 116.0 * f[..., 1] - 16.0
        lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
        lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
        return lab

    def lab_to_srgb(self, lab: np.ndarray) -> np.ndarray:
        lab = np.asarray(lab, dtype=np.float64)
        fy = (lab[..., 0] + 16.0) / 116.0
        f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], axis=-1)
        xyz = np.where(f > _DELTA, f**3, 3.0 * _DELTA**2 * (f - 4.0 / 29.0)) * _WHITE_D65
        lin = xyz @ _XYZ_TO_SRGB.T
        lin = np.clip(lin, 0.0, None)
        srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
        return np.clip(srgb, 0.0, 1.0)

    def delta_e_76(self, lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
        a = np.asarray(lab_a, dtype=np.float64)
        b = np.asarray(lab_b, dtype=np.float64)
        return np.asarray(np.linalg.norm(a - b, axis=-1))

    def delta_e_2000(self, lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
        a = np.asarray(lab_a, dtype=np.float64)
        b = np.asarray(lab_b, dtype=np.float64)
        l1, a1, b1 = a[..., 0], a[..., 1], a[..., 2]
        l2, a2, b2 = b[..., 0], b[..., 1], b[..., 2]

        c1 = np.hypot(a1, b1)
        c2 = np.hypot(a2, b2)
        c_bar = (c1 + c2) / 2.0
        g = 0.5 * (1.0 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
        a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
        c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
        # Sharma convention: h' = 0 where C' = 0.
        h1p = np.where(c1p == 0.0, 0.0, np.degrees(np.arctan2(b1, a1p)) % 360.0)
        h2p = np.where(c2p == 0.0, 0.0, np.degrees(np.arctan2(b2, a2p)) % 360.0)

        dlp = l2 - l1
        dcp = c2p - c1p
        zero_chroma = c1p * c2p == 0.0
        dh = h2p - h1p
        dh = np.where(dh > 180.0, dh - 360.0, dh)
        dh = np.where(dh < -180.0, dh + 360.0, dh)
        dh = np.where(zero_chroma, 0.0, dh)
        dhp_big = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2.0)

        l_bar = (l1 + l2) / 2.0
        c_barp = (c1p + c2p) / 2.0
        h_sum = h1p + h2p
        h_diff = np.abs(h1p - h2p)
        h_bar = np.where(
            zero_chroma,
            h_sum,
            np.where(
                h_diff <= 180.0,
                h_sum / 2.0,
                np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
            ),
        )
        t = (
            1.0
            - 0.17 * np.cos(np.radians(h_bar - 30.0))
            + 0.24 * np.cos(np.radians(2.0 * h_bar))
            + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
            - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
        )
        d_theta = 30.0 * np.exp(-(((h_bar - 275.0) / 25.0) ** 2))
        r_c = 2.0 * np.sqrt(c_barp**7 / (c_barp**7 + 25.0**7))
        s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
        s_c = 1.0 + 0.045 * c_barp
        s_h = 1.0 + 0.015 * c_barp * t
        r_t = -r_c * np.sin(np.radians(2.0 * d_theta))
        return np.asarray(
            np.sqrt(
                (dlp / s_l) ** 2
                + (dcp / s_c) ** 2
                + (dhp_big / s_h) ** 2
                + r_t * (dcp / s_c) * (dhp_big / s_h)
            )
        )

    def colorfulness(self, srgb: np.ndarray) -> float:
        img = np.asarray(srgb, dtype=np.float64) * 255.0
        rg = img[..., 0] - img[..., 1]
        yb = 0.5 * (img[..., 0] + img[..., 1]) - img[..., 2]
        sigma = float(np.hypot(rg.std(), yb.std()))
        mu = float(np.hypot(rg.mean(), yb.mean()))
        return sigma + 0.3 * mu
