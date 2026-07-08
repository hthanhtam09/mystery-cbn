"""Golden tests for the Preprocessing stage.

Pinned digests are of the float32 working-raster buffer for a deterministic
gradient fixture. Validity is tied to the pinned OpenCV build (5.x) of the
benchmark container (BENCHMARK_SPEC §9.1) — an OpenCV upgrade that changes
these bytes is a golden-update event, reviewed like any other
(BENCHMARK_SPEC §4.3).
"""

from __future__ import annotations

import hashlib

import numpy as np

from mysterycbn.model.records import Provenance, RasterImage
from mysterycbn.stages.raster.preprocess import preprocess_raster

_GOLDEN = {
    "bilateral": "41baed2288cd63a327855e9ad7f71ff8b880a1f58c5cd52779798b0a6d4b9097",
    "guided": "ea37800435eaebcda42fe1172268aad057466ea2d461dd098ca3b0025ba50a13",
    "resize_only": "620650babc24e3b553dfab9b0e84887ad14c82db8175042a2d892b6f7369aa89",
}


def _source() -> RasterImage:
    h, w = 200, 300
    y, x = np.mgrid[0:h, 0:w]
    pixels = np.stack([x / (w - 1), y / (h - 1), (x + y) / (w + h - 2)], axis=2)
    prov = Provenance("load", "1.0.0", "0" * 64, "1" * 64)
    return RasterImage(pixels.astype(np.float32), 0.0, 1.0, False, 1, prov)


def _digest(raster: RasterImage) -> str:
    return hashlib.sha256(raster.pixels.tobytes()).hexdigest()


def test_golden_bilateral_default() -> None:
    out = preprocess_raster(_source(), max_working_px=128)
    assert out.pixels.shape == (85, 128, 3)
    assert _digest(out) == _GOLDEN["bilateral"]


def test_golden_guided_alternative() -> None:
    out = preprocess_raster(_source(), max_working_px=128, impl="guided")
    assert _digest(out) == _GOLDEN["guided"]


def test_golden_resize_only_and_determinism() -> None:
    a = preprocess_raster(_source(), max_working_px=128, smooth_passes=0)
    b = preprocess_raster(_source(), max_working_px=128, smooth_passes=0)
    assert _digest(a) == _GOLDEN["resize_only"]
    assert _digest(a) == _digest(b)  # I2: byte-identical across runs
