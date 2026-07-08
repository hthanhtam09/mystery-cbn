"""Golden tests for the Quantization stage.

Digest = SHA-256 of (label-map bytes ‖ palette LAB rounded to 6 decimals) on
a deterministic gradient fixture, seed 0. Pure-NumPy math (no BLAS-order
dependence in the reduction paths used), so digests are expected stable
across platforms of the pinned container; a NumPy upgrade that changes them
is a reviewed golden-update event (BENCHMARK_SPEC §4.3).
"""

from __future__ import annotations

import hashlib

import numpy as np

from mysterycbn.model.records import Provenance, RasterImage
from mysterycbn.stages.raster.quantize import quantize_raster

_GOLDEN = {
    "labkmeans": "05d5ba4fba9ff2d8ea184ea4016ff398908b3310d14a71270c442130eb651042",
    "mediancut": "cb47150da7fb22229121bdbccd0236cce455258631a0fad733af3c319f04831c",
}


def _fixture() -> RasterImage:
    h, w = 96, 128
    y, x = np.mgrid[0:h, 0:w]
    px = np.stack([x / (w - 1), y / (h - 1), 0.5 * np.ones((h, w))], axis=2)
    prov = Provenance("preprocess", "1.0.0", "0" * 64, "1" * 64)
    return RasterImage(px.astype(np.float32), 0.4, 1.0, False, 1, prov)


def _digest(impl: str) -> str:
    lm, pal = quantize_raster(_fixture(), n_colors=8, impl=impl, seed=0)  # type: ignore[arg-type]
    lab = np.round(np.array([c.lab for c in pal.colors]), 6)
    return hashlib.sha256(lm.labels.tobytes() + lab.tobytes()).hexdigest()


def test_golden_labkmeans_default() -> None:
    assert _digest("labkmeans") == _GOLDEN["labkmeans"]


def test_golden_mediancut_alternative() -> None:
    assert _digest("mediancut") == _GOLDEN["mediancut"]
