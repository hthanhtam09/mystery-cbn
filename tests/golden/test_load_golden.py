"""Golden tests for the Raster Load stage.

Fixtures are generated deterministically in-test (lossless formats only, so
decoded pixels are platform-stable); the golden values are SHA-256 digests of
the canonical float32 output buffer. Regenerating goldens requires a reviewed
update to the digests below (BENCHMARK_SPEC §4.3). JPEG is excluded: decoder
output may legally vary across libjpeg builds, so JPEG is covered by
statistical assertions in the unit suite instead.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image

from mysterycbn.stages.raster.load import load_bytes

_GOLDEN_GRADIENT = "0fc708874cb24f3cc334356914fa39a7d245e0bf6ef14f3ab71baa8ef6499d9a"


def _gradient(h: int = 64, w: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.stack(
        [x * 255 // (w - 1), y * 255 // (h - 1), (x + y) * 255 // (w + h - 2)], axis=2
    ).astype(np.uint8)


def _encode(fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(_gradient()).save(buf, fmt, **kw)
    return buf.getvalue()


def _digest(data: bytes) -> str:
    return hashlib.sha256(load_bytes(data).pixels.tobytes()).hexdigest()


def test_golden_png() -> None:
    assert _digest(_encode("PNG")) == _GOLDEN_GRADIENT


def test_golden_webp_lossless() -> None:
    assert _digest(_encode("WEBP", lossless=True)) == _GOLDEN_GRADIENT


def test_golden_tiff() -> None:
    assert _digest(_encode("TIFF")) == _GOLDEN_GRADIENT


def test_golden_cross_format_agreement() -> None:
    """The loader is format-agnostic: all lossless containers of the same
    content must produce byte-identical canonical rasters."""
    digests = {
        _digest(_encode("PNG")),
        _digest(_encode("WEBP", lossless=True)),
        _digest(_encode("TIFF")),
        _digest(_encode("BMP")),
    }
    assert digests == {_GOLDEN_GRADIENT}
