"""Benchmarks for the Raster Load stage (budget: 2 MP JPEG ≤ 0.15 s,
ENGINE_SPEC §4). Authoritative numbers come from the pinned container."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from mysterycbn.stages.raster.load import load_bytes

RNG = np.random.default_rng(0)


def _fixture(fmt: str, mp: float, **kw: object) -> bytes:
    side = int((mp * 1e6) ** 0.5)
    arr = RNG.integers(0, 256, (side, side, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **kw)
    return buf.getvalue()


def test_bench_load_jpeg_2mp(benchmark: Any) -> None:
    data = _fixture("JPEG", 2.0, quality=90)
    img = benchmark(load_bytes, data)
    assert img.pixels.shape[2] == 3


def test_bench_load_png_2mp(benchmark: Any) -> None:
    data = _fixture("PNG", 2.0)
    img = benchmark(load_bytes, data)
    assert img.pixels.dtype.name == "float32"


def test_bench_load_webp_05mp(benchmark: Any) -> None:
    data = _fixture("WEBP", 0.5, quality=90)
    img = benchmark(load_bytes, data)
    assert img.resize_factor == 1.0
