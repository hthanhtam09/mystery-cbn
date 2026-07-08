"""Shared synthetic-image fixtures for the visual debugger's tests. No real
photographs are used -- generated in-repo (ARCHITECTURE.md §10 legal
invariant), matching the pattern already used by
``tests/integration/test_convert_end_to_end.py``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def two_tone_image_bytes() -> bytes:
    w, h = 64, 64
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (220, 30, 30)
    arr[:, w // 2 :] = (30, 30, 220)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()
