"""Tests for loader, preprocess, and quantize using synthetic images."""

import numpy as np
import pytest
from PIL import Image

from mysterycbn.core.config import PreprocessConfig, QuantizeConfig
from mysterycbn.core.errors import InputError
from mysterycbn.io.loader import load_image
from mysterycbn.modules.preprocess import preprocess, resize_to_working
from mysterycbn.modules.quantize import quantize


def make_disk_image(size: int = 200) -> np.ndarray:
    """White background, red disk, blue square — 3 exactly-known colors."""
    img = np.ones((size, size, 3), dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    disk = (yy - size // 3) ** 2 + (xx - size // 3) ** 2 < (size // 5) ** 2
    img[disk] = [0.85, 0.1, 0.1]
    img[int(size * 0.55) : int(size * 0.9), int(size * 0.55) : int(size * 0.9)] = [
        0.1,
        0.15,
        0.8,
    ]
    return img


class TestLoader:
    def test_roundtrip_png(self, tmp_path):
        arr = (make_disk_image(64) * 255).astype(np.uint8)
        path = tmp_path / "img.png"
        Image.fromarray(arr).save(path)
        loaded = load_image(path)
        assert loaded.shape == (64, 64, 3)
        assert loaded.dtype == np.float32
        assert np.allclose(loaded, arr / 255.0, atol=1 / 255)

    def test_alpha_flattened_to_white(self, tmp_path):
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)  # fully transparent black
        path = tmp_path / "img.png"
        Image.fromarray(rgba, "RGBA").save(path)
        loaded = load_image(path)
        assert np.allclose(loaded, 1.0)

    def test_missing_file(self):
        with pytest.raises(InputError, match="not found"):
            load_image("/nonexistent/file.png")

    def test_unsupported_suffix(self, tmp_path):
        path = tmp_path / "img.bmp"
        path.write_bytes(b"xx")
        with pytest.raises(InputError, match="unsupported"):
            load_image(path)

    def test_too_small(self, tmp_path):
        path = tmp_path / "img.png"
        Image.new("RGB", (8, 8)).save(path)
        with pytest.raises(InputError, match="too small"):
            load_image(path)


class TestPreprocess:
    def test_resize_caps_longest_side(self):
        img = np.zeros((1000, 2000, 3), dtype=np.float32)
        out, scale = resize_to_working(img, 500)
        assert max(out.shape[:2]) == 500
        assert scale == pytest.approx(4.0)

    def test_never_upscales(self):
        img = np.zeros((100, 100, 3), dtype=np.float32)
        out, scale = resize_to_working(img, 500)
        assert out.shape == img.shape and scale == 1.0

    def test_smoothing_preserves_flat_colors(self):
        img = make_disk_image()
        out, _ = preprocess(img, PreprocessConfig(max_working_px=400))
        # Deep inside the disk the color must be essentially unchanged.
        assert np.allclose(out[66, 66], [0.85, 0.1, 0.1], atol=0.03)
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestQuantize:
    def test_recovers_known_colors(self):
        img = make_disk_image()
        label_map, palette = quantize(img, QuantizeConfig(n_colors=3, min_delta_e=0))
        assert label_map.shape == img.shape[:2]
        assert len(palette) == 3
        # Color 1 must be the dominant one: white background.
        assert palette[0].rgb == pytest.approx((255, 255, 255), abs=8)

    def test_deterministic(self):
        img = make_disk_image()
        cfg = QuantizeConfig(n_colors=5)
        lm1, p1 = quantize(img, cfg)
        lm2, p2 = quantize(img, cfg)
        assert np.array_equal(lm1, lm2)
        assert [c.rgb for c in p1.colors] == [c.rgb for c in p2.colors]

    def test_delta_e_merge_reduces_palette(self):
        img = make_disk_image()  # only 3 real colors
        _, palette = quantize(img, QuantizeConfig(n_colors=12, min_delta_e=8.0))
        assert len(palette) <= 5  # near-duplicates folded together

    def test_numbers_are_one_based_and_sequential(self):
        _, palette = quantize(make_disk_image(), QuantizeConfig(n_colors=3))
        assert [c.number for c in palette.colors] == list(range(1, len(palette) + 1))
