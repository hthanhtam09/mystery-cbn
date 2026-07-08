"""Unit tests for the Raster Load stage (ENGINE_SPEC §4)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageCms

from mysterycbn.foundation.errors import ConfigError, InputError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.stages.raster.load import LoadStage, SourceBytes, load_bytes, load_path

_H, _W = 64, 96


def _gradient() -> np.ndarray:
    y, x = np.mgrid[0:_H, 0:_W]
    return np.stack(
        [x * 255 // (_W - 1), y * 255 // (_H - 1), (x + y) * 255 // (_W + _H - 2)], axis=2
    ).astype(np.uint8)


def _encode(arr: np.ndarray, fmt: str, mode: str | None = None, **kw: object) -> bytes:
    img = Image.fromarray(arr)
    if mode:
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


# ------------------------------------------------------------------- formats


@pytest.mark.parametrize(
    ("fmt", "kw"),
    [("PNG", {}), ("WEBP", {"lossless": True}), ("TIFF", {}), ("BMP", {})],
)
def test_lossless_formats_decode_exactly(fmt: str, kw: dict[str, object]) -> None:
    img = load_bytes(_encode(_gradient(), fmt, **kw))
    assert img.pixels.shape == (_H, _W, 3)
    np.testing.assert_allclose(img.pixels, _gradient().astype(np.float32) / 255.0, atol=1e-7)
    assert img.work_scale == 0.0
    assert img.resize_factor == 1.0


def test_jpeg_decodes_approximately() -> None:
    img = load_bytes(_encode(_gradient(), "JPEG", quality=95))
    assert img.pixels.shape == (_H, _W, 3)
    assert float(np.abs(img.pixels - _gradient() / 255.0).mean()) < 0.02


def test_unsupported_format_rejected() -> None:
    with pytest.raises(InputError, match="unsupported format"):
        load_bytes(_encode(_gradient(), "GIF"))


def test_corrupt_and_empty_rejected(tmp_path: object) -> None:
    with pytest.raises(InputError, match="cannot decode"):
        load_bytes(b"not an image at all")
    with pytest.raises(InputError, match="empty"):
        SourceBytes(b"")


# ---------------------------------------------------------------- guards


def test_size_guards() -> None:
    with pytest.raises(InputError, match="max_pixels"):
        load_bytes(_encode(_gradient(), "PNG"), max_pixels=100)
    small = np.zeros((32, 100, 3), dtype=np.uint8)
    with pytest.raises(InputError, match="≥ 64"):
        load_bytes(_encode(small, "PNG"))


# ------------------------------------------------------------ orientation


_INVERSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


@pytest.mark.parametrize("tag", [1, 2, 3, 4, 5, 6, 7, 8])
def test_exif_orientations_produce_identical_content(tag: int) -> None:
    base = _gradient()
    img = Image.fromarray(base)
    if tag in _INVERSE:
        img = img.transpose(_INVERSE[tag])
    exif = Image.Exif()
    exif[0x0112] = tag
    buf = io.BytesIO()
    img.save(buf, "PNG", exif=exif)
    loaded = load_bytes(buf.getvalue())
    np.testing.assert_allclose(loaded.pixels, base.astype(np.float32) / 255.0, atol=1e-7)
    assert loaded.exif_orientation == tag


# ------------------------------------------------------------------- alpha


def test_alpha_composited_over_white() -> None:
    rgba = np.zeros((_H, _W, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255  # pure red
    rgba[:, :, 3] = 128  # half transparent
    img = load_bytes(_encode(rgba, "PNG"))
    alpha = 128.0 / 255.0
    expected = np.array([1.0 * alpha + (1 - alpha), 1 - alpha, 1 - alpha], dtype=np.float32)
    np.testing.assert_allclose(img.pixels[0, 0], expected, atol=1e-6)


def test_fully_transparent_becomes_white() -> None:
    rgba = np.zeros((_H, _W, 4), dtype=np.uint8)  # black, alpha 0
    img = load_bytes(_encode(rgba, "PNG"))
    np.testing.assert_allclose(img.pixels, 1.0, atol=1e-7)


# ------------------------------------------------- bit depth & gray modes


def test_grayscale_replicated_to_three_channels() -> None:
    img = load_bytes(_encode(_gradient(), "PNG", mode="L"))
    assert img.pixels.shape == (_H, _W, 3)
    np.testing.assert_array_equal(img.pixels[..., 0], img.pixels[..., 1])
    np.testing.assert_array_equal(img.pixels[..., 1], img.pixels[..., 2])


def test_16bit_grayscale_normalized() -> None:
    deep = (np.linspace(0, 65535, _H * _W).reshape(_H, _W)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(deep).save(buf, "PNG")  # uint16 array → I;16 mode
    img = load_bytes(buf.getvalue())
    assert float(img.pixels.max()) <= 1.0
    assert float(img.pixels.max()) > 0.99  # top of the 16-bit range reached


# --------------------------------------------------------------------- ICC


def test_icc_profile_applied_flag_and_near_identity_for_srgb() -> None:
    srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    data = _encode(_gradient(), "PNG", icc_profile=srgb_profile.tobytes())
    img = load_bytes(data)
    assert img.icc_applied
    # sRGB → sRGB conversion must be a near-identity.
    assert float(np.abs(img.pixels - _gradient() / 255.0).max()) < 0.02


def test_missing_profile_with_assume_srgb_disabled() -> None:
    with pytest.raises(InputError, match="no ICC profile"):
        load_bytes(_encode(_gradient(), "PNG"), assume_srgb=False)


# ------------------------------------------------------- provenance & stage


def test_provenance_and_path_loading(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data = _encode(_gradient(), "PNG")
    img = load_bytes(data, config_hash="ab" * 32)
    assert img.provenance.stage_name == "load"
    assert img.provenance.config_hash == "ab" * 32
    p = tmp_path / "img.png"
    p.write_bytes(data)
    from_path = load_path(p)
    np.testing.assert_array_equal(from_path.pixels, img.pixels)
    with pytest.raises(InputError, match="cannot read"):
        load_path(tmp_path / "missing.png")


def test_load_stage_via_context() -> None:
    stage = LoadStage()
    assert stage.requires == ("source_bytes",)
    assert stage.provides == ("raster_source",)
    ctx = InMemoryContext(seed=0)
    ctx.put("source_bytes", SourceBytes(_encode(_gradient(), "PNG")))
    stage.run(ctx)
    assert ctx.has("raster_source")
    with pytest.raises(ConfigError, match="max_pixels"):
        LoadStage.from_config({"max_pixels": "many"}, "0" * 64)
