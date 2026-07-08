"""Raster Load stage: decode any supported source into the canonical raster
(ENGINE_SPEC.md §4; design contract mirrors docs/modules/ conventions).

Pipeline order of operations is normative: decode → EXIF orientation →
ICC → sRGB → alpha over white → bit-depth normalization. Pillow is an
implementation detail and never appears in the interface (ARCHITECTURE.md §1.1).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from mysterycbn.foundation.errors import ConfigError, InputError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Provenance, RasterImage

STAGE_NAME = "load"
STAGE_VERSION = "1.0.0"

SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "TIFF", "BMP"})
_MIN_SIDE = 64
_DEFAULT_MAX_PIXELS = 100_000_000
_UNSET_HASH = "0" * 64

_ORIENTATION_TAG = 0x0112
_SRGB_PROFILE = ImageCms.createProfile("sRGB")

# Modes Pillow decodes 16-bit/32-bit grayscale into.
_DEEP_GRAY_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N", "I"})


@dataclass(frozen=True)
class SourceBytes:
    """The pipeline's initial artifact: raw source file bytes + provenance."""

    data: bytes
    provenance: Provenance = field(init=False)

    def __post_init__(self) -> None:
        if not self.data:
            raise InputError("source is empty")
        object.__setattr__(
            self,
            "provenance",
            Provenance(
                stage_name="source",
                stage_version=STAGE_VERSION,
                config_hash=_UNSET_HASH,
                source_hash=hashlib.sha256(self.data).hexdigest(),
            ),
        )


def _open(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InputError(f"cannot decode image: {exc}") from exc
    return img


def _apply_icc(img: Image.Image, icc_bytes: bytes) -> Image.Image:
    """Convert from the embedded profile to sRGB, relative colorimetric.

    Alpha is carried around the conversion (profiles describe color channels
    only).
    """
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    rgb = img.convert("RGB")
    source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
    converted = ImageCms.profileToProfile(
        rgb,
        source_profile,
        _SRGB_PROFILE,
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    assert converted is not None  # non-inPlace call always returns an image
    if alpha is not None:
        converted.putalpha(alpha)
    return converted


def _to_float_rgb(img: Image.Image) -> np.ndarray:
    """Normalize to float32 (H, W, 3) in [0, 1]; alpha composited over white."""
    if img.mode in _DEEP_GRAY_MODES:
        gray = np.asarray(img, dtype=np.float32)
        divisor = 65535.0 if "16" in img.mode else max(float(gray.max()), 1.0)
        gray = np.clip(gray / divisor, 0.0, 1.0)
        return np.repeat(gray[:, :, None], 3, axis=2)

    has_alpha = "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info)
    arr = np.asarray(img.convert("RGBA" if has_alpha else "RGB"), dtype=np.float32) / 255.0
    if has_alpha:
        alpha = arr[:, :, 3:4]
        return np.asarray(arr[:, :, :3] * alpha + (1.0 - alpha))
    return np.asarray(arr[:, :, :3])


def load_bytes(
    data: bytes,
    *,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    assume_srgb: bool = True,
    config_hash: str = _UNSET_HASH,
) -> RasterImage:
    """Decode ``data`` into the canonical raster (ENGINE_SPEC §4 algorithm).

    Raises ``InputError`` on undecodable, unsupported, oversized, or
    undersized input, and on a missing profile when ``assume_srgb`` is false.
    """
    img = _open(data)
    if img.format not in SUPPORTED_FORMATS:
        raise InputError(
            f"unsupported format {img.format!r} (supported: {sorted(SUPPORTED_FORMATS)})"
        )
    width, height = img.size
    if width * height > max_pixels:
        raise InputError(f"image has {width * height} px, exceeding max_pixels={max_pixels}")
    if min(width, height) < _MIN_SIDE:
        raise InputError(f"min side is {min(width, height)} px; ≥ {_MIN_SIDE} required")

    raw_tag = img.getexif().get(_ORIENTATION_TAG, 1)
    exif_orientation = int(raw_tag) if isinstance(raw_tag, int) and 1 <= raw_tag <= 8 else 1
    oriented = ImageOps.exif_transpose(img)
    assert oriented is not None  # non-inPlace call always returns an image

    icc_applied = False
    icc_bytes = oriented.info.get("icc_profile")
    if icc_bytes and oriented.mode not in _DEEP_GRAY_MODES:
        oriented = _apply_icc(oriented, icc_bytes)
        icc_applied = True
    elif not icc_bytes and not assume_srgb:
        raise InputError("image has no ICC profile and assume_srgb is disabled")

    return RasterImage(
        pixels=_to_float_rgb(oriented),
        work_scale=0.0,
        resize_factor=1.0,
        icc_applied=icc_applied,
        exif_orientation=exif_orientation,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=hashlib.sha256(data).hexdigest(),
        ),
    )


def load_path(
    path: str | Path,
    *,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    assume_srgb: bool = True,
    config_hash: str = _UNSET_HASH,
) -> RasterImage:
    """File-path convenience wrapper over :func:`load_bytes`."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    return load_bytes(data, max_pixels=max_pixels, assume_srgb=assume_srgb, config_hash=config_hash)


class LoadStage:
    """Stage wrapper: ``source_bytes`` → ``raster_source`` (kernel Stage protocol)."""

    def __init__(
        self,
        *,
        max_pixels: int = _DEFAULT_MAX_PIXELS,
        assume_srgb: bool = True,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        self._max_pixels = max_pixels
        self._assume_srgb = assume_srgb
        self._config_hash = config_hash

    @classmethod
    def from_config(cls, section: Mapping[str, object], config_hash: str) -> LoadStage:
        """Build from the resolved ``load`` config section."""
        max_pixels = section.get("max_pixels", _DEFAULT_MAX_PIXELS)
        assume_srgb = section.get("assume_srgb", True)
        if not isinstance(max_pixels, int) or not isinstance(assume_srgb, bool):
            raise ConfigError("load config: max_pixels must be int, assume_srgb bool")
        return cls(max_pixels=max_pixels, assume_srgb=assume_srgb, config_hash=config_hash)

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("source_bytes",)

    @property
    def provides(self) -> tuple[str, ...]:
        return ("raster_source",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        source = ctx.get("source_bytes")
        if not isinstance(source, SourceBytes):
            raise InputError(f"artifact 'source_bytes' has wrong type {type(source).__name__}")
        ctx.put(
            "raster_source",
            load_bytes(
                source.data,
                max_pixels=self._max_pixels,
                assume_srgb=self._assume_srgb,
                config_hash=self._config_hash,
            ),
        )
