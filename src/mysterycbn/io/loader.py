"""Image loading: decode, EXIF orientation, normalize to float32 sRGB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..core.errors import InputError
from ..core.pipeline import FunctionStage
from ..core.types import PipelineContext

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def load_image(path: str | Path) -> np.ndarray:
    """Decode ``path`` to an H×W×3 float32 sRGB array in [0, 1].

    Applies EXIF orientation and flattens transparency onto white (the page
    background), so downstream stages never see an alpha channel.
    """
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InputError(f"unsupported input format: {path.suffix!r}")
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
                img = img.convert("RGBA")
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            rgb = img.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
    except UnidentifiedImageError as exc:
        raise InputError(f"cannot decode image: {path}") from exc
    if array.shape[0] < 32 or array.shape[1] < 32:
        raise InputError(f"image too small ({array.shape[1]}×{array.shape[0]} px)")
    return array


def make_load_stage(path: str | Path) -> FunctionStage:
    def _run(ctx: PipelineContext) -> None:
        ctx.image = load_image(path)

    return FunctionStage("load", _run, requires=(), provides=("image",))
