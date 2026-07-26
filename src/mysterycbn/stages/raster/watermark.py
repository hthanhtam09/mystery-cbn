"""Generator-watermark removal (preprocess helper, not a pipeline stage).

Gemini stamps every image it returns with a small four-pointed sparkle in
the lower-right area. It is a **uniform-alpha white overlay**, not opaque
pixels: inside the glyph the image reads ``obs = orig*(1-a) + 1.0*a`` with
a measured on real samples at ``a = 0.3034``. Left in place it survives
quantization as its own numbered region on the finished page.

Why un-blending rather than inpainting: the sparkle is routinely stamped
on top of real line work (a brick seam, a leaf outline, the rim of a
stool). Inpainting throws that content away and reconstructs a smear,
which is visibly worse than the watermark it removed. Inverting the blend
instead -- ``orig = (obs - a) / (1 - a)`` -- restores the covered pixels
essentially exactly, outlines included, and touches nothing outside the
glyph.

Geometry and alpha were fitted against the Gemini samples in
``mystery-cbn-api/debug/failed_inputs`` (three distinct 1824x2336 images)
by minimizing the post-removal residual over a flat-background patch:
mean |error| drops 5.51 -> 1.20 gray levels, the remainder being the
source's own background grain. The glyph sits at a fixed relative
position and is round in *pixels*, so both radii are expressed as
fractions of image width.

``remove_gemini_watermark`` is guarded: it probes for the glyph's
signature step at the known boundary and returns the input untouched
when no watermark is there, so running it on non-Gemini artwork is a
no-op rather than a star-shaped bruise.
"""

from __future__ import annotations

import numpy as np

# Fitted against real Gemini output (see module docstring). Center is a
# fraction of width/height; both radii are fractions of WIDTH so the glyph
# stays round regardless of the source's aspect ratio.
WATERMARK_CENTER_X_FRAC = 0.8684211
WATERMARK_CENTER_Y_FRAC = 0.8973673
WATERMARK_RADIUS_X_FRAC = 0.0275822
WATERMARK_RADIUS_Y_FRAC = 0.0268640
# Superellipse exponent < 1 => concave sides, i.e. a four-pointed sparkle.
WATERMARK_EXPONENT = 0.70
WATERMARK_ALPHA = 0.3034

# Presence test. Measured medians: 0.177-0.301 with the watermark present,
# and <= 0.020 without it (non-Gemini images, and Gemini images probed at
# nine wrong locations), so 0.10 separates the two with wide margin.
_DETECT_THRESHOLD = 0.10
_PROBE_SAMPLES = 256
_PROBE_INNER_SCALE = 0.78
_PROBE_OUTER_SCALE = 1.22
# Guards the (1 - background) divisor when the background is already white.
_MIN_HEADROOM = 10.0 / 255.0
_SUPERSAMPLE = 4


def _star_coverage(
    height: int, width: int, cx: float, cy: float, rx: float, ry: float, exponent: float
) -> np.ndarray:
    """Antialiased [0, 1] coverage of the sparkle over a full-size canvas.

    Supersampled ``_SUPERSAMPLE``x per axis, evaluated only on the glyph's
    bounding box -- the rest of the canvas is exactly zero, which is what
    keeps the removal local.
    """
    y0 = max(0, int(cy - ry) - 4)
    y1 = min(height, int(cy + ry) + 5)
    x0 = max(0, int(cx - rx) - 4)
    x1 = min(width, int(cx + rx) + 5)
    coverage = np.zeros((height, width), dtype=np.float32)
    if y1 <= y0 or x1 <= x0:
        return coverage
    bh, bw = y1 - y0, x1 - x0
    ss = _SUPERSAMPLE
    yy, xx = np.mgrid[0 : bh * ss, 0 : bw * ss]
    px = x0 + (xx + 0.5) / ss
    py = y0 + (yy + 0.5) / ss
    inside = (
        (np.abs(px - cx) / rx) ** exponent + (np.abs(py - cy) / ry) ** exponent
    ) <= 1.0
    coverage[y0:y1, x0:x1] = inside.reshape(bh, ss, bw, ss).mean(axis=(1, 3))
    return coverage


def detect_watermark_alpha(
    pixels: np.ndarray, cx: float, cy: float, rx: float, ry: float, exponent: float
) -> float:
    """Median alpha implied by the intensity step across the glyph boundary.

    Samples pairs of points just inside and just outside the fitted outline.
    Real artwork crossing that outline produces steps of random sign, so the
    median lands near zero; the watermark produces a consistent positive step
    of ``WATERMARK_ALPHA``. Background is assumed locally continuous across
    the ~8px probe span, which holds well below the glyph's own scale.
    """
    height, width = pixels.shape[:2]
    t = np.linspace(0.0, 2.0 * np.pi, _PROBE_SAMPLES, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    # Superellipse parametrization: |x/rx|^e + |y/ry|^e == 1 for all t.
    bx = rx * np.sign(ct) * np.abs(ct) ** (2.0 / exponent)
    by = ry * np.sign(st) * np.abs(st) ** (2.0 / exponent)

    def sample(scale: float) -> np.ndarray:
        xs = np.clip((cx + bx * scale).round().astype(int), 0, width - 1)
        ys = np.clip((cy + by * scale).round().astype(int), 0, height - 1)
        return pixels[ys, xs]

    inner = sample(_PROBE_INNER_SCALE)
    outer = sample(_PROBE_OUTER_SCALE)
    headroom = np.clip(1.0 - outer, _MIN_HEADROOM, None)
    return float(np.median(((inner - outer) / headroom).mean(axis=1)))


def remove_gemini_watermark(
    pixels: np.ndarray,
    *,
    alpha: float = WATERMARK_ALPHA,
    center_x_frac: float = WATERMARK_CENTER_X_FRAC,
    center_y_frac: float = WATERMARK_CENTER_Y_FRAC,
    radius_x_frac: float = WATERMARK_RADIUS_X_FRAC,
    radius_y_frac: float = WATERMARK_RADIUS_Y_FRAC,
    exponent: float = WATERMARK_EXPONENT,
    detect_threshold: float = _DETECT_THRESHOLD,
) -> np.ndarray:
    """Invert the sparkle's alpha blend, or return ``pixels`` unchanged.

    Expects float32 sRGB in [0, 1] (the engine's raster representation).
    Returns the input array itself when no watermark is detected, so callers
    can enable this unconditionally.
    """
    height, width = pixels.shape[:2]
    cx = width * center_x_frac
    cy = height * center_y_frac
    rx = width * radius_x_frac
    ry = width * radius_y_frac
    # Too small to carry the glyph, or the glyph would fall outside the frame.
    if rx < 2.0 or ry < 2.0 or cx + rx >= width or cy + ry >= height:
        return pixels

    source = np.asarray(pixels, dtype=np.float32)
    if detect_watermark_alpha(source, cx, cy, rx, ry, exponent) < detect_threshold:
        return pixels

    a = (_star_coverage(height, width, cx, cy, rx, ry, exponent) * alpha)[:, :, None]
    recovered = (source - a) / np.clip(1.0 - a, 1e-3, None)
    return np.clip(recovered, 0.0, 1.0).astype(np.float32)
