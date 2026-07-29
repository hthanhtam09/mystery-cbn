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

The opacity and color of the overlay are re-measured per image rather
than taken from those constants -- the generator stamps at different
opacities -- against a **per-pixel** estimate of the background under
the glyph (``_estimate_overlay``).

``remove_gemini_watermark`` is guarded twice: it probes for the glyph's
signature step at the known boundary, and then requires that a plausible
near-white overlay is actually measurable there. Either gate failing
returns the input untouched, so running it on non-Gemini artwork is a
no-op rather than a star-shaped bruise.
"""

from __future__ import annotations

import cv2
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
# The fitted center is exact only at the resolution the constants were
# measured on (1824x2336). Images resized/cropped to a different aspect ratio
# carry the same glyph at a *drifted* fractional position, so a single fixed
# probe misses it and the sparkle survives onto the page. When the fixed probe
# comes up empty we sweep a small window of candidate centers and take the
# strongest response. Measured separation over this window stays wide --
# watermarked frames score >= 0.167, clean art <= 0.070 -- so the threshold
# still cannot be tripped by real content. Steps are ~1% of the frame apart,
# fine enough that the best candidate lands within a pixel or two of the glyph.
_SEARCH_HALF_FRAC = 0.06
_SEARCH_STEPS = 13
# Guards the (1 - background) divisor when the background is already white.
_MIN_HEADROOM = 10.0 / 255.0
_SUPERSAMPLE = 4

# Second gate, applied after the overlay is measured (see _estimate_overlay).
# The boundary-step probe alone is not sufficient: real artwork whose own edge
# happens to run along the glyph outline can score above _DETECT_THRESHOLD
# (measured 0.121 on a clean, non-Gemini illustration). A genuine sparkle
# always measures a strong, near-white overlay against its own local
# background -- real stamps come in at alpha 0.28-0.51 with every channel of
# C above 0.98 -- while a false positive collapses to alpha ~0.05 and a tinted
# C. Removal only runs when the measurement is plausible on both counts.
_MIN_PLAUSIBLE_ALPHA = 0.15
_MAX_PLAUSIBLE_ALPHA = 0.98
_MIN_OVERLAY_WHITENESS = 0.80
# Background needs this much room above the observed pixel for the per-pixel
# alpha estimate to carry signal.
_MIN_ESTIMATE_HEADROOM = 0.15


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


def _locate_watermark(
    source: np.ndarray,
    cx0: float,
    cy0: float,
    rx: float,
    ry: float,
    exponent: float,
    *,
    half_frac: float,
    steps: int,
) -> tuple[float, float, float]:
    """Best ``(detection, cx, cy)`` for the glyph near its expected center.

    Probes the fitted center first (the common case: an un-resized frame, where
    this returns immediately) and, only if that comes up empty, sweeps a
    ``2*half_frac`` window to recover a glyph shifted by a resize/crop. Only
    candidates whose full glyph stays inside the frame are considered.
    """
    height, width = source.shape[:2]
    best = (detect_watermark_alpha(source, cx0, cy0, rx, ry, exponent), cx0, cy0)
    if best[0] >= _DETECT_THRESHOLD:
        return best

    # Coarse sweep locates the glyph to within a grid cell; the detector's
    # boundary-step statistic is only good to a pixel or two, and removing at a
    # center off by even ~1px subtracts the star pattern in the wrong place --
    # which *adds* a faint inverse sparkle instead of erasing one. So the coarse
    # hit is refined to sub-pixel by the centroid of the glyph's own excess
    # brightness (the white overlay raises intensity in exactly its footprint).
    best = _sweep_centers(
        source, best, cx0, cy0, width * half_frac, height * half_frac, steps, rx, ry, exponent
    )
    if best[0] < _DETECT_THRESHOLD:
        return best
    cx, cy = _refine_center_by_centroid(source, best[1], best[2], rx, ry)
    return (best[0], cx, cy)


def _refine_center_by_centroid(
    source: np.ndarray, cx: float, cy: float, rx: float, ry: float
) -> tuple[float, float]:
    """Sub-pixel center from the intensity-excess centroid in a local patch.

    The sparkle is a white overlay, so within the patch it is the brightest
    excess over the local background (estimated as the patch-border median,
    which the small glyph does not reach). Dark line work under the glyph has
    zero excess and so cannot pull the centroid. Falls back to the input center
    if the patch is degenerate or carries no excess."""
    height, width = source.shape[:2]
    pad = int(round(max(rx, ry) * 1.8)) + 2
    x0, x1 = max(0, int(cx) - pad), min(width, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(height, int(cy) + pad + 1)
    patch = source[y0:y1, x0:x1].mean(axis=2)
    if patch.size == 0:
        return cx, cy
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    background = float(np.median(border))
    excess = np.clip(patch - background, 0.0, None)
    # Keep only the clearly-lit glyph body, not background grain.
    excess[excess < 0.25 * float(excess.max() or 1.0)] = 0.0
    total = float(excess.sum())
    if total <= 0.0:
        return cx, cy
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((xx * excess).sum() / total), float((yy * excess).sum() / total)


def _sweep_centers(
    source: np.ndarray,
    best: tuple[float, float, float],
    cx_c: float,
    cy_c: float,
    half_x: float,
    half_y: float,
    n: int,
    rx: float,
    ry: float,
    exponent: float,
) -> tuple[float, float, float]:
    """Return the strongest ``(det, cx, cy)`` over an ``n x n`` grid of centers
    centered on ``(cx_c, cy_c)``, keeping any prior ``best``. Candidates whose
    glyph would leave the frame are skipped."""
    height, width = source.shape[:2]
    xs = [cx for cx in np.linspace(cx_c - half_x, cx_c + half_x, n) if rx <= cx < width - rx]
    ys = [cy for cy in np.linspace(cy_c - half_y, cy_c + half_y, n) if ry <= cy < height - ry]
    for cy in ys:
        for cx in xs:
            det = detect_watermark_alpha(source, cx, cy, rx, ry, exponent)
            if det > best[0]:
                best = (det, cx, cy)
    return best


def _nearest_background(patch: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """``patch`` with every ``footprint`` pixel replaced by its nearest
    non-footprint neighbour -- the local background the glyph is covering."""
    covered = footprint.astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        covered, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    # distanceTransformWithLabels numbers the *zero* pixels 1..N; invert that
    # numbering into the coordinates it came from.
    rows, cols = np.nonzero(covered == 0)
    lookup = np.zeros((int(labels.max()) + 1, 2), dtype=np.int32)
    lookup[labels[rows, cols]] = np.stack([rows, cols], axis=1)
    nearest = lookup[labels]
    return patch[nearest[..., 0], nearest[..., 1]]


def _estimate_overlay(
    source: np.ndarray, cx: float, cy: float, rx: float, ry: float, exponent: float
) -> tuple[float, np.ndarray] | None:
    """Per-image ``(alpha, color)`` of the sparkle overlay, inverted from the
    blend model ``obs = orig*(1-a) + C*a`` with ``a = coverage*alpha``.

    ``WATERMARK_ALPHA`` and an implicit white ``C`` were fitted on one batch,
    but the generator stamps the glyph at different opacities (measured 0.30 on
    the original fit, ~0.5 on newer output) and its "white" can read faintly
    warm. Removing a 0.5 glyph with a fixed 0.3 barely touches it -- the
    sparkle survives -- so both are measured per image.

    The background under the glyph is estimated **per pixel**, each interior
    pixel taking the value of the nearest pixel outside the glyph's footprint.
    A single patch-wide background value is wrong whenever the sparkle
    straddles an edge, which is common (it is stamped blind): on a real sample
    it landed across a grass shadow boundary, so the outside median mixed the
    two tones and ``C`` came out at (0.63, 0.68, 1.00) -- a blue overlay.
    Un-blending white pixels as blue leaves the sparkle behind as a tinted
    stain. Comparing each pixel against the tone it actually sits on makes both
    sides of the edge agree on one alpha. Nearest-pixel beats a smooth
    inpainted fill here because content that runs *through* the glyph -- a line
    crossing it -- is sampled from its own continuation just outside, where an
    inpaint would smear it: on a synthetic stamp over a black line this
    recovers alpha to 5 decimals against inpainting's 0.3 percentage points.

    Note this estimates the *overlay*, never the artwork: the recovered pixels
    still come from inverting the blend, so line work under the glyph survives.

    * ``bg`` per pixel, from the nearest pixel outside the glyph footprint;
    * ``alpha`` as the median of ``(obs - bg)/(1 - bg)`` over interior pixels
      and channels with enough headroom (i.e. taking ``C = 1`` for that step);
    * ``C`` per channel from the median of ``(obs - bg)/alpha + bg``.

    Returns ``None`` when the patch is degenerate or the measurement is not a
    plausible white overlay -- which is also how a false positive from the
    boundary probe is rejected. The caller then leaves the image untouched."""
    height, width = source.shape[:2]
    pad = int(round(max(rx, ry) * 1.8)) + 2
    x0, x1 = max(0, int(cx) - pad), min(width, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(height, int(cy) + pad + 1)
    patch = source[y0:y1, x0:x1]
    cov = _star_coverage(height, width, cx, cy, rx, ry, exponent)[y0:y1, x0:x1]
    interior = cov >= 0.95
    if int(interior.sum()) < 8 or int((cov <= 0.0).sum()) < 8:
        return None

    background = _nearest_background(patch, cov > 0.0)

    obs = patch[interior]
    bg = background[interior]
    headroom = 1.0 - bg
    usable = headroom > _MIN_ESTIMATE_HEADROOM
    if int(usable.sum()) < 8:
        return None
    alpha = float(np.median(((obs - bg) / np.clip(headroom, _MIN_HEADROOM, None))[usable]))
    if not _MIN_PLAUSIBLE_ALPHA <= alpha <= _MAX_PLAUSIBLE_ALPHA:
        return None
    color = np.clip(np.median((obs - bg) / alpha + bg, axis=0), 0.0, 1.0).astype(np.float32)
    if float(color.min()) < _MIN_OVERLAY_WHITENESS:
        return None
    return alpha, color


def remove_gemini_watermark(
    pixels: np.ndarray,
    *,
    alpha: float | None = None,
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
    can enable this unconditionally. The glyph is located by a small windowed
    search around its fitted position so a frame resized to a different aspect
    ratio -- which drifts the sparkle off the fixed probe -- is still cleaned,
    and its opacity is estimated per image so a stronger stamp than the fitted
    one is still fully un-blended. Pass ``alpha`` to force a fixed opacity.
    """
    height, width = pixels.shape[:2]
    cx = width * center_x_frac
    cy = height * center_y_frac
    rx = width * radius_x_frac
    ry = width * radius_y_frac
    # Too small to carry the glyph.
    if rx < 2.0 or ry < 2.0:
        return pixels

    source = np.asarray(pixels, dtype=np.float32)
    det, cx, cy = _locate_watermark(
        source, cx, cy, rx, ry, exponent, half_frac=_SEARCH_HALF_FRAC, steps=_SEARCH_STEPS
    )
    if det < detect_threshold:
        return pixels

    color: np.ndarray | float = 1.0
    if alpha is None:
        estimated = _estimate_overlay(source, cx, cy, rx, ry, exponent)
        if estimated is None:
            # No measurable white overlay here: the probe fired on the
            # artwork's own edge. Guessing the fitted constant would stamp an
            # inverse sparkle onto clean art, so leave the image alone.
            return pixels
        alpha, color = estimated
    a = (_star_coverage(height, width, cx, cy, rx, ry, exponent) * alpha)[:, :, None]
    recovered = (source - color * a) / np.clip(1.0 - a, 1e-3, None)
    return np.clip(recovered, 0.0, 1.0).astype(np.float32)
