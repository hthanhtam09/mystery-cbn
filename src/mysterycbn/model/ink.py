"""Ink-line artifacts: preserved thin dark line work (whiskers, fine
line-art) that color quantization would otherwise lose.

Ink is carried as a *render-only overlay*, never as regions or palette
entries: a reserved-label / region representation would create thousands of
hairline faces that fail the fidelity/topology/printability gates and demand
a legend color + number. Instead the detector emits a boolean ``InkMask`` at
working-raster resolution, which the vectorizer turns into centerline
polylines in page-point space (``InkOverlay``). The three renderers draw the
overlay as plain black strokes on top of the coloring page; the four
canonical validators never see it (they read only graph/vector/label/palette
artifacts), so ink is fidelity-safe by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mysterycbn.model._utils import readonly, require
from mysterycbn.model.records import Provenance


@dataclass(frozen=True)
class InkMask:
    """Boolean (H, W) mask of detected ink pixels, at the working-raster
    resolution (same shape as the ``LabelMap``/``component_map``). An all-False
    mask is the disabled / no-detection case."""

    mask: np.ndarray
    provenance: Provenance

    def __post_init__(self) -> None:
        m = readonly(self.mask, np.bool_)
        require(m.ndim == 2, f"ink mask must be 2-D, got {m.shape}")
        object.__setattr__(self, "mask", m)

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.mask.shape),
            "ink_px": int(self.mask.sum()),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class InkOverlay:
    """Ink line work as centerline polylines in page points (x, y), plus the
    black stroke width in pt. ``polylines=()`` is the empty (disabled) overlay;
    renderers still emit the (empty) ink layer to keep the layer set fixed."""

    polylines: tuple[np.ndarray, ...]
    stroke_pt: float
    provenance: Provenance

    def __post_init__(self) -> None:
        require(self.stroke_pt >= 0.0, "stroke_pt must be ≥ 0")
        cleaned = tuple(readonly(p, np.float64) for p in self.polylines)
        for p in cleaned:
            require(p.ndim == 2 and p.shape[1] == 2, f"polyline must be (N, 2), got {p.shape}")
            require(p.shape[0] >= 2, "polyline needs ≥ 2 points")
        object.__setattr__(self, "polylines", cleaned)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_polylines": len(self.polylines),
            "stroke_pt": self.stroke_pt,
            "provenance": self.provenance.to_dict(),
        }
