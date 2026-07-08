"""Concrete geometry value types satisfying the protocols in :mod:`.types`.

All are frozen; coordinate arrays are stored read-only (global immutability
rule, DATA_MODEL_SPEC §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array, dtype=np.float64)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Pt:
    """A 2-D point (units per context: px in the raster frame, pt in the page frame)."""

    x: float
    y: float


@dataclass(frozen=True)
class PolylineData:
    """An ordered point chain.

    ``coords`` is an (N, 2) float64 array of (x, y) rows and never repeats the
    first vertex; ``is_closed`` implies an implicit closing segment.
    """

    coords: np.ndarray
    is_closed: bool = False

    def __post_init__(self) -> None:
        coords = _readonly(self.coords)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (N, 2), got shape {coords.shape}")
        minimum = 3 if self.is_closed else 2
        if coords.shape[0] < minimum:
            raise ValueError(
                f"{'closed' if self.is_closed else 'open'} polyline needs ≥ {minimum} "
                f"points, got {coords.shape[0]}"
            )
        object.__setattr__(self, "coords", coords)


@dataclass(frozen=True)
class BezierChainData:
    """A chain of cubic Bézier segments fitted to one arc.

    ``control_points`` is (S, 4, 2) float64; consecutive segments share their
    boundary point exactly. ``arc_id`` is −1 until a stage binds the chain to
    an arc (the kernel is arc-agnostic).
    """

    control_points: np.ndarray
    arc_id: int = field(default=-1)

    def __post_init__(self) -> None:
        ctrl = _readonly(self.control_points)
        if ctrl.ndim != 3 or ctrl.shape[1:] != (4, 2) or ctrl.shape[0] < 1:
            raise ValueError(f"control_points must be (S≥1, 4, 2), got shape {ctrl.shape}")
        if not np.array_equal(ctrl[1:, 0, :], ctrl[:-1, 3, :]):
            raise ValueError("consecutive segments must share endpoints exactly")
        object.__setattr__(self, "control_points", ctrl)
