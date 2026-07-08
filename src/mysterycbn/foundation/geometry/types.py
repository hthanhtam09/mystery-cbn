"""Geometry value types spoken by all internal interfaces (ARCHITECTURE.md §1.1, §6)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Point(Protocol):
    """A 2-D point in the current domain's units."""

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...


@runtime_checkable
class Polyline(Protocol):
    """An ordered open chain of points, stored as an (N, 2) float array."""

    @property
    def coords(self) -> np.ndarray: ...

    @property
    def is_closed(self) -> bool: ...


@runtime_checkable
class Arc(Protocol):
    """A shared boundary between exactly two labels in the arc graph."""

    @property
    def arc_id(self) -> int: ...

    @property
    def polyline(self) -> Polyline: ...

    @property
    def left_label(self) -> int: ...

    @property
    def right_label(self) -> int: ...


@runtime_checkable
class Face(Protocol):
    """A closed region defined as an ordered walk of directed arcs."""

    @property
    def face_id(self) -> int: ...

    @property
    def label(self) -> int: ...

    @property
    def arc_walk(self) -> Sequence[tuple[int, bool]]:
        """Ordered (arc_id, reversed) pairs whose concatenation closes the face."""
        ...


@runtime_checkable
class BezierChain(Protocol):
    """A G1-continuous chain of cubic Bézier segments fitted to one arc."""

    @property
    def arc_id(self) -> int: ...

    @property
    def control_points(self) -> np.ndarray:
        """(S, 4, 2) array: S segments, four control points each."""
        ...
