"""Geometry kernel API: pure functions over geometry types (ARCHITECTURE.md §6, §15)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from mysterycbn.foundation.geometry.types import BezierChain, Point, Polyline


class GeometryKernel(ABC):
    """The single home of the system's hardest math: crack tracing, arc graphs,
    simplification, Bézier fitting, polylabel, and robust predicates.

    Implementations must be deterministic and pass the shared property-test suite.
    """

    @abstractmethod
    def trace_cracks(self, label_map: np.ndarray) -> Sequence[Polyline]:
        """Extract crack-boundary polylines between differing labels of an int32 label raster."""

    @abstractmethod
    def simplify_polyline(self, polyline: Polyline, tolerance: float) -> Polyline:
        """Topology-preserving Visvalingam–Whyatt simplification with sidedness guard."""

    @abstractmethod
    def fit_bezier_chain(
        self, polyline: Polyline, max_error: float, corner_angle_deg: float
    ) -> BezierChain:
        """Fit a G1 cubic Bézier chain preserving corners above the angle threshold."""

    @abstractmethod
    def pole_of_inaccessibility(self, boundary: Polyline) -> tuple[Point, float]:
        """Return the interior point farthest from the boundary and its clearance radius."""

    @abstractmethod
    def inscribed_circle_diameter(self, boundary: Polyline) -> float:
        """Diameter of the largest circle inscribed in the closed boundary."""

    @abstractmethod
    def is_watertight(self, polylines: Sequence[Polyline], page_area: float) -> bool:
        """Re-prove invariant I3: the polylines form a gap- and overlap-free planar partition."""
