"""Physical unit conversion — the ONLY place units convert (ARCHITECTURE.md §2, §4.1)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0


class UnitConverter(ABC):
    """Converts between millimetres, points, and working pixels for one run's scale."""

    @property
    @abstractmethod
    def work_scale(self) -> float:
        """Points per working pixel for the current run (``s`` in MATH_SPEC §1.3)."""

    @abstractmethod
    def mm_to_pt(self, mm: float) -> float:
        """Convert millimetres to points."""

    @abstractmethod
    def pt_to_mm(self, pt: float) -> float:
        """Convert points to millimetres."""

    @abstractmethod
    def px_to_pt(self, px: float) -> float:
        """Convert working pixels to points using the run's work scale."""

    @abstractmethod
    def pt_to_px(self, pt: float) -> float:
        """Convert points to working pixels using the run's work scale."""


class PageUnits(UnitConverter):
    """Default converter for one run.

    ``work_scale`` is the single scale factor Φ applies (MATH_SPEC §1.3);
    constructing this object is the only sanctioned way to convert px↔pt.
    """

    def __init__(self, work_scale: float) -> None:
        """``work_scale``: points per working pixel; must be finite and positive."""
        if not math.isfinite(work_scale) or work_scale <= 0.0:
            raise ValueError(f"work_scale must be finite and positive, got {work_scale}")
        self._work_scale = work_scale

    @property
    def work_scale(self) -> float:
        return self._work_scale

    def mm_to_pt(self, mm: float) -> float:
        return mm / MM_PER_INCH * PT_PER_INCH

    def pt_to_mm(self, pt: float) -> float:
        return pt / PT_PER_INCH * MM_PER_INCH

    def px_to_pt(self, px: float) -> float:
        return px * self._work_scale

    def pt_to_px(self, pt: float) -> float:
        return pt / self._work_scale
