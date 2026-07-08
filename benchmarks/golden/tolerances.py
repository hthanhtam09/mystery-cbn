"""Tolerance configuration for golden comparisons (docs/GOLDEN_TEST_STANDARDS.md §4).

Centralizes every pass/fail threshold used by ``compare.py`` so tolerances
are declared once, are overridable per-call (e.g. a looser band for a known
noisy category), and are recorded in the report for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenTolerances:
    """Thresholds for the three comparison axes (BENCHMARK_SPEC.md §4.2 for
    the perceptual/structural pair; topology tolerance is new in Sprint 21).
    """

    ssim_min: float = 0.97
    """Perceptual: minimum acceptable luminance SSIM of the solved preview."""

    segment_count_tolerance: float = 0.10
    """SVG structural: per-arc Bezier-segment-count relative tolerance."""

    coord_rms_max_mm: float = 0.3
    """SVG structural: max coordinate RMS delta (mm) when byte hash differs."""

    topology_region_count_tolerance: float = 0.0
    """Topology: relative tolerance on face/region count delta. 0.0 means
    an exact match is required -- region count is analytic ground truth
    for a deterministic pipeline on a fixed fixture (BENCHMARK_SPEC.md §3)."""

    topology_arc_count_tolerance: float = 0.0
    """Topology: relative tolerance on arc count delta."""

    def to_dict(self) -> dict[str, float]:
        return {
            "ssim_min": self.ssim_min,
            "segment_count_tolerance": self.segment_count_tolerance,
            "coord_rms_max_mm": self.coord_rms_max_mm,
            "topology_region_count_tolerance": self.topology_region_count_tolerance,
            "topology_arc_count_tolerance": self.topology_arc_count_tolerance,
        }


DEFAULT_TOLERANCES = GoldenTolerances()
