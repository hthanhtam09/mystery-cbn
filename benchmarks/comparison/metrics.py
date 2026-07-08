"""Technical-quality metrics for the Sprint 24 comparison framework, beyond
what Sprint 23's ``validate/quality_metrics.py`` already computes.

Average edge length and region size distribution have no prior
implementation anywhere in the repo (confirmed absent, not merely
unexported) -- both are independently re-derived here from the same final
``curve_set``/``label_plan`` artifacts the Sprint 23 validator reads,
following the same "flatten and measure" convention rather than trusting
any construction-time value. Label density is new too: labels per unit
printable page area, a proxy for how visually busy the final page is.

No engine, stage, or rendering code is touched -- purely observational
measurement, exactly like ``quality_metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mysterycbn.model.layout import LabelPlan
from mysterycbn.model.vector import CurveSet
from mysterycbn.stages.vector.arcgraph import content_box_pt
from mysterycbn.validate.common import face_area_pt2, flatten_face_rings

_FLATTEN_MM = 0.1
_MM_TO_PT = 72.0 / 25.4
_PT2_PER_CM2 = _MM_TO_PT**2 * 100.0


@dataclass(frozen=True)
class RegionSizeDistribution:
    """Summary statistics of face area (mm²) across a ``CurveSet``."""

    count: int
    min_mm2: float
    max_mm2: float
    mean_mm2: float
    median_mm2: float
    stdev_mm2: float
    p10_mm2: float
    p90_mm2: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "min_mm2": self.min_mm2,
            "max_mm2": self.max_mm2,
            "mean_mm2": self.mean_mm2,
            "median_mm2": self.median_mm2,
            "stdev_mm2": self.stdev_mm2,
            "p10_mm2": self.p10_mm2,
            "p90_mm2": self.p90_mm2,
        }


def _face_areas_mm2(curve_set: CurveSet, tolerance_pt: float) -> np.ndarray:
    pt2_per_mm2 = _MM_TO_PT**2
    areas = [
        abs(face_area_pt2(face, curve_set, tolerance_pt)) / pt2_per_mm2 for face in curve_set.faces
    ]
    return np.asarray(areas, dtype=np.float64)


def region_size_distribution(
    curve_set: CurveSet, *, tolerance_mm: float = _FLATTEN_MM
) -> RegionSizeDistribution:
    """Per-face area (mm²) distribution across the final vector geometry.

    Complements Sprint 23's QM-13 (a bare count) and QM-14 (a bare mean
    compactness) with the shape of the size distribution itself -- e.g. two
    fixtures can share the same region count and mean compactness while one
    has a long tail of tiny slivers and the other doesn't, and neither
    QM-13 nor QM-14 alone shows that.
    """
    tolerance_pt = tolerance_mm * _MM_TO_PT
    areas = _face_areas_mm2(curve_set, tolerance_pt)
    if areas.size == 0:
        return RegionSizeDistribution(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return RegionSizeDistribution(
        count=int(areas.size),
        min_mm2=float(areas.min()),
        max_mm2=float(areas.max()),
        mean_mm2=float(areas.mean()),
        median_mm2=float(np.median(areas)),
        stdev_mm2=float(areas.std()) if areas.size > 1 else 0.0,
        p10_mm2=float(np.percentile(areas, 10)),
        p90_mm2=float(np.percentile(areas, 90)),
    )


def average_edge_length_mm(curve_set: CurveSet, *, tolerance_mm: float = _FLATTEN_MM) -> float:
    """Mean length (mm) of every flattened boundary edge across every face's
    rings, independently re-derived by flattening each face's outer/hole
    rings (not reusing any construction-time arc length). A companion
    measure to QM-08's curvature *energy*: this is the underlying edge
    scale that energy is normalized against, and is useful on its own to
    compare how finely two configurations tessellate boundaries."""
    tolerance_pt = tolerance_mm * _MM_TO_PT
    total_length_pt = 0.0
    total_edges = 0
    for face in curve_set.faces:
        for ring in flatten_face_rings(face, curve_set, tolerance_pt):
            edges = np.roll(ring, -1, axis=0) - ring
            lengths = np.linalg.norm(edges, axis=1)
            total_length_pt += float(np.sum(lengths))
            total_edges += len(lengths)
    if total_edges == 0:
        return 0.0
    return (total_length_pt / total_edges) / _MM_TO_PT


def label_density_per_cm2(label_plan: LabelPlan, *, page_mm: tuple[float, float, float]) -> float:
    """Printed labels per cm² of printable page area (content box, margins
    excluded) -- a proxy for how visually busy/cluttered the final page is.
    Uses the same content-box definition ``ArcGraphStage``'s Φ letterbox
    and every renderer already agree on (``stages.vector.arcgraph
    .content_box_pt``), so this measures against the same printable area
    the engine itself targets, not an approximation."""
    _, _, width_pt, height_pt = content_box_pt(page_mm)
    area_cm2 = (width_pt * height_pt) / _PT2_PER_CM2
    if area_cm2 <= 0.0:
        return 0.0
    return len(label_plan.labels) / area_cm2
