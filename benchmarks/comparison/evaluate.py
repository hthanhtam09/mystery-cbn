"""Assembles the full Sprint 24 technical-quality snapshot for one
``PresetRun`` -- region count, compactness, boundary smoothness, average
edge length, region size distribution, label density, printability --
combining Sprint 23's ``validate/quality_metrics.py`` (reused, not
re-derived) with this package's net-new metrics (``metrics.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.comparison.metrics import (
    RegionSizeDistribution,
    average_edge_length_mm,
    label_density_per_cm2,
    region_size_distribution,
)
from benchmarks.comparison.runner import PresetRun
from mysterycbn.model.layout import LabelPlan
from mysterycbn.model.records import Palette
from mysterycbn.model.reports import MetricResult
from mysterycbn.model.vector import CurveSet
from mysterycbn.validate.printability import validate_printability
from mysterycbn.validate.quality_metrics import compute_quality_metrics


@dataclass(frozen=True)
class QualitySnapshot:
    """Every Sprint 24 technical-quality measurement for one (fixture,
    preset) run, in one place."""

    fixture_id: str
    category: str
    preset: str
    n_colors: int
    d_min_mm: float
    region_count: int
    mean_compactness: float
    boundary_smoothness: float
    average_edge_length_mm: float
    region_size_distribution: RegionSizeDistribution
    label_density_per_cm2: float
    printability_score: float
    tiny_region_pct: float
    quality_metrics: dict[str, MetricResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "category": self.category,
            "preset": self.preset,
            "n_colors": self.n_colors,
            "d_min_mm": self.d_min_mm,
            "region_count": self.region_count,
            "mean_compactness": self.mean_compactness,
            "boundary_smoothness": self.boundary_smoothness,
            "average_edge_length_mm": self.average_edge_length_mm,
            "region_size_distribution": self.region_size_distribution.to_dict(),
            "label_density_per_cm2": self.label_density_per_cm2,
            "printability_score": self.printability_score,
            "tiny_region_pct": self.tiny_region_pct,
            "quality_metrics": {k: v.to_dict() for k, v in self.quality_metrics.items()},
        }


def evaluate_preset_run(
    preset_run: PresetRun, *, page_mm: tuple[float, float, float]
) -> QualitySnapshot:
    """Compute the full technical-quality snapshot for one preset run.
    Re-runs ``validate_printability`` to get a fresh ``tiny_region_pct`` at
    this preset's ``d_min_mm`` (the pipeline's own validation pass already
    ran once at construction time with the same settings; re-deriving here
    keeps this module decoupled from having to thread that result through
    ``PresetRun``, and printability is inexpensive to recompute)."""
    ctx = preset_run.run.ctx
    curve_set = ctx.get("curve_set")
    palette = ctx.get("palette")
    assert isinstance(curve_set, CurveSet)
    assert isinstance(palette, Palette)

    printability_report = validate_printability(ctx, d_min_mm=preset_run.d_min_mm)
    printability_metrics = dict(printability_report.metrics)

    # Re-fetch after validate_printability: a leader-demotion repair
    # rebinds "label_plan" to a new LabelPlan in ctx (printability.py),
    # so the pre-repair reference above would understate label density
    # for any face demoted to a leader.
    label_plan = ctx.get("label_plan")
    assert isinstance(label_plan, LabelPlan)

    quality_metrics = compute_quality_metrics(ctx, printability_metrics=printability_metrics)

    return QualitySnapshot(
        fixture_id=preset_run.fixture.fixture_id,
        category=preset_run.fixture.category,
        preset=preset_run.preset,
        n_colors=preset_run.n_colors,
        d_min_mm=preset_run.d_min_mm,
        region_count=len(curve_set.faces),
        mean_compactness=quality_metrics["QM-14"].value,
        boundary_smoothness=quality_metrics["QM-08"].value,
        average_edge_length_mm=average_edge_length_mm(curve_set),
        region_size_distribution=region_size_distribution(curve_set),
        label_density_per_cm2=label_density_per_cm2(label_plan, page_mm=page_mm),
        printability_score=quality_metrics["printability_score"].value,
        tiny_region_pct=printability_metrics["tiny_region_pct"],
        quality_metrics=quality_metrics,
    )
