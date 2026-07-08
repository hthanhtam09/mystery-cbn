"""Assembles the Sprint 24 technical-quality comparison report: every
dataset fixture, run under every difficulty preset, evaluated
(``evaluate.py``) and cross-preset recommendations generated
(``recommend.py``) -- one JSON-serializable ``ComparisonReport`` per run,
following the same run_id/timestamp/summary conventions as
``benchmarks/golden/report.py``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from benchmarks.comparison.evaluate import QualitySnapshot, evaluate_preset_run
from benchmarks.comparison.recommend import Recommendation, recommend_across_presets
from benchmarks.comparison.runner import (
    PRESETS,
    PresetRun,
    run_category_across_presets,
    run_examples_across_presets,
    run_full_dataset_across_presets,
)
from benchmarks.datasets.loaders import DatasetFixture
from benchmarks.framework.pipeline import PAGE_MM
from mysterycbn import __version__ as ENGINE_VERSION


@dataclass(frozen=True)
class FixtureComparison:
    """One fixture's snapshots across every preset it was run under, plus
    the cross-preset recommendations derived from them."""

    fixture_id: str
    category: str
    snapshots: tuple[QualitySnapshot, ...]
    recommendations: tuple[Recommendation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "category": self.category,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


@dataclass(frozen=True)
class ComparisonReport:
    """The full Sprint 24 report: every fixture's cross-preset comparison."""

    run_id: str
    engine_version: str
    timestamp: float
    presets: tuple[str, ...]
    fixtures: tuple[FixtureComparison, ...]

    @property
    def all_recommendations(self) -> tuple[Recommendation, ...]:
        return tuple(r for fx in self.fixtures for r in fx.recommendations)

    @property
    def cautions(self) -> tuple[Recommendation, ...]:
        return tuple(r for r in self.all_recommendations if r.severity == "caution")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "engine_version": self.engine_version,
            "timestamp": self.timestamp,
            "presets": list(self.presets),
            "fixtures": [fx.to_dict() for fx in self.fixtures],
            "summary": {
                "fixture_count": len(self.fixtures),
                "recommendation_count": len(self.all_recommendations),
                "caution_count": len(self.cautions),
            },
        }


def _evaluate_fixture(
    fx: DatasetFixture, preset_runs: tuple[PresetRun, ...], *, page_mm: tuple[float, float, float]
) -> FixtureComparison:
    snapshots = tuple(evaluate_preset_run(pr, page_mm=page_mm) for pr in preset_runs)
    recommendations = recommend_across_presets(snapshots)
    return FixtureComparison(
        fixture_id=fx.fixture_id,
        category=fx.category,
        snapshots=snapshots,
        recommendations=recommendations,
    )


def _build_report(
    pairs: Iterable[tuple[DatasetFixture, tuple[PresetRun, ...]]],
    *,
    presets: tuple[str, ...],
    page_mm: tuple[float, float, float],
) -> ComparisonReport:
    fixtures = tuple(
        _evaluate_fixture(fx, preset_runs, page_mm=page_mm) for fx, preset_runs in pairs
    )
    return ComparisonReport(
        run_id=uuid.uuid4().hex[:12],
        engine_version=ENGINE_VERSION,
        timestamp=time.time(),
        presets=presets,
        fixtures=fixtures,
    )


def compare_examples(
    *, presets: tuple[str, ...] = PRESETS, page_mm: tuple[float, float, float] = PAGE_MM
) -> ComparisonReport:
    """The small one-per-category example ladder, compared across presets."""
    return _build_report(
        run_examples_across_presets(presets=presets), presets=presets, page_mm=page_mm
    )


def compare_category(
    category: str,
    *,
    presets: tuple[str, ...] = PRESETS,
    page_mm: tuple[float, float, float] = PAGE_MM,
) -> ComparisonReport:
    """Every tier/difficulty fixture in one category, compared across presets."""
    return _build_report(
        run_category_across_presets(category, presets=presets), presets=presets, page_mm=page_mm
    )


def compare_full_dataset(
    *, presets: tuple[str, ...] = PRESETS, page_mm: tuple[float, float, float] = PAGE_MM
) -> ComparisonReport:
    """Every registered dataset fixture, compared across presets."""
    return _build_report(
        run_full_dataset_across_presets(presets=presets), presets=presets, page_mm=page_mm
    )
