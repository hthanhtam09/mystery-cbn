"""Unit tests for the rule-based recommendation engine, using synthetic
``QualitySnapshot``s so each rule is exercised deterministically without
running the real pipeline."""

from __future__ import annotations

from benchmarks.comparison.evaluate import QualitySnapshot
from benchmarks.comparison.metrics import RegionSizeDistribution
from benchmarks.comparison.recommend import recommend_across_presets, recommend_for_pair

_EMPTY_DIST = RegionSizeDistribution(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _snapshot(preset: str, **overrides: object) -> QualitySnapshot:
    base = dict(
        fixture_id="F-test",
        category="animals",
        preset=preset,
        n_colors=10,
        d_min_mm=3.5,
        region_count=100,
        mean_compactness=0.5,
        boundary_smoothness=0.02,
        average_edge_length_mm=1.0,
        region_size_distribution=_EMPTY_DIST,
        label_density_per_cm2=0.1,
        printability_score=1.0,
        tiny_region_pct=0.0,
        quality_metrics={},
    )
    base.update(overrides)
    return QualitySnapshot(**base)  # type: ignore[arg-type]


def test_clean_pair_produces_no_recommendations() -> None:
    a = _snapshot("easy")
    b = _snapshot("medium")
    assert recommend_for_pair(a, b) == ()


def test_region_count_growth_triggers_info() -> None:
    a = _snapshot("easy", region_count=100)
    b = _snapshot("medium", region_count=140)
    recs = recommend_for_pair(a, b)
    assert any(r.severity == "info" and "region count" in r.message for r in recs)


def test_smoothness_regression_triggers_caution() -> None:
    a = _snapshot("easy", boundary_smoothness=0.02)
    b = _snapshot("medium", boundary_smoothness=0.05)
    recs = recommend_for_pair(a, b)
    assert any(r.severity == "caution" and "smoothness" in r.message for r in recs)


def test_compactness_drop_triggers_caution() -> None:
    a = _snapshot("easy", mean_compactness=0.5)
    b = _snapshot("medium", mean_compactness=0.3)
    recs = recommend_for_pair(a, b)
    assert any(r.severity == "caution" and "compactness" in r.message for r in recs)


def test_tiny_region_increase_triggers_caution() -> None:
    a = _snapshot("easy", tiny_region_pct=0.0)
    b = _snapshot("medium", tiny_region_pct=10.0)
    recs = recommend_for_pair(a, b)
    assert any(r.severity == "caution" and "tiny-region" in r.message for r in recs)


def test_label_density_increase_triggers_info() -> None:
    a = _snapshot("easy", label_density_per_cm2=0.1)
    b = _snapshot("medium", label_density_per_cm2=0.2)
    recs = recommend_for_pair(a, b)
    assert any(r.severity == "info" and "label density" in r.message for r in recs)


def test_edge_length_collapse_triggers_info() -> None:
    a = _snapshot("easy", average_edge_length_mm=2.0)
    b = _snapshot("medium", average_edge_length_mm=0.8)
    recs = recommend_for_pair(a, b)
    assert any(r.severity == "info" and "edge length" in r.message for r in recs)


def test_printability_near_floor_triggers_caution() -> None:
    a = _snapshot("easy", printability_score=0.9)
    b = _snapshot("medium", printability_score=0.55)
    recs = recommend_for_pair(a, b)
    assert any(r.severity == "caution" and "printability" in r.message for r in recs)


def test_recommendations_carry_fixture_and_preset_identity() -> None:
    a = _snapshot("easy", region_count=100)
    b = _snapshot("hard", region_count=200)
    recs = recommend_for_pair(a, b)
    assert recs
    for r in recs:
        assert r.fixture_id == "F-test"
        assert r.category == "animals"
        assert r.from_preset == "easy"
        assert r.to_preset == "hard"


def test_recommend_across_presets_covers_consecutive_pairs_only() -> None:
    snapshots = (
        _snapshot("easy", region_count=100),
        _snapshot("medium", region_count=100),
        _snapshot("hard", region_count=200),
    )
    recs = recommend_across_presets(snapshots)
    # Only medium->hard should fire the region-count-growth rule; easy->medium is flat.
    assert all(r.from_preset == "medium" and r.to_preset == "hard" for r in recs)


def test_no_rule_ever_raises_on_zero_valued_snapshots() -> None:
    a = _snapshot(
        "easy",
        region_count=0,
        mean_compactness=0.0,
        boundary_smoothness=0.0,
        average_edge_length_mm=0.0,
        label_density_per_cm2=0.0,
        printability_score=0.5,
        tiny_region_pct=0.0,
    )
    b = _snapshot(
        "medium",
        region_count=0,
        mean_compactness=0.0,
        boundary_smoothness=0.0,
        average_edge_length_mm=0.0,
        label_density_per_cm2=0.0,
        printability_score=0.5,
        tiny_region_pct=0.0,
    )
    assert recommend_for_pair(a, b) == ()
