"""Tests for QualitySnapshot assembly."""

from __future__ import annotations

import json

from benchmarks.comparison.evaluate import evaluate_preset_run
from benchmarks.comparison.runner import run_fixture_under_preset
from benchmarks.datasets.loaders import load_fixture
from benchmarks.framework.pipeline import PAGE_MM


def test_snapshot_fields_are_populated() -> None:
    fx = load_fixture("D-animals-examples-01")
    pr = run_fixture_under_preset(fx, "medium")
    snap = evaluate_preset_run(pr, page_mm=PAGE_MM)

    assert snap.fixture_id == fx.fixture_id
    assert snap.category == fx.category
    assert snap.preset == "medium"
    assert snap.region_count > 0
    assert snap.region_size_distribution.count == snap.region_count
    assert snap.label_density_per_cm2 > 0.0
    assert 0.0 <= snap.printability_score <= 1.0
    assert 0.0 <= snap.tiny_region_pct <= 100.0


def test_snapshot_to_dict_is_json_serializable() -> None:
    fx = load_fixture("D-flowers-examples-01")
    pr = run_fixture_under_preset(fx, "easy")
    snap = evaluate_preset_run(pr, page_mm=PAGE_MM)
    assert json.dumps(snap.to_dict())


def test_snapshot_region_count_matches_quality_metrics_qm13() -> None:
    fx = load_fixture("D-cartoons-examples-01")
    pr = run_fixture_under_preset(fx, "hard")
    snap = evaluate_preset_run(pr, page_mm=PAGE_MM)
    assert snap.region_count == int(snap.quality_metrics["QM-13"].value)
