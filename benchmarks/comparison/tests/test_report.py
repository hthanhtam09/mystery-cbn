"""Tests for the top-level ComparisonReport assembly."""

from __future__ import annotations

import json

from benchmarks.comparison.report import compare_category, compare_examples
from benchmarks.comparison.runner import PRESETS
from benchmarks.datasets.metadata_schema import CATEGORIES


def test_compare_examples_covers_every_category() -> None:
    report = compare_examples()
    assert {fx.category for fx in report.fixtures} == set(CATEGORIES)


def test_compare_examples_runs_every_preset_per_fixture() -> None:
    report = compare_examples()
    for fx in report.fixtures:
        assert [s.preset for s in fx.snapshots] == list(PRESETS)


def test_compare_category_scopes_to_one_category() -> None:
    report = compare_category("architecture")
    assert report.fixtures
    assert all(fx.category == "architecture" for fx in report.fixtures)


def test_report_to_dict_is_json_serializable() -> None:
    report = compare_examples()
    assert json.dumps(report.to_dict())


def test_summary_counts_match_fixtures_and_recommendations() -> None:
    report = compare_examples()
    d = report.to_dict()
    summary = d["summary"]
    assert isinstance(summary, dict)
    assert summary["fixture_count"] == len(report.fixtures)
    assert summary["recommendation_count"] == len(report.all_recommendations)
    assert summary["caution_count"] == len(report.cautions)


def test_run_id_is_unique_per_call() -> None:
    a = compare_category("food")
    b = compare_category("food")
    assert a.run_id != b.run_id
