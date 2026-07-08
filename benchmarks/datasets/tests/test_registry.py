"""Unit tests for the fixture registry."""

from __future__ import annotations

import pytest

from benchmarks.datasets.metadata_schema import CATEGORIES
from benchmarks.datasets.registry import (
    available_example_ids,
    available_fixture_ids,
    available_golden_ids,
    fixture_ids_for_category,
    generate_labels,
    get_entry,
)


def test_every_category_has_an_example() -> None:
    example_categories = {get_entry(fid).category for fid in available_example_ids()}
    assert example_categories == set(CATEGORIES)


def test_every_category_has_three_difficulties() -> None:
    for category in CATEGORIES:
        ids = fixture_ids_for_category(category)
        difficulties = {get_entry(fid).difficulty for fid in ids if "datasets" in fid}
        assert difficulties == {"easy", "medium", "hard"}


def test_golden_subset_is_one_per_category() -> None:
    golden_categories = [get_entry(fid).category for fid in available_golden_ids()]
    assert sorted(golden_categories) == sorted(CATEGORIES)


def test_unknown_fixture_raises() -> None:
    with pytest.raises(KeyError, match="unknown fixture"):
        get_entry("D-does-not-exist")


def test_unknown_category_raises() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        fixture_ids_for_category("dinosaurs")


def test_generate_labels_matches_declared_shape() -> None:
    for fixture_id in available_fixture_ids():
        entry = get_entry(fixture_id)
        labels = generate_labels(fixture_id)
        assert labels.shape == (entry.height, entry.width)
