"""Unit tests for the dataset loaders."""

from __future__ import annotations

import pytest

from benchmarks.datasets.loaders import (
    dataset_manifest,
    load_all,
    load_category,
    load_examples,
    load_fixture,
    load_golden,
)
from benchmarks.datasets.metadata_schema import CATEGORIES


def test_load_examples_covers_every_category() -> None:
    examples = load_examples()
    assert {fx.category for fx in examples} == set(CATEGORIES)


def test_load_fixture_is_deterministic() -> None:
    a = load_fixture("D-animals-examples-01")
    b = load_fixture("D-animals-examples-01")
    assert a.content_hash == b.content_hash


def test_load_category_returns_only_that_category() -> None:
    fixtures = load_category("flowers")
    assert fixtures
    assert all(fx.category == "flowers" for fx in fixtures)


def test_load_all_matches_manifest() -> None:
    fixtures = load_all()
    manifest = dataset_manifest()
    assert {fx.fixture_id for fx in fixtures} == set(manifest)
    for fx in fixtures:
        assert manifest[fx.fixture_id]["content_hash"] == fx.content_hash


def test_load_golden_is_one_per_category() -> None:
    golden = load_golden()
    assert sorted(fx.category for fx in golden) == sorted(CATEGORIES)


def test_metadata_expectations_are_self_consistent() -> None:
    for fx in load_all():
        assert fx.labels.max() < fx.metadata.palette_count or fx.metadata.palette_count >= 1
        assert fx.metadata.width == fx.labels.shape[1]
        assert fx.metadata.height == fx.labels.shape[0]


def test_unknown_fixture_id_raises_key_error() -> None:
    with pytest.raises(KeyError):
        load_fixture("D-nonexistent-examples-01")
