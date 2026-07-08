"""Dataset loaders (docs/DATASET_STANDARDS.md §5): the public API for
fetching one or many fixtures by id, category, or tier.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from benchmarks.datasets.metadata_schema import CATEGORIES, FixtureMetadata
from benchmarks.datasets.registry import (
    DATASET_VERSION,
    available_example_ids,
    available_fixture_ids,
    available_golden_ids,
    fixture_ids_for_category,
    generate_labels,
    get_entry,
)


@dataclass(frozen=True)
class DatasetFixture:
    """A generated label map plus its declared metadata."""

    labels: np.ndarray
    metadata: FixtureMetadata

    @property
    def fixture_id(self) -> str:
        return self.metadata.fixture_id

    @property
    def category(self) -> str:
        return self.metadata.category

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.labels.tobytes()).hexdigest()


def load_fixture(fixture_id: str) -> DatasetFixture:
    """Deterministically load one fixture by id."""
    entry = get_entry(fixture_id)
    labels = generate_labels(fixture_id)
    return DatasetFixture(labels=labels, metadata=entry.metadata())


def load_examples() -> tuple[DatasetFixture, ...]:
    """One fixture per category -- the small demo/docs ladder."""
    return tuple(load_fixture(fid) for fid in available_example_ids())


def load_category(category: str) -> tuple[DatasetFixture, ...]:
    """Every fixture (all tiers/difficulties) for one category."""
    return tuple(load_fixture(fid) for fid in fixture_ids_for_category(category))


def load_all() -> tuple[DatasetFixture, ...]:
    """Every registered fixture, in deterministic id order."""
    return tuple(load_fixture(fid) for fid in available_fixture_ids())


def load_golden() -> tuple[DatasetFixture, ...]:
    """The frozen per-category subset blessed for golden comparison."""
    return tuple(load_fixture(fid) for fid in available_golden_ids())


def dataset_manifest() -> dict[str, dict[str, object]]:
    """Content-hash manifest of every fixture (docs/DATASET_STANDARDS.md §6),
    the dataset analogue of ``benchmarks/framework/fixtures.fixture_manifest``.
    """
    manifest: dict[str, dict[str, object]] = {}
    for fixture_id in available_fixture_ids():
        fx = load_fixture(fixture_id)
        manifest[fixture_id] = {
            "category": fx.category,
            "content_hash": fx.content_hash,
            "shape": list(fx.labels.shape),
            **fx.metadata.to_dict(),
            "dataset_version": DATASET_VERSION,
        }
    return manifest


__all__ = [
    "CATEGORIES",
    "DatasetFixture",
    "dataset_manifest",
    "load_all",
    "load_category",
    "load_examples",
    "load_fixture",
    "load_golden",
]
