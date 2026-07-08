"""Unit tests for the category-specific synthetic generators."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.datasets.generators import generate_category_labels
from benchmarks.datasets.metadata_schema import CATEGORIES


@pytest.mark.parametrize("category", CATEGORIES)
def test_generator_is_deterministic(category: str) -> None:
    a = generate_category_labels(category, seed=1, width=64, height=64, k=6)
    b = generate_category_labels(category, seed=1, width=64, height=64, k=6)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("category", CATEGORIES)
def test_generator_shape_and_dtype(category: str) -> None:
    labels = generate_category_labels(category, seed=0, width=48, height=32, k=4)
    assert labels.shape == (32, 48)
    assert labels.dtype == np.int32
    assert labels.min() >= 0


def test_unknown_category_raises() -> None:
    with pytest.raises(KeyError, match="unknown category"):
        generate_category_labels("dinosaurs", seed=0, width=8, height=8, k=2)
