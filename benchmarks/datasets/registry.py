"""Fixture registry: the single source of truth mapping fixture ids to
generator parameters and declared metadata (docs/DATASET_STANDARDS.md §4).

Three tiers, mirroring tests/golden vs benchmarks/{perf,quality}:

- ``examples``: one small fixture per category, for docs/demos/smoke checks.
- ``datasets``: the full per-category ladder (varied sizes/difficulties)
  used by quality/perf benchmark suites.
- ``golden``: the frozen subset blessed for golden comparison
  (BENCHMARK_SPEC.md §4), a name-only pointer into ``datasets``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.datasets.generators import generate_category_labels
from benchmarks.datasets.metadata_schema import CATEGORIES, Difficulty, FixtureMetadata

DATASET_VERSION = 1


@dataclass(frozen=True)
class RegistryEntry:
    fixture_id: str
    category: str
    seed: int
    width: int
    height: int
    difficulty: Difficulty
    palette_count: int
    expected_region_count: int
    expected_printability: float

    def metadata(self) -> FixtureMetadata:
        return FixtureMetadata(
            fixture_id=self.fixture_id,
            category=self.category,  # type: ignore[arg-type]
            width=self.width,
            height=self.height,
            difficulty=self.difficulty,
            palette_count=self.palette_count,
            expected_region_count=self.expected_region_count,
            expected_printability=self.expected_printability,
        )


def _entry(
    category: str,
    tier: str,
    variant: str,
    *,
    seed: int,
    width: int,
    height: int,
    difficulty: Difficulty,
    palette_count: int,
    expected_region_count: int,
    expected_printability: float,
) -> RegistryEntry:
    return RegistryEntry(
        fixture_id=f"D-{category}-{tier}-{variant}",
        category=category,
        seed=seed,
        width=width,
        height=height,
        difficulty=difficulty,
        palette_count=palette_count,
        expected_region_count=expected_region_count,
        expected_printability=expected_printability,
    )


# One "examples" fixture per category: small, fast, used for demos/docs.
_EXAMPLES: dict[str, RegistryEntry] = {}
for i, cat in enumerate(CATEGORIES):
    entry = _entry(
        cat,
        "examples",
        "01",
        seed=i,
        width=128,
        height=128,
        difficulty="easy",
        palette_count=6,
        expected_region_count=8,
        expected_printability=0.9,
    )
    _EXAMPLES[entry.fixture_id] = entry

# Full per-category ladder: easy/medium/hard variant per category.
_DATASETS: dict[str, RegistryEntry] = {}
_DIFFICULTY_PARAMS: dict[Difficulty, dict[str, object]] = {
    "easy": {"width": 256, "height": 256, "palette_count": 6, "expected_region_count": 10},
    "medium": {"width": 512, "height": 512, "palette_count": 10, "expected_region_count": 24},
    "hard": {"width": 768, "height": 768, "palette_count": 16, "expected_region_count": 48},
}
for cat_i, cat in enumerate(CATEGORIES):
    for diff in ("easy", "medium", "hard"):
        params = _DIFFICULTY_PARAMS[diff]
        entry = _entry(
            cat,
            "datasets",
            diff,
            seed=100 * cat_i + hash(diff) % 97,
            width=params["width"],  # type: ignore[arg-type]
            height=params["height"],  # type: ignore[arg-type]
            difficulty=diff,
            palette_count=params["palette_count"],  # type: ignore[arg-type]
            expected_region_count=params["expected_region_count"],  # type: ignore[arg-type]
            expected_printability=0.85 if diff != "hard" else 0.7,
        )
        _DATASETS[entry.fixture_id] = entry

# Golden subset: the "medium" variant per category, frozen for golden compare.
_GOLDEN_IDS: tuple[str, ...] = tuple(f"D-{cat}-datasets-medium" for cat in CATEGORIES)

_ALL: dict[str, RegistryEntry] = {**_EXAMPLES, **_DATASETS}


def available_fixture_ids() -> tuple[str, ...]:
    return tuple(sorted(_ALL))


def available_example_ids() -> tuple[str, ...]:
    return tuple(sorted(_EXAMPLES))


def available_golden_ids() -> tuple[str, ...]:
    return _GOLDEN_IDS


def get_entry(fixture_id: str) -> RegistryEntry:
    if fixture_id not in _ALL:
        raise KeyError(f"unknown fixture {fixture_id!r}; available: {available_fixture_ids()}")
    return _ALL[fixture_id]


def fixture_ids_for_category(category: str) -> tuple[str, ...]:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; must be one of {CATEGORIES}")
    return tuple(sorted(fid for fid, e in _ALL.items() if e.category == category))


def generate_labels(fixture_id: str) -> np.ndarray:
    entry = get_entry(fixture_id)
    return generate_category_labels(
        entry.category,
        seed=entry.seed,
        width=entry.width,
        height=entry.height,
        k=entry.palette_count,
    )
