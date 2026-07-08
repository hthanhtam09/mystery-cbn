"""Metadata schema for the categorized dataset (docs/DATASET_STANDARDS.md §2).

Every dataset entry carries a ``FixtureMetadata`` record alongside its
generated label map. The schema is deliberately independent of the QM-01..33
battery (BENCHMARK_SPEC.md/QUALITY_SPEC.md) -- these are dataset-declared
expectations checked by dataset-level tests, not engine quality gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

CATEGORIES = (
    "animals",
    "flowers",
    "people",
    "landscape",
    "architecture",
    "food",
    "vehicles",
    "cartoons",
)

Category = Literal[
    "animals",
    "flowers",
    "people",
    "landscape",
    "architecture",
    "food",
    "vehicles",
    "cartoons",
]

Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class FixtureMetadata:
    """Declared, checkable expectations for one dataset fixture."""

    fixture_id: str
    category: Category
    width: int
    height: int
    difficulty: Difficulty
    palette_count: int
    expected_region_count: int
    expected_printability: float

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}; must be one of {CATEGORIES}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width/height must be positive, got {self.width}x{self.height}")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"unknown difficulty {self.difficulty!r}")
        if self.palette_count <= 0:
            raise ValueError(f"palette_count must be positive, got {self.palette_count}")
        if self.expected_region_count <= 0:
            raise ValueError(
                f"expected_region_count must be positive, got {self.expected_region_count}"
            )
        if not (0.0 <= self.expected_printability <= 1.0):
            raise ValueError(
                f"expected_printability must be in [0, 1], got {self.expected_printability}"
            )

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000.0

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["resolution"] = list(self.resolution)
        return d
