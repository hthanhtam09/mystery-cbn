"""Shared data model flowing through the pipeline.

The :class:`PipelineContext` is a blackboard: raster stages populate arrays,
graph stages populate the region graph, vector stages populate curve sets.
Stages declare (in code) which fields they require; :mod:`core.pipeline`
enforces those declarations so a mis-ordered pipeline fails immediately with
a clear message instead of an AttributeError deep inside a stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import EngineConfig


@dataclass(frozen=True)
class PaletteColor:
    """One legend entry. LAB is the authoritative space; sRGB is for output."""

    number: int  # 1-based number printed on the page
    lab: tuple[float, float, float]
    rgb: tuple[int, int, int]  # 0-255 sRGB

    @property
    def hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)


@dataclass(frozen=True)
class Palette:
    colors: tuple[PaletteColor, ...]

    def __len__(self) -> int:
        return len(self.colors)

    def __getitem__(self, index: int) -> PaletteColor:
        return self.colors[index]


@dataclass(frozen=True)
class ImageStats:
    """Cheap global statistics used to auto-tune stage parameters."""

    mean_lab: tuple[float, float, float]
    colorfulness: float  # Hasler–Süsstrunk metric
    edge_density: float  # fraction of Canny-positive pixels
    distinct_hues: int


@dataclass
class StageTiming:
    name: str
    seconds: float


@dataclass
class StageTrace:
    """Timings plus optional per-stage debug artifacts (label-map snapshots)."""

    timings: list[StageTiming] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable state shared by all stages of one conversion run."""

    config: EngineConfig
    trace: StageTrace = field(default_factory=StageTrace)

    # Raster domain (working resolution)
    image: np.ndarray | None = None  # H×W×3 float32 sRGB in [0,1]
    work_scale: float = 1.0  # output px per working px
    stats: ImageStats | None = None
    palette: Palette | None = None
    label_map: np.ndarray | None = None  # H×W int32 palette indices

    # Graph domain
    region_graph: Any | None = None  # modules.regions.RegionGraph

    # Vector domain
    boundaries: Any | None = None  # geometry.arcgraph.ArcGraph
    curves: Any | None = None
    labels: Any | None = None
    legend: Any | None = None

    def require(self, *fields_: str) -> None:
        """Raise with a clear message if a prerequisite field is unset."""
        missing = [f for f in fields_ if getattr(self, f) is None]
        if missing:
            raise AttributeError(
                f"pipeline context missing required field(s): {', '.join(missing)}"
            )
