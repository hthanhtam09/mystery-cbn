"""Immutable pipeline artifacts (ARCHITECTURE.md §4.1).

Stages never call each other; they communicate only through these typed artifacts.
Every artifact carries provenance naming the stage and config hash that produced it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from mysterycbn.foundation.geometry.types import Arc, BezierChain, Face, Point


@runtime_checkable
class Provenance(Protocol):
    """Who made this artifact, from what."""

    @property
    def stage_name(self) -> str: ...

    @property
    def stage_version(self) -> str: ...

    @property
    def config_hash(self) -> str: ...

    @property
    def source_hash(self) -> str: ...


@runtime_checkable
class Artifact(Protocol):
    """Base contract every artifact satisfies."""

    @property
    def provenance(self) -> Provenance: ...


@runtime_checkable
class RasterImage(Artifact, Protocol):
    """H×W×3 float32 sRGB in [0, 1], EXIF-oriented, ICC-normalized."""

    @property
    def pixels(self) -> np.ndarray: ...

    @property
    def work_scale(self) -> float:
        """Working pixels per point; 0.0 until the preprocess stage sets it."""
        ...


@runtime_checkable
class ImageStats(Artifact, Protocol):
    """Global statistics feeding auto-tune proposals."""

    @property
    def colorfulness(self) -> float: ...

    @property
    def edge_density(self) -> float: ...

    @property
    def luminance_histogram(self) -> np.ndarray: ...


@runtime_checkable
class Palette(Artifact, Protocol):
    """Extracted palette; LAB is authoritative, sRGB is derived."""

    @property
    def lab(self) -> np.ndarray: ...

    @property
    def srgb(self) -> np.ndarray: ...

    @property
    def size(self) -> int: ...


@runtime_checkable
class LabelMap(Artifact, Protocol):
    """H×W int32 palette indices over the working raster."""

    @property
    def labels(self) -> np.ndarray: ...


@runtime_checkable
class RegionRecord(Protocol):
    """One connected region node."""

    @property
    def region_id(self) -> int: ...

    @property
    def label(self) -> int: ...

    @property
    def area_px(self) -> int: ...

    @property
    def bbox(self) -> tuple[int, int, int, int]: ...

    @property
    def seed_px(self) -> tuple[int, int]: ...


@runtime_checkable
class RegionGraph(Artifact, Protocol):
    """Regions plus adjacency (shared boundary length, ΔE); label raster stays authoritative."""

    @property
    def regions(self) -> Sequence[RegionRecord]: ...

    def neighbors(self, region_id: int) -> Sequence[int]:
        """Adjacent region ids."""
        ...

    def edge_weight(self, region_a: int, region_b: int) -> tuple[float, float]:
        """(shared boundary length in px, ΔE between mean colors) for an adjacency edge."""
        ...


@runtime_checkable
class ArcGraph(Artifact, Protocol):
    """Shared-boundary topology in physical units. After this artifact exists,
    no stage may consult the raster (domain boundary, ARCHITECTURE.md §1.2)."""

    @property
    def arcs(self) -> Sequence[Arc]: ...

    @property
    def faces(self) -> Sequence[Face]: ...


@runtime_checkable
class CurveSet(Artifact, Protocol):
    """Per-arc Bézier chains in points; faces carried over unchanged from the ArcGraph."""

    @property
    def chains(self) -> Sequence[BezierChain]: ...

    @property
    def faces(self) -> Sequence[Face]: ...


@runtime_checkable
class RegionLabel(Protocol):
    """Number placement for one region."""

    @property
    def region_id(self) -> int: ...

    @property
    def anchor(self) -> Point: ...

    @property
    def font_size_pt(self) -> float: ...

    @property
    def leader_line(self) -> Sequence[Point] | None: ...


@runtime_checkable
class LabelPlan(Artifact, Protocol):
    """All region number placements."""

    @property
    def labels(self) -> Sequence[RegionLabel]: ...


@runtime_checkable
class LegendPlan(Artifact, Protocol):
    """Palette order, chip layout, and the mystery renumbering map."""

    @property
    def palette_order(self) -> Sequence[int]: ...

    @property
    def renumbering(self) -> Mapping[int, int]:
        """Original palette index → printed number."""
        ...


@runtime_checkable
class RunReport(Artifact, Protocol):
    """Timings, warnings, metrics, and the reproducibility record (ARCHITECTURE.md §7)."""

    @property
    def resolved_config(self) -> Mapping[str, object]: ...

    @property
    def engine_version(self) -> str: ...

    @property
    def input_hash(self) -> str: ...

    @property
    def warnings(self) -> Sequence[str]: ...

    @property
    def stage_timings_s(self) -> Mapping[str, float]: ...


@runtime_checkable
class OutputBundle(Protocol):
    """Atomic final output: all artifacts validated and written, or none (ARCHITECTURE.md §11)."""

    @property
    def svg(self) -> bytes: ...

    @property
    def pdf(self) -> bytes | None: ...

    @property
    def previews_png(self) -> Mapping[str, bytes]: ...

    @property
    def report(self) -> RunReport: ...
