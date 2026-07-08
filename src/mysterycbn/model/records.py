"""Concrete raster/graph-domain artifacts (DATA_MODEL_SPEC.md §2–§8).

All classes are frozen; arrays are read-only. Constructor validation covers
local well-formedness; cross-object invariants I1–I4 belong to the validation
subsystem, not to these constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mysterycbn.foundation.color import DefaultColorScience
from mysterycbn.model._utils import array_meta, readonly, require, require_hex64

_COLOR = DefaultColorScience()

_MIN_RASTER_SIDE = 64
_MAX_PALETTE = 64


@dataclass(frozen=True)
class Provenance:
    """Who produced an artifact, from what (DATA_MODEL_SPEC §2)."""

    stage_name: str
    stage_version: str
    config_hash: str
    source_hash: str

    def __post_init__(self) -> None:
        require(bool(self.stage_name), "stage_name must be non-empty")
        require(bool(self.stage_version), "stage_version must be non-empty")
        require_hex64(self.config_hash, "config_hash")
        require_hex64(self.source_hash, "source_hash")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "stage_version": self.stage_version,
            "config_hash": self.config_hash,
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True)
class RasterImage:
    """Canonical raster: H×W×3 float32 sRGB in [0, 1] (DATA_MODEL_SPEC §3)."""

    pixels: np.ndarray
    work_scale: float
    resize_factor: float
    icc_applied: bool
    exif_orientation: int
    provenance: Provenance

    def __post_init__(self) -> None:
        px = readonly(self.pixels, np.float32)
        require(
            px.ndim == 3 and px.shape[2] == 3,
            f"pixels must be (H, W, 3), got {px.shape}",
        )
        require(
            px.shape[0] >= _MIN_RASTER_SIDE and px.shape[1] >= _MIN_RASTER_SIDE,
            f"raster sides must be ≥ {_MIN_RASTER_SIDE}, got {px.shape[:2]}",
        )
        require(bool(np.isfinite(px).all()), "pixels must be finite")
        require(float(px.min()) >= 0.0 and float(px.max()) <= 1.0, "pixels must be in [0, 1]")
        require(self.work_scale >= 0.0, "work_scale must be ≥ 0 (0 = unset)")
        require(0.0 < self.resize_factor <= 1.0, "resize_factor must be in (0, 1]")
        require(1 <= self.exif_orientation <= 8, "exif_orientation must be an EXIF tag 1–8")
        object.__setattr__(self, "pixels", px)

    def to_dict(self) -> dict[str, object]:
        """Metadata form; the pixel payload lives in the debug-snapshot regime."""
        return {
            "pixels": array_meta(self.pixels),
            "work_scale": self.work_scale,
            "resize_factor": self.resize_factor,
            "icc_applied": self.icc_applied,
            "exif_orientation": self.exif_orientation,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class ImageStats:
    """Global image statistics feeding auto-tune proposals (ENGINE_SPEC §6).

    ``luminance_histogram`` is 64 uniform bins over L* ∈ [0, 100], normalized
    to sum 1. ``brightness``/``contrast`` are mean/std of L*; ``saturation``
    is mean chroma C* = hypot(a*, b*); ``entropy_bits`` is the histogram's
    Shannon entropy.
    """

    colorfulness: float
    edge_density: float
    luminance_histogram: np.ndarray
    lab_mean: tuple[float, float, float]
    lab_std: tuple[float, float, float]
    brightness: float
    contrast: float
    saturation: float
    entropy_bits: float
    provenance: Provenance

    def __post_init__(self) -> None:
        hist = readonly(self.luminance_histogram, np.float64)
        require(hist.shape == (64,), f"histogram must be (64,), got {hist.shape}")
        require(abs(float(hist.sum()) - 1.0) <= 1e-9, "histogram must sum to 1")
        require(float(hist.min()) >= 0.0, "histogram bins must be ≥ 0")
        require(self.colorfulness >= 0.0, "colorfulness must be ≥ 0")
        require(0.0 <= self.edge_density <= 1.0, "edge_density must be in [0, 1]")
        require(self.contrast >= 0.0 and self.saturation >= 0.0, "std/chroma must be ≥ 0")
        require(0.0 <= self.entropy_bits <= 6.0 + 1e-9, "entropy of 64 bins is ≤ 6 bits")
        object.__setattr__(self, "luminance_histogram", hist)

    def to_dict(self) -> dict[str, object]:
        return {
            "colorfulness": self.colorfulness,
            "edge_density": self.edge_density,
            "luminance_histogram": self.luminance_histogram.tolist(),
            "lab_mean": list(self.lab_mean),
            "lab_std": list(self.lab_std),
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "entropy_bits": self.entropy_bits,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class PaletteColor:
    """One palette entry; LAB is authoritative, sRGB derived (DATA_MODEL_SPEC §4)."""

    index: int
    lab: tuple[float, float, float]
    srgb: tuple[float, float, float]
    coverage_px: int

    def __post_init__(self) -> None:
        require(self.index >= 0, "index must be ≥ 0")
        require(self.coverage_px >= 0, "coverage_px must be ≥ 0")
        derived = _COLOR.lab_to_srgb(np.array(self.lab, dtype=np.float64))
        require(
            bool(np.allclose(np.array(self.srgb), derived, atol=1e-9)),
            "srgb must equal the gamut-clamped conversion of lab (use from_lab)",
        )

    @classmethod
    def from_lab(
        cls, index: int, lab: tuple[float, float, float], coverage_px: int
    ) -> PaletteColor:
        """Construct with ``srgb`` derived from ``lab`` — the sanctioned path."""
        srgb = _COLOR.lab_to_srgb(np.array(lab, dtype=np.float64))
        return cls(
            index=index,
            lab=lab,
            srgb=(float(srgb[0]), float(srgb[1]), float(srgb[2])),
            coverage_px=coverage_px,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "lab": list(self.lab),
            "srgb": list(self.srgb),
            "coverage_px": self.coverage_px,
        }


@dataclass(frozen=True)
class Palette:
    """Ordered color set with a cached ΔE00 table (DATA_MODEL_SPEC §5).

    ``min_delta_e`` records the separation the producing stage guarantees;
    the constructor re-checks it (ENGINE_SPEC §7 R4).
    """

    colors: tuple[PaletteColor, ...]
    provenance: Provenance
    min_delta_e: float = 0.0
    delta_e_table: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        k = len(self.colors)
        require(2 <= k <= _MAX_PALETTE, f"palette size must be in [2, {_MAX_PALETTE}], got {k}")
        require(
            tuple(c.index for c in self.colors) == tuple(range(k)),
            "colors must be sorted by dense index 0…K−1",
        )
        lab = np.array([c.lab for c in self.colors], dtype=np.float64)
        table = _COLOR.delta_e_2000(lab[:, None, :], lab[None, :, :])
        table.setflags(write=False)
        object.__setattr__(self, "delta_e_table", table)
        if self.min_delta_e > 0.0:
            off = table[~np.eye(k, dtype=bool)]
            require(
                float(off.min()) >= self.min_delta_e,
                f"palette separation {float(off.min()):.3f} below min_delta_e {self.min_delta_e}",
            )

    @property
    def size(self) -> int:
        """Number of palette entries (K)."""
        return len(self.colors)

    def to_dict(self) -> dict[str, object]:
        return {
            "colors": [c.to_dict() for c in self.colors],
            "min_delta_e": self.min_delta_e,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class LabelMap:
    """Per-pixel palette assignment, H×W int32 (DATA_MODEL_SPEC §6)."""

    labels: np.ndarray
    provenance: Provenance

    def __post_init__(self) -> None:
        lb = readonly(self.labels, np.int32)
        require(lb.ndim == 2 and lb.size > 0, f"labels must be non-empty 2-D, got {lb.shape}")
        require(int(lb.min()) >= 0, "labels must be ≥ 0")
        object.__setattr__(self, "labels", lb)

    def validate_against(self, palette: Palette) -> None:
        """Pair check: every label must index the same-generation palette."""
        require(
            int(self.labels.max()) < palette.size,
            f"label {int(self.labels.max())} out of range for palette of {palette.size}",
        )

    def to_dict(self) -> dict[str, object]:
        """Metadata form; the label payload lives in the debug-snapshot regime."""
        return {"labels": array_meta(self.labels), "provenance": self.provenance.to_dict()}


@dataclass(frozen=True)
class Region:
    """One 4-connected region record (DATA_MODEL_SPEC §7).

    ``centroid`` is the mean of the region's pixel centers, (row, col) in px.
    """

    region_id: int
    label: int
    area_px: int
    bbox: tuple[int, int, int, int]
    seed_px: tuple[int, int]
    perimeter_px: int
    centroid: tuple[float, float]

    def __post_init__(self) -> None:
        require(self.region_id >= 0 and self.label >= 0, "ids must be ≥ 0")
        require(self.area_px >= 1, "area_px must be ≥ 1")
        require(self.perimeter_px >= 4, "perimeter_px must be ≥ 4")
        r0, c0, r1, c1 = self.bbox
        require(r0 <= r1 and c0 <= c1, f"bbox must be ordered, got {self.bbox}")
        require(
            r0 <= self.seed_px[0] <= r1 and c0 <= self.seed_px[1] <= c1,
            "seed_px must lie inside bbox",
        )
        require(
            r0 <= self.centroid[0] <= r1 and c0 <= self.centroid[1] <= c1,
            "centroid must lie inside bbox",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "label": self.label,
            "area_px": self.area_px,
            "bbox": list(self.bbox),
            "seed_px": list(self.seed_px),
            "perimeter_px": self.perimeter_px,
            "centroid": list(self.centroid),
        }


@dataclass(frozen=True)
class RegionGraph:
    """Region adjacency graph + authoritative component map (DATA_MODEL_SPEC §8).

    ``edges`` rows are ``(a, b, delta_e, boundary_px)`` with ``a < b``, sorted
    lexicographically.
    """

    regions: tuple[Region, ...]
    component_map: np.ndarray
    edges: tuple[tuple[int, int, float, int], ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        cmap = readonly(self.component_map, np.int32)
        r = len(self.regions)
        require(r >= 1, "at least one region required")
        require(
            tuple(reg.region_id for reg in self.regions) == tuple(range(r)),
            "regions must be sorted by dense region_id",
        )
        require(cmap.ndim == 2 and cmap.size > 0, "component_map must be non-empty 2-D")
        ids = np.unique(cmap)
        require(
            int(ids[0]) == 0 and int(ids[-1]) == r - 1 and len(ids) == r,
            "component_map values must be dense in [0, R)",
        )
        prev: tuple[int, int] | None = None
        for a, b, delta_e, boundary in self.edges:
            require(a < b, f"edge ({a}, {b}) must satisfy a < b")
            require(0 <= a < r and b < r, f"edge ({a}, {b}) references unknown region")
            require(delta_e >= 0.0 and boundary >= 1, "edge weights must be non-negative")
            require(prev is None or (a, b) > prev, "edges must be sorted and unique")
            prev = (a, b)
        object.__setattr__(self, "component_map", cmap)

    def neighbors(self, region_id: int) -> tuple[int, ...]:
        """Adjacent region ids of ``region_id``."""
        out = [b if a == region_id else a for a, b, _, _ in self.edges if region_id in (a, b)]
        return tuple(sorted(out))

    def edge_weight(self, region_a: int, region_b: int) -> tuple[float, float]:
        """(boundary length px, ΔE00) for an adjacency edge; raises if absent."""
        key = (min(region_a, region_b), max(region_a, region_b))
        for a, b, delta_e, boundary in self.edges:
            if (a, b) == key:
                return float(boundary), delta_e
        raise KeyError(f"no edge between regions {region_a} and {region_b}")

    def to_dict(self) -> dict[str, object]:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "component_map": array_meta(self.component_map),
            "edges": [list(e) for e in self.edges],
            "provenance": self.provenance.to_dict(),
        }
