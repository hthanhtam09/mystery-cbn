"""Concrete layout-domain artifacts: Label and Legend (DATA_MODEL_SPEC.md §15–§16)."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from mysterycbn.model._utils import require
from mysterycbn.model.records import Provenance


class LabelMode(enum.Enum):
    """Placement kind of a region's printed number."""

    IN_REGION = "in_region"
    LEADER = "leader"


@dataclass(frozen=True)
class Label:
    """Placement of one region's printed number (DATA_MODEL_SPEC §15).

    Rotation deliberately does not exist in this model (QM-23 is structural).
    """

    region_id: int
    printed_number: int
    anchor: tuple[float, float]
    font_size_pt: float
    mode: LabelMode
    clearance_pt: float
    leader: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        require(self.region_id >= 0, "region_id must be ≥ 0")
        require(self.printed_number >= 1, "printed numbers are 1-based")
        require(self.font_size_pt > 0.0, "font_size_pt must be positive")
        require(self.clearance_pt >= 0.0, "clearance_pt must be ≥ 0")
        if self.mode is LabelMode.LEADER:
            require(self.leader is not None, "LEADER labels must carry a leader segment")
        else:
            require(self.leader is None, "IN_REGION labels must not carry a leader")

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "printed_number": self.printed_number,
            "anchor": list(self.anchor),
            "font_size_pt": self.font_size_pt,
            "mode": self.mode.value,
            "clearance_pt": self.clearance_pt,
            "leader": [list(p) for p in self.leader] if self.leader else None,
        }


@dataclass(frozen=True)
class Legend:
    """Number↔color key layout plus the palette permutation (DATA_MODEL_SPEC §16).

    ``printed_number = permutation[palette_index] + 1`` (printed numbers are
    1-based). ``chips`` are ``(palette_index, (x, y), side_pt)`` in printed
    order; ``band_rect`` is ``(x, y, w, h)`` in pt.
    """

    permutation: tuple[int, ...]
    chips: tuple[tuple[int, tuple[float, float], float], ...]
    band_rect: tuple[float, float, float, float]
    number_font_pt: float
    provenance: Provenance

    def __post_init__(self) -> None:
        k = len(self.permutation)
        require(k >= 2, "legend needs ≥ 2 colors")
        require(sorted(self.permutation) == list(range(k)), "permutation must be a bijection")
        require(len(self.chips) == k, "one chip per palette entry required")
        require(self.number_font_pt > 0.0, "number_font_pt must be positive")
        bx, by, bw, bh = self.band_rect
        require(bw > 0.0 and bh > 0.0, "band_rect must have positive extent")
        for palette_index, (cx, cy), side in self.chips:
            require(
                0 <= palette_index < k, f"chip references unknown palette index {palette_index}"
            )
            require(side > 0.0, "chip side must be positive")
            require(
                bx <= cx and by <= cy and cx + side <= bx + bw and cy + side <= by + bh,
                f"chip for palette index {palette_index} lies outside the legend band",
            )

    def printed_number(self, palette_index: int) -> int:
        """The 1-based number printed for ``palette_index``."""
        return self.permutation[palette_index] + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "permutation": list(self.permutation),
            "chips": [[i, list(pos), side] for i, pos, side in self.chips],
            "band_rect": list(self.band_rect),
            "number_font_pt": self.number_font_pt,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class LabelPlan:
    """All label placements: ``labels`` sorted by region_id + provenance
    (DATA_MODEL_SPEC §15 — LabelPlan has no other fields)."""

    labels: tuple[Label, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        ids = [label.region_id for label in self.labels]
        require(
            ids == sorted(ids) and len(set(ids)) == len(ids),
            "labels must be sorted by unique region_id",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": [label.to_dict() for label in self.labels],
            "provenance": self.provenance.to_dict(),
        }
