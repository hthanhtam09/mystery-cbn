"""Rule-based recommendation generation from cross-preset technical-quality
deltas (docs/TECHNICAL_QUALITY_COMPARISON.md §6).

Each rule inspects one relative delta between two presets' ``QualitySnapshot``s
and, if the delta crosses a documented threshold, emits a ``Recommendation``.
Rules are independent and side-effect-free -- adding a rule never changes
another rule's output, and no rule can block or fail a comparison (this is
advisory tooling, not a gate, mirroring quality_metrics.py's Monitor-only
posture)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from benchmarks.comparison.evaluate import QualitySnapshot

_SMOOTHNESS_REGRESSION_RATIO = 1.5
_COMPACTNESS_REGRESSION_RATIO = 0.85
_TINY_REGION_INCREASE_PP = 5.0
_LABEL_DENSITY_INCREASE_RATIO = 1.5
_EDGE_LENGTH_COLLAPSE_RATIO = 0.5


@dataclass(frozen=True)
class Recommendation:
    """One advisory finding comparing two presets on one fixture."""

    fixture_id: str
    category: str
    from_preset: str
    to_preset: str
    severity: str  # "info" | "caution"
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "category": self.category,
            "from_preset": self.from_preset,
            "to_preset": self.to_preset,
            "severity": self.severity,
            "message": self.message,
        }


def _rec(a: QualitySnapshot, b: QualitySnapshot, *, severity: str, message: str) -> Recommendation:
    return Recommendation(
        fixture_id=a.fixture_id,
        category=a.category,
        from_preset=a.preset,
        to_preset=b.preset,
        severity=severity,
        message=message,
    )


def _region_count_rule(a: QualitySnapshot, b: QualitySnapshot) -> Recommendation | None:
    if a.region_count <= 0:
        return None
    ratio = b.region_count / a.region_count
    if ratio >= 1.3:
        return _rec(
            a,
            b,
            severity="info",
            message=(
                f"region count grows {ratio:.2f}x ({a.region_count} -> {b.region_count}) "
                f"going from {a.preset} to {b.preset}"
            ),
        )
    return None


def _smoothness_rule(a: QualitySnapshot, b: QualitySnapshot) -> Recommendation | None:
    if a.boundary_smoothness <= 0.0:
        return None
    ratio = b.boundary_smoothness / a.boundary_smoothness
    if ratio >= _SMOOTHNESS_REGRESSION_RATIO:
        return _rec(
            a,
            b,
            severity="caution",
            message=(
                f"boundary smoothness (curvature energy) worsens {ratio:.2f}x "
                f"({a.boundary_smoothness:.4f} -> {b.boundary_smoothness:.4f} rad^2/mm) "
                f"going from {a.preset} to {b.preset} -- boundaries are noisier, "
                "not just more detailed"
            ),
        )
    return None


def _compactness_rule(a: QualitySnapshot, b: QualitySnapshot) -> Recommendation | None:
    if a.mean_compactness <= 0.0:
        return None
    ratio = b.mean_compactness / a.mean_compactness
    if ratio <= _COMPACTNESS_REGRESSION_RATIO:
        return _rec(
            a,
            b,
            severity="caution",
            message=(
                f"mean region compactness drops {(1 - ratio) * 100:.0f}% "
                f"({a.mean_compactness:.3f} -> {b.mean_compactness:.3f}) "
                f"going from {a.preset} to {b.preset} -- regions are becoming more jagged/elongated"
            ),
        )
    return None


def _tiny_region_rule(a: QualitySnapshot, b: QualitySnapshot) -> Recommendation | None:
    delta_pp = b.tiny_region_pct - a.tiny_region_pct
    if delta_pp >= _TINY_REGION_INCREASE_PP:
        return _rec(
            a,
            b,
            severity="caution",
            message=(
                f"tiny-region percentage rises {delta_pp:.1f} points "
                f"({a.tiny_region_pct:.1f}% -> {b.tiny_region_pct:.1f}%) "
                f"going from {a.preset} to {b.preset} -- more faces need leader-line demotion"
            ),
        )
    return None


def _label_density_rule(a: QualitySnapshot, b: QualitySnapshot) -> Recommendation | None:
    if a.label_density_per_cm2 <= 0.0:
        return None
    ratio = b.label_density_per_cm2 / a.label_density_per_cm2
    if ratio >= _LABEL_DENSITY_INCREASE_RATIO:
        return _rec(
            a,
            b,
            severity="info",
            message=(
                f"label density grows {ratio:.2f}x "
                f"({a.label_density_per_cm2:.3f} -> {b.label_density_per_cm2:.3f} labels/cm^2) "
                f"going from {a.preset} to {b.preset} -- the page will read as visually busier"
            ),
        )
    return None


def _edge_length_rule(a: QualitySnapshot, b: QualitySnapshot) -> Recommendation | None:
    if a.average_edge_length_mm <= 0.0:
        return None
    ratio = b.average_edge_length_mm / a.average_edge_length_mm
    if ratio <= _EDGE_LENGTH_COLLAPSE_RATIO:
        return _rec(
            a,
            b,
            severity="info",
            message=(
                f"average boundary edge length halves or more "
                f"({a.average_edge_length_mm:.3f}mm -> {b.average_edge_length_mm:.3f}mm) "
                f"going from {a.preset} to {b.preset} -- finer tessellation, "
                "expect larger SVG/PDF output"
            ),
        )
    return None


def _printability_rule(a: QualitySnapshot, b: QualitySnapshot) -> Recommendation | None:
    if b.printability_score < a.printability_score and b.printability_score <= 0.6:
        return _rec(
            a,
            b,
            severity="caution",
            message=(
                f"printability score falls to {b.printability_score:.2f} "
                f"(from {a.printability_score:.2f}) going from {a.preset} to {b.preset} -- "
                "approaching the floor where too many faces need leader lines"
            ),
        )
    return None


_RULES = (
    _region_count_rule,
    _smoothness_rule,
    _compactness_rule,
    _tiny_region_rule,
    _label_density_rule,
    _edge_length_rule,
    _printability_rule,
)


def recommend_for_pair(a: QualitySnapshot, b: QualitySnapshot) -> tuple[Recommendation, ...]:
    """Every rule's verdict for one ordered (a -> b) preset pair on one
    fixture, in rule-declaration order. Rules that see nothing worth
    flagging simply contribute nothing -- an empty tuple means clean, not
    unmeasured."""
    return tuple(rec for rule in _RULES if (rec := rule(a, b)) is not None)


def recommend_across_presets(
    snapshots: tuple[QualitySnapshot, ...],
) -> tuple[Recommendation, ...]:
    """Every rule's verdict across every consecutive preset pair (easy->medium,
    medium->hard, ...) for one fixture's ordered snapshot sequence."""
    recs: list[Recommendation] = []
    for a, b in itertools.pairwise(snapshots):
        recs.extend(recommend_for_pair(a, b))
    return tuple(recs)
