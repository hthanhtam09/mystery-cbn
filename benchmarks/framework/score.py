"""Engine Score (BENCHMARK_SPEC.md §10.2): a single scalar communication
device, never a gate. Weighted geometric mean of six dimension subscores.

The framework computes each dimension from whatever QM metrics it actually
measured (this session's harness covers QM-01/02/10/11/16/18/21/24/26/28/30/31,
not the full QM-01..33 battery -- see quality.py's module docstring); a
dimension whose input metric was never measured defaults to a neutral 1.0
rather than silently zeroing out the score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from mysterycbn.model.reports import MetricResult

_WEIGHTS: dict[str, float] = {
    "fidelity": 0.30,
    "geometry": 0.25,
    "printability": 0.15,
    "color": 0.15,
    "speed": 0.10,
    "efficiency": 0.05,
}

_METRIC_FOR_DIMENSION: dict[str, str] = {
    "fidelity": "QM-18",
    "geometry": "QM-02",
    "printability": "QM-11",
    "color": "QM-16",
    "speed": "QM-30",
    "efficiency": "QM-31",
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def _dimension_fidelity(values: list[float]) -> float:
    """Stand-in from QM-18 (face-label agreement) since this harness has no
    raster fixture to measure QM-17 (solved-preview SSIM) from."""
    if not values:
        return 1.0
    return _clip01(_mean(values) / 0.995)


def _dimension_geometry(values: list[float]) -> float:
    """Stand-in from QM-02 (watertightness) since QM-08 (curvature energy)
    is not computed by this harness."""
    if not values:
        return 1.0
    worst = max(values)
    return _clip01(min(1.0, 1e-4 / max(worst, 1e-4)))


def _dimension_printability(values: list[float]) -> float:
    if not values:
        return 1.0
    leader_ratio_proxy = _mean(values)
    return _clip01(max(0.5, 1.0 - leader_ratio_proxy / 100.0))


def _dimension_color(values: list[float]) -> float:
    """Stand-in from QM-16 (palette separation, higher is better) since
    QM-15 (quantization DeltaE, lower is better) needs a raster fixture."""
    if not values:
        return 1.0
    return _clip01(min(1.0, _mean(values) / 12.0))


def _dimension_speed(values: list[float]) -> float:
    if not values:
        return 1.0
    return _clip01(min(1.0, 12.0 / max(_mean(values), 12.0)))


def _dimension_efficiency(values: list[float]) -> float:
    if not values:
        return 1.0
    return _clip01(min(1.0, 600.0 / max(_mean(values), 600.0)))


_DIMENSION_FNS: dict[str, Callable[[list[float]], float]] = {
    "fidelity": _dimension_fidelity,
    "geometry": _dimension_geometry,
    "printability": _dimension_printability,
    "color": _dimension_color,
    "speed": _dimension_speed,
    "efficiency": _dimension_efficiency,
}


def _collect_values(
    metrics: Mapping[str, Mapping[str, Mapping[str, MetricResult]]], metric_id: str
) -> list[float]:
    values: list[float] = []
    for per_fixture in metrics.values():
        for per_preset in per_fixture.values():
            result = per_preset.get(metric_id)
            if result is not None:
                values.append(result.value)
    return values


def compute_score(
    metrics: Mapping[str, Mapping[str, Mapping[str, MetricResult]]],
) -> tuple[float, dict[str, float]]:
    """(score_total, score_dimensions) per §10.2's weighted geometric mean.
    ``metrics`` is fixture -> preset -> metric_id -> MetricResult (the same
    shape as ``BenchmarkReport.metrics``)."""
    dims: dict[str, float] = {}
    for name, metric_id in _METRIC_FOR_DIMENSION.items():
        values = _collect_values(metrics, metric_id)
        dims[name] = _DIMENSION_FNS[name](values)

    score_total = 100.0
    for name, weight in _WEIGHTS.items():
        score_total *= dims[name] ** weight

    return round(score_total, 4), {k: round(v, 6) for k, v in dims.items()}
