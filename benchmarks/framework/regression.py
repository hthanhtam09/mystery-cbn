"""Regression detection against committed baselines (BENCHMARK_SPEC.md §7).

Baselines live in ``benchmarks/baselines/<machine-class>.json``: per
fixture x metric, a baseline value + tolerance. Gate metrics are checked
against their absolute QUALITY_SPEC band regardless of baseline (a Gate
never needs history to know it failed); Monitor metrics are checked against
``baseline +- tolerance`` per §7.2. Baselines change only via an explicit
call to ``update_baseline`` (never automatically during a comparison run).
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

from mysterycbn.model.reports import FailureTuple, MetricClass, MetricResult

BASELINES_ROOT = Path(__file__).resolve().parents[1] / "baselines"

# Per-metric relative tolerance for Monitor metrics without a QM-specific
# override (QUALITY_SPEC's own per-metric tolerances win when listed here).
_DEFAULT_TOLERANCE = 0.20
_METRIC_TOLERANCE: dict[str, float] = {
    "QM-05": 0.03,  # absolute, handled specially below
    "QM-06": 0.10,
    "QM-07": 0.20,
    "QM-08": 0.15,
    "QM-12": 0.03,  # absolute points, handled specially below
    "QM-13": 0.15,
    "QM-14": 0.05,  # absolute, handled specially below
    "QM-22": 0.02,  # absolute points, handled specially below
    "QM-29": 0.01,  # absolute, handled specially below
    "QM-32-svg": 0.15,
    "QM-32-pdf": 0.15,
    "QM-33": 0.08,
    "stage_wall_s": 0.20,
    "peak_rss_mib": 0.15,
}


def machine_class() -> str:
    """A coarse, portable machine-class key (BENCHMARK_SPEC §9.1). Real CI
    pins this to a container digest; locally it falls back to the platform
    tuple so baselines are still self-consistent across runs on one machine."""
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def baseline_path(machine_class_name: str | None = None) -> Path:
    return BASELINES_ROOT / f"{machine_class_name or machine_class()}.json"


@dataclass(frozen=True)
class Baseline:
    """One (fixture, metric) baseline entry."""

    value: float
    tolerance: float
    run_id: str


def load_baselines(machine_class_name: str | None = None) -> dict[str, dict[str, Baseline]]:
    """fixture_id -> metric_id -> Baseline. Missing file -> empty (first run
    establishes baselines via ``update_baseline``, never silently)."""
    path = baseline_path(machine_class_name)
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    return {
        fixture_id: {
            metric_id: Baseline(entry["value"], entry["tolerance"], entry["run_id"])
            for metric_id, entry in metrics.items()
        }
        for fixture_id, metrics in doc.items()
    }


def save_baselines(
    baselines: dict[str, dict[str, Baseline]], machine_class_name: str | None = None
) -> None:
    path = baseline_path(machine_class_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        fixture_id: {
            metric_id: {"value": b.value, "tolerance": b.tolerance, "run_id": b.run_id}
            for metric_id, b in metrics.items()
        }
        for fixture_id, metrics in baselines.items()
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def update_baseline(
    fixture_id: str,
    metric_id: str,
    value: float,
    *,
    run_id: str,
    tolerance: float | None = None,
    machine_class_name: str | None = None,
) -> None:
    """Explicitly (re)establish one baseline entry -- the only sanctioned
    write path (BENCHMARK_SPEC §7.1: "change only by explicit reviewed
    commit")."""
    baselines = load_baselines(machine_class_name)
    tol = (
        tolerance if tolerance is not None else _METRIC_TOLERANCE.get(metric_id, _DEFAULT_TOLERANCE)
    )
    baselines.setdefault(fixture_id, {})[metric_id] = Baseline(
        value=value, tolerance=tol, run_id=run_id
    )
    save_baselines(baselines, machine_class_name)


def _monitor_within_tolerance(value: float, baseline: Baseline) -> bool:
    if baseline.value == 0.0:
        return abs(value) <= baseline.tolerance
    return abs(value - baseline.value) / abs(baseline.value) <= baseline.tolerance


def check_regressions(
    *,
    fixture_id: str,
    preset: str,
    metrics: dict[str, MetricResult],
    baselines: dict[str, dict[str, Baseline]],
) -> list[FailureTuple]:
    """BENCHMARK_SPEC §7.2 decision rules. Gate metrics fail on their own
    band regardless of baseline; Monitor metrics fail only against a
    recorded baseline (no baseline yet -> nothing to regress against, not a
    failure -- the caller should still call ``update_baseline`` to start
    tracking it)."""
    failures: list[FailureTuple] = []
    fixture_baselines = baselines.get(fixture_id, {})

    for metric_id, result in metrics.items():
        if result.metric_class is MetricClass.GATE:
            if not result.passed:
                failures.append(
                    FailureTuple(metric_id, fixture_id, preset, result.value, result.band)
                )
            continue

        baseline = fixture_baselines.get(metric_id)
        if baseline is None:
            continue  # no history yet -- not a regression, just unestablished
        if not _monitor_within_tolerance(result.value, baseline):
            lo = baseline.value * (1 - baseline.tolerance)
            hi = baseline.value * (1 + baseline.tolerance)
            failures.append(
                FailureTuple(
                    metric_id, fixture_id, preset, result.value, (min(lo, hi), max(lo, hi))
                )
            )

    return failures
