"""Assembles one full ``BenchmarkReport`` (BENCHMARK_SPEC.md §11) by running
performance measurement, quality validation, golden comparison, and
regression detection over a fixture ladder, then scoring the run (§10.2).
"""

from __future__ import annotations

import platform
import resource
import subprocess
import time
import uuid
from dataclasses import dataclass

from benchmarks.framework.fixtures import (
    DATASET_VERSION,
    Fixture,
    load_full_ladder,
    load_smoke_fixtures,
)
from benchmarks.framework.perf import REPEATS_DEFAULT, measure_performance
from benchmarks.framework.pipeline import run_pipeline
from benchmarks.framework.quality import compute_quality_report
from benchmarks.framework.regression import check_regressions, load_baselines
from benchmarks.framework.score import compute_score
from benchmarks.framework.visual import compare_to_golden, has_golden
from mysterycbn import __version__ as ENGINE_VERSION
from mysterycbn.model.reports import (
    BenchmarkReport,
    FailureTuple,
    GoldenOutcome,
    MachineFingerprint,
    MetricResult,
)

REPORT_SCHEMA = 3
SCORE_VERSION = 1
PRESET = "medium"  # the framework runs a single synthetic preset today


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _canary() -> float:
    """A tiny fixed synthetic workload (BENCHMARK_SPEC §9.1 calibration
    canary) -- not gated here (no fleet of runners to compare against), but
    recorded so a future CI job can add the >10% deviation abort."""
    import numpy as np

    t0 = time.perf_counter()
    a = np.random.default_rng(0).random((300, 300))
    b = np.random.default_rng(1).random((300, 300))
    _ = a @ b
    return time.perf_counter() - t0


def _machine_fingerprint() -> MachineFingerprint:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_gib = (
        peak / (1024.0 * 1024.0)
        if platform.system() != "Linux"
        else peak / (1024.0 * 1024.0 * 1024.0)
    )
    return MachineFingerprint(
        cpu=platform.processor() or platform.machine(),
        cores=_cpu_count(),
        memory_gib=max(memory_gib, 0.1),
        container_digest="local-dev",
        kernel=platform.release(),
        lockfile_hash="unpinned",
        canary_s=_canary(),
    )


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 1


@dataclass(frozen=True)
class FixtureResult:
    """Everything measured for one fixture, before assembly into the
    BenchmarkReport's aggregate structures."""

    fixture: Fixture
    perf_metrics: dict[str, MetricResult]
    stage_wall_s: dict[str, float]
    quality_metrics: dict[str, MetricResult]
    golden_outcome: GoldenOutcome
    ssim_solved: float | None
    fatal_findings: list[str]


def run_fixture(fixture: Fixture, *, repeats: int = REPEATS_DEFAULT) -> FixtureResult:
    """Full measurement of one fixture: perf, quality, golden."""
    perf_report = measure_performance(fixture, repeats=repeats)
    run = run_pipeline(fixture)
    quality_report = compute_quality_report(run)
    golden = compare_to_golden(run) if has_golden(fixture.fixture_id) else None

    return FixtureResult(
        fixture=fixture,
        perf_metrics=perf_report.metrics,
        stage_wall_s=perf_report.stage_wall_s,
        quality_metrics=quality_report.metrics,
        golden_outcome=golden.svg_outcome if golden else GoldenOutcome.INCOMPATIBLE,
        ssim_solved=golden.ssim_solved if golden else None,
        fatal_findings=quality_report.fatal_findings,
    )


def build_report(
    *,
    suite: str = "full",
    repeats: int = REPEATS_DEFAULT,
    require_golden: bool = False,
) -> BenchmarkReport:
    """Run the named suite ("smoke" | "full") and assemble the complete
    ``BenchmarkReport`` (§11), including regression checks against the
    machine class's committed baselines and the §10.2 Engine Score."""
    fixtures = load_smoke_fixtures() if suite == "smoke" else load_full_ladder()
    baselines = load_baselines()

    metrics: dict[str, dict[str, dict[str, MetricResult]]] = {}
    stages: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    golden: dict[str, GoldenOutcome] = {}
    failures: list[FailureTuple] = []

    for fixture in fixtures:
        result = run_fixture(fixture, repeats=repeats)
        combined = {**result.perf_metrics, **result.quality_metrics}
        metrics[fixture.fixture_id] = {PRESET: combined}
        stages[fixture.fixture_id] = {
            PRESET: {stage: {"wall_s": wall} for stage, wall in result.stage_wall_s.items()}
        }
        golden[f"{fixture.fixture_id}/{PRESET}"] = result.golden_outcome

        failures.extend(
            check_regressions(
                fixture_id=fixture.fixture_id,
                preset=PRESET,
                metrics=combined,
                baselines=baselines,
            )
        )
        if result.fatal_findings:
            failures.append(
                FailureTuple(
                    metric="fatal_finding",
                    fixture=fixture.fixture_id,
                    preset=PRESET,
                    value=float(len(result.fatal_findings)),
                    band=(0.0, 0.0),
                )
            )
        if (
            require_golden
            and golden[f"{fixture.fixture_id}/{PRESET}"] is GoldenOutcome.INCOMPATIBLE
        ):
            failures.append(
                FailureTuple(
                    metric="golden",
                    fixture=fixture.fixture_id,
                    preset=PRESET,
                    value=0.0,
                    band=(1.0, 1.0),
                )
            )

    score_total, score_dims = compute_score(metrics)

    return BenchmarkReport(
        run_id=str(uuid.uuid4()),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_sha=_git_sha(),
        engine_version=ENGINE_VERSION,
        machine=_machine_fingerprint(),
        dataset_version=DATASET_VERSION,
        score_version=SCORE_VERSION,
        report_schema=REPORT_SCHEMA,
        metrics=metrics,
        stages=stages,
        golden=golden,
        score_total=score_total,
        score_dimensions=score_dims,
        accepted=len(failures) == 0,
        failures=tuple(failures),
    )
