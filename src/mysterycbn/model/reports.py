"""Concrete report and bundle objects (DATA_MODEL_SPEC.md §18–§20).

These three schemas are semver-governed public contracts (unlike the debug
snapshots of the other artifacts).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

from mysterycbn.model._utils import require, require_hex64


def _deep_json_safe(value: object) -> object:
    """Recursively convert nested ``Mapping``s (e.g. the ``MappingProxyType``
    tree ``LayeredResolver.as_mapping()`` returns) to plain ``dict``s so
    ``json.dumps`` never trips on a non-serializable mapping type buried
    below the top level."""
    if isinstance(value, Mapping):
        return {k: _deep_json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_deep_json_safe(v) for v in value]
    return value


class Severity(enum.Enum):
    """Finding severity (ENGINE_SPEC §25 / validate subsystem)."""

    INFO = "info"
    WARNING = "warning"
    REPAIRED = "repaired"
    FATAL = "fatal"


@dataclass(frozen=True)
class Finding:
    """One validation finding, locatable and severity-graded (DATA_MODEL_SPEC §18)."""

    severity: Severity
    invariant: str
    message: str
    location: str
    repair_applied: bool = False

    def __post_init__(self) -> None:
        require(bool(self.invariant), "invariant must be non-empty (e.g. 'I3')")
        require(bool(self.message), "message must be non-empty")
        require(bool(self.location), "location must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity.value,
            "invariant": self.invariant,
            "message": self.message,
            "location": self.location,
            "repair_applied": self.repair_applied,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of one validator (DATA_MODEL_SPEC §18).

    ``passed`` is derived: true iff no FATAL finding remains.
    """

    validator_name: str
    findings: tuple[Finding, ...]
    metrics: Mapping[str, float]
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        require(bool(self.validator_name), "validator_name must be non-empty")
        object.__setattr__(
            self,
            "passed",
            all(f.severity is not Severity.FATAL for f in self.findings),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "validator_name": self.validator_name,
            "findings": [f.to_dict() for f in self.findings],
            "metrics": dict(self.metrics),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RunReport:
    """Reproducibility record embedded in every OutputBundle (DATA_MODEL_SPEC §19)."""

    resolved_config: Mapping[str, object]
    engine_version: str
    input_hash: str
    seed: int
    warnings: tuple[str, ...]
    stage_timings_s: Mapping[str, float]
    validation: tuple[ValidationReport, ...]
    renumber_map: tuple[int, ...]

    def __post_init__(self) -> None:
        require(bool(self.engine_version), "engine_version must be non-empty")
        require_hex64(self.input_hash, "input_hash")
        require(self.seed >= 0, "seed must be ≥ 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "resolved_config": _deep_json_safe(self.resolved_config),
            "engine_version": self.engine_version,
            "input_hash": self.input_hash,
            "seed": self.seed,
            "warnings": list(self.warnings),
            "stage_timings_s": dict(self.stage_timings_s),
            "validation": [v.to_dict() for v in self.validation],
            "renumber_map": list(self.renumber_map),
        }


class MetricClass(enum.Enum):
    """Quality-metric class (QUALITY_SPEC §1.2)."""

    GATE = "gate"
    MONITOR = "monitor"


@dataclass(frozen=True)
class MetricResult:
    """One measured QM value against its band (DATA_MODEL_SPEC §20)."""

    value: float
    band: tuple[float, float]
    metric_class: MetricClass
    passed: bool

    def __post_init__(self) -> None:
        require(self.band[0] <= self.band[1], "band must be (lo, hi) with lo ≤ hi")

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "band": list(self.band),
            "class": self.metric_class.value,
            "pass": self.passed,
        }


@dataclass(frozen=True)
class QualityMetricsReport:
    """Sprint 23 output-quality measurements: region statistics, tiny
    regions, boundary smoothness, mean compactness, palette quality, label
    fit rate, label overlap rate, SVG/PDF validity, printability score
    (validate/quality_metrics.py). Purely observational -- unlike
    ``RunReport.validation``'s 4 canonical reports, nothing here can block
    ``OutputBundle`` construction; this is what ``metrics.json`` serializes.
    """

    metrics: Mapping[str, MetricResult]

    def to_dict(self) -> dict[str, object]:
        return {name: result.to_dict() for name, result in self.metrics.items()}


_PREVIEW_KEYS = frozenset({"lineart", "solved"})
_VALIDATOR_COUNT = 4  # fidelity, topology, printability, palette


@dataclass(frozen=True)
class OutputBundle:
    """Atomic final deliverable: all validated, or never constructed (DATA_MODEL_SPEC §19)."""

    svg: bytes
    pdf: bytes | None
    previews: Mapping[str, bytes]
    report: RunReport
    quality: QualityMetricsReport

    def __post_init__(self) -> None:
        require(len(self.svg) > 0, "svg must be non-empty")
        require(
            set(self.previews) == _PREVIEW_KEYS,
            f"previews keys must be exactly {sorted(_PREVIEW_KEYS)}",
        )
        require(
            len(self.report.validation) == _VALIDATOR_COUNT,
            f"report must embed exactly {_VALIDATOR_COUNT} validation reports",
        )
        require(
            all(v.passed for v in self.report.validation),
            "an OutputBundle may only exist when every validator passed",
        )

    def to_dict(self) -> dict[str, object]:
        """Byte payloads reported as sizes; the bundle bytes ARE the outputs."""
        return {
            "svg_bytes": len(self.svg),
            "pdf_bytes": len(self.pdf) if self.pdf is not None else None,
            "previews_bytes": {k: len(v) for k, v in self.previews.items()},
            "report": self.report.to_dict(),
            "quality": self.quality.to_dict(),
        }


class GoldenOutcome(enum.Enum):
    """Golden-comparison outcome (BENCHMARK_SPEC §4.2)."""

    IDENTICAL = "identical"
    CHANGED_COMPATIBLE = "changed_compatible"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class MachineFingerprint:
    """Benchmark environment identity (BENCHMARK_SPEC §9.1)."""

    cpu: str
    cores: int
    memory_gib: float
    container_digest: str
    kernel: str
    lockfile_hash: str
    canary_s: float

    def __post_init__(self) -> None:
        require(self.cores >= 1, "cores must be ≥ 1")
        require(self.memory_gib > 0.0, "memory_gib must be positive")
        require(self.canary_s > 0.0, "canary_s must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu": self.cpu,
            "cores": self.cores,
            "memory_gib": self.memory_gib,
            "container_digest": self.container_digest,
            "kernel": self.kernel,
            "lockfile_hash": self.lockfile_hash,
            "canary_s": self.canary_s,
        }


@dataclass(frozen=True)
class FailureTuple:
    """One acceptance failure, surfaced verbatim in the PR status (BENCHMARK_SPEC §8)."""

    metric: str
    fixture: str
    preset: str
    value: float
    band: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "fixture": self.fixture,
            "preset": self.preset,
            "value": self.value,
            "band": list(self.band),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """One benchmark run's complete result (DATA_MODEL_SPEC §20, BENCHMARK_SPEC §11).

    ``metrics`` is fixture → preset → QM-id → MetricResult; ``stages`` is
    fixture → preset → stage → {"wall_s": …, "rss_mib": …}.
    """

    run_id: str
    timestamp_utc: str
    git_sha: str
    engine_version: str
    machine: MachineFingerprint
    dataset_version: int
    score_version: int
    report_schema: int
    metrics: Mapping[str, Mapping[str, Mapping[str, MetricResult]]]
    stages: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]
    golden: Mapping[str, GoldenOutcome]
    score_total: float
    score_dimensions: Mapping[str, float]
    accepted: bool
    failures: tuple[FailureTuple, ...]

    def __post_init__(self) -> None:
        for name in ("run_id", "timestamp_utc", "git_sha", "engine_version"):
            require(bool(getattr(self, name)), f"{name} must be non-empty")
        require(self.dataset_version >= 1, "dataset_version must be ≥ 1")
        require(self.score_version >= 1, "score_version must be ≥ 1")
        require(self.report_schema >= 1, "report_schema must be ≥ 1")
        require(0.0 <= self.score_total <= 100.0, "score_total must be in [0, 100]")
        require(
            self.accepted == (len(self.failures) == 0),
            "accepted must hold iff failures is empty (BENCHMARK_SPEC §11)",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "git_sha": self.git_sha,
            "engine_version": self.engine_version,
            "machine": self.machine.to_dict(),
            "dataset_version": self.dataset_version,
            "score_version": self.score_version,
            "report_schema": self.report_schema,
            "metrics": {
                fixture: {
                    preset: {qm: m.to_dict() for qm, m in per_preset.items()}
                    for preset, per_preset in per_fixture.items()
                }
                for fixture, per_fixture in self.metrics.items()
            },
            "stages": {
                fixture: {
                    preset: {stage: dict(vals) for stage, vals in per_preset.items()}
                    for preset, per_preset in per_fixture.items()
                }
                for fixture, per_fixture in self.stages.items()
            },
            "golden": {k: v.value for k, v in self.golden.items()},
            "score": {"total": self.score_total, "dimensions": dict(self.score_dimensions)},
            "verdict": {
                "accepted": self.accepted,
                "failures": [f.to_dict() for f in self.failures],
            },
        }
