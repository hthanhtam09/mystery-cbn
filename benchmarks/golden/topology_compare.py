"""Topology comparison between a golden fingerprint and a candidate run
(docs/GOLDEN_TEST_STANDARDS.md §5).

Distinct from ``mysterycbn.validate.topology.validate_topology``, which
re-proves QM-01/QM-02 invariants (watertightness, pair-constancy) on a
*single* run. This module diffs a frozen golden's region/arc/face counts
against a fresh run to catch the "engine produced structurally different
topology" case that a byte/SSIM comparison alone can miss (e.g. same visual
appearance, different region count from a merge-threshold change) -- an
axis BENCHMARK_SPEC.md §4.2's SVG/PNG comparisons don't cover.

The golden side is a small JSON fingerprint (``TopologyFingerprint``), not a
live pipeline run: goldens are persisted to disk (bless-time only,
BENCHMARK_SPEC.md §4.3), so the comparison must work from what's on disk.
Region/arc/face counts are analytic ground truth for a fixed, deterministic
fixture (BENCHMARK_SPEC.md §3): re-running the same pipeline on the same
input must reproduce the same topology exactly, so the default tolerance is
zero (``tolerances.GoldenTolerances.topology_region_count_tolerance``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from benchmarks.framework.pipeline import PipelineRun
from benchmarks.golden.tolerances import DEFAULT_TOLERANCES, GoldenTolerances
from mysterycbn.model.records import RegionGraph


@dataclass(frozen=True)
class TopologyFingerprint:
    """The persisted, comparable summary of one run's topology."""

    region_count: int
    arc_count: int
    face_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "region_count": self.region_count,
            "arc_count": self.arc_count,
            "face_count": self.face_count,
        }

    @staticmethod
    def from_dict(d: Mapping[str, int]) -> TopologyFingerprint:
        return TopologyFingerprint(
            region_count=d["region_count"],
            arc_count=d["arc_count"],
            face_count=d["face_count"],
        )


def fingerprint_run(run: PipelineRun) -> TopologyFingerprint:
    """Extract the comparable topology summary from a pipeline run."""
    region_graph = run.region_graph
    assert isinstance(region_graph, RegionGraph)
    return TopologyFingerprint(
        region_count=len(region_graph.regions),
        arc_count=len(run.arc_graph.arcs),
        face_count=len(run.arc_graph.faces),
    )


@dataclass(frozen=True)
class TopologyComparison:
    """Region/arc/face count delta between a golden fingerprint and a
    candidate run."""

    passed: bool
    golden: TopologyFingerprint
    candidate: TopologyFingerprint
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "golden": self.golden.to_dict(),
            "candidate": self.candidate.to_dict(),
            "details": self.details,
        }


def _within_tolerance(golden: int, candidate: int, tolerance: float) -> bool:
    if golden == candidate:
        return True
    if golden == 0:
        return candidate == 0
    return abs(candidate - golden) / golden <= tolerance


def compare_topology(
    golden: TopologyFingerprint,
    candidate_run: PipelineRun,
    *,
    tolerances: GoldenTolerances = DEFAULT_TOLERANCES,
) -> TopologyComparison:
    """Compare a frozen golden fingerprint against a fresh pipeline run. A
    count outside its tolerance band fails the comparison even if the
    SVG/perceptual checks pass -- topology is an independent axis."""
    candidate = fingerprint_run(candidate_run)

    region_ok = _within_tolerance(
        golden.region_count, candidate.region_count, tolerances.topology_region_count_tolerance
    )
    arc_ok = _within_tolerance(
        golden.arc_count, candidate.arc_count, tolerances.topology_arc_count_tolerance
    )
    face_ok = _within_tolerance(
        golden.face_count, candidate.face_count, tolerances.topology_region_count_tolerance
    )

    return TopologyComparison(
        passed=region_ok and arc_ok and face_ok,
        golden=golden,
        candidate=candidate,
        details={
            "region_count_ok": region_ok,
            "arc_count_ok": arc_ok,
            "face_count_ok": face_ok,
            "region_count_tolerance": tolerances.topology_region_count_tolerance,
            "arc_count_tolerance": tolerances.topology_arc_count_tolerance,
        },
    )
