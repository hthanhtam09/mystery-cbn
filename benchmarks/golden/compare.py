"""Combined golden comparison: perceptual + SVG structural + topology
(docs/GOLDEN_TEST_STANDARDS.md §5).

Reuses ``benchmarks/framework/visual.py``'s byte-hash/structural-diff/SSIM
primitives (BENCHMARK_SPEC.md §4.2) rather than re-implementing them, and
adds the topology axis (``topology_compare.py``) that framework doesn't
cover. One ``GoldenReport`` per fixture records all three outcomes plus the
tolerances used, so a report is self-describing without cross-referencing
config.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.framework.pipeline import PipelineRun
from benchmarks.framework.visual import (
    _decode_png,
    _rasterize_preview,
    _ssim,
    _structural_diff_ok,
)
from benchmarks.golden import storage
from benchmarks.golden.tolerances import DEFAULT_TOLERANCES, GoldenTolerances
from benchmarks.golden.topology_compare import TopologyComparison, compare_topology
from mysterycbn.model.reports import GoldenOutcome


@dataclass(frozen=True)
class GoldenReport:
    """One fixture's full golden-comparison outcome."""

    fixture_id: str
    category: str
    svg_outcome: GoldenOutcome
    ssim_solved: float | None
    topology: TopologyComparison | None
    tolerances: GoldenTolerances
    details: dict[str, object]

    @property
    def passed(self) -> bool:
        topology_ok = self.topology is None or self.topology.passed
        return self.svg_outcome is not GoldenOutcome.INCOMPATIBLE and topology_ok

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "category": self.category,
            "passed": self.passed,
            "svg_outcome": self.svg_outcome.value,
            "ssim_solved": self.ssim_solved,
            "topology": self.topology.to_dict() if self.topology is not None else None,
            "tolerances": self.tolerances.to_dict(),
            "details": self.details,
        }


def compare_run_to_golden(
    run: PipelineRun,
    *,
    category: str,
    tolerances: GoldenTolerances = DEFAULT_TOLERANCES,
) -> GoldenReport:
    """Compare one pipeline run against its stored golden. Missing golden
    is reported as INCOMPATIBLE with a clear reason -- never silently
    treated as a pass (BENCHMARK_SPEC.md §4.2's outcome enum)."""
    fixture_id = run.fixture_id
    if not storage.has_golden(fixture_id):
        return GoldenReport(
            fixture_id=fixture_id,
            category=category,
            svg_outcome=GoldenOutcome.INCOMPATIBLE,
            ssim_solved=None,
            topology=None,
            tolerances=tolerances,
            details={"reason": "no golden recorded for this fixture"},
        )

    golden_svg = storage.read_golden_svg(fixture_id)
    if golden_svg == run.svg_bytes:
        svg_outcome = GoldenOutcome.IDENTICAL
        details: dict[str, object] = {"byte_identical": True}
    else:
        ok, diff_details = _structural_diff_ok(golden_svg, run.svg_bytes)
        svg_outcome = GoldenOutcome.CHANGED_COMPATIBLE if ok else GoldenOutcome.INCOMPATIBLE
        details = {"byte_identical": False, **diff_details}

    ssim_solved: float | None = None
    golden_preview = storage.read_golden_preview(fixture_id)
    if golden_preview is not None and run.pdf_bytes is not None:
        golden_luma = _decode_png(golden_preview)
        candidate_luma = _rasterize_preview(run.pdf_bytes)
        if candidate_luma is not None:
            ssim_solved = _ssim(golden_luma, candidate_luma)
            if ssim_solved < tolerances.ssim_min and svg_outcome is not GoldenOutcome.IDENTICAL:
                svg_outcome = GoldenOutcome.INCOMPATIBLE
            details["ssim_threshold"] = tolerances.ssim_min

    golden_topology = storage.read_golden_topology(fixture_id)
    topology = compare_topology(golden_topology, run, tolerances=tolerances)

    return GoldenReport(
        fixture_id=fixture_id,
        category=category,
        svg_outcome=svg_outcome,
        ssim_solved=ssim_solved,
        topology=topology,
        tolerances=tolerances,
        details=details,
    )
