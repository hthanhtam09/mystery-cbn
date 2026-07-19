"""Validation aggregation: run all validators, decide pass/abort
(ENGINE_SPEC.md §25; OutputBundle requires exactly the 4 canonical reports
-- fidelity, topology, printability, palette -- per DATA_MODEL_SPEC §19).

Repair loop: a repaired run re-validates from scratch, up to
``max_repair_iterations`` (default 2) before giving up as FATAL (§25
"Quality requirements"). Only the printability validator currently declares
repairs (leader-line demotion); topology is never repaired (§25.2).

SVG/PDF structural conformance (QM-26..28) is a fifth, separate report --
``OutputBundle`` does not embed it (only the 4 canonical validators), so
callers that also render should check ``output_validity`` themselves before
constructing the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

from mysterycbn.foundation.errors import QualityError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.reports import ValidationReport
from mysterycbn.validate.fidelity import (
    FIDELITY_MIN_AGREEMENT_DEFAULT,
    FIDELITY_MIN_AGREEMENT_FILLER_DEFAULT,
    validate_fidelity,
)
from mysterycbn.validate.output_validity import validate_output_validity
from mysterycbn.validate.palette import PALETTE_WARN_DELTA_E_DEFAULT, validate_palette
from mysterycbn.validate.printability import D_MIN_MM_DEFAULT, validate_printability
from mysterycbn.validate.topology import validate_topology

MAX_REPAIR_ITERATIONS = 2
_VALIDATOR_ORDER = ("fidelity", "topology", "printability", "palette")


@dataclass(frozen=True)
class ValidationSettings:
    """The ``validate.*`` config section (ENGINE_SPEC §25 table)."""

    d_min_mm: float = D_MIN_MM_DEFAULT
    font_min_pt: float = 6.0
    palette_warn_delta_e: float = PALETTE_WARN_DELTA_E_DEFAULT
    fidelity_min_agreement: float = FIDELITY_MIN_AGREEMENT_DEFAULT
    fidelity_min_agreement_filler: float = FIDELITY_MIN_AGREEMENT_FILLER_DEFAULT
    merge_delta_e: float = 7.0
    warn_is_fatal: bool = False


def _run_once(ctx: PipelineContext, settings: ValidationSettings) -> tuple[ValidationReport, ...]:
    """One pass of all 4 canonical validators, in the order OutputBundle expects."""
    fidelity = validate_fidelity(
        ctx,
        fidelity_min_agreement=settings.fidelity_min_agreement,
        fidelity_min_agreement_filler=settings.fidelity_min_agreement_filler,
    )
    topology = validate_topology(ctx)
    printability = validate_printability(
        ctx,
        d_min_mm=settings.d_min_mm,
        font_min_pt=settings.font_min_pt,
    )
    palette = validate_palette(
        ctx,
        merge_delta_e=settings.merge_delta_e,
        palette_warn_delta_e=settings.palette_warn_delta_e,
        warn_is_fatal=settings.warn_is_fatal,
    )
    return (fidelity, topology, printability, palette)


def _has_repair(reports: tuple[ValidationReport, ...]) -> bool:
    return any(f.repair_applied for report in reports for f in report.findings)


def run_validation(
    ctx: PipelineContext,
    settings: ValidationSettings | None = None,
) -> tuple[ValidationReport, ...]:
    """Run the full I1-I4 validation gate with the declared repair loop
    (ENGINE_SPEC §25). Raises ``QualityError`` if any validator still has a
    FATAL finding after ``MAX_REPAIR_ITERATIONS`` repair-and-recheck passes.

    Returns the 4 canonical reports (fidelity, topology, printability,
    palette) in the order ``OutputBundle`` requires.
    """
    settings = settings or ValidationSettings()
    reports = _run_once(ctx, settings)
    iterations = 0
    while (
        not all(r.passed for r in reports)
        and _has_repair(reports)
        and iterations < MAX_REPAIR_ITERATIONS
    ):
        iterations += 1
        reports = _run_once(ctx, settings)

    if not all(r.passed for r in reports):
        failed = [r.validator_name for r in reports if not r.passed]
        raise QualityError(
            f"validation gate failed (unrepairable FATAL in: {failed})", reports=reports
        )

    return reports


def run_output_validity(ctx: PipelineContext) -> ValidationReport:
    """Run the SVG/PDF structural report (QM-26..28) -- separate from the
    4 canonical ``OutputBundle`` reports; call and check before rendering
    a final bundle."""
    report = validate_output_validity(ctx)
    return report
