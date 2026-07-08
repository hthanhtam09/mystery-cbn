"""Validation subsystem: proves invariants I1–I4 (ARCHITECTURE.md §0, §6).

Validators never mutate artifacts except through declared, logged repairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mysterycbn.model.context import PipelineContext
from mysterycbn.model.reports import Severity

__all__ = ["Finding", "Severity", "ValidationReport", "Validator"]


@runtime_checkable
class Finding(Protocol):
    """One validation finding, locatable and severity-graded."""

    @property
    def severity(self) -> Severity: ...

    @property
    def invariant(self) -> str:
        """Which invariant this concerns (e.g. ``"I3"``)."""
        ...

    @property
    def message(self) -> str: ...

    @property
    def location(self) -> str:
        """Human-locatable reference (region id, arc id, page coordinates)."""
        ...

    @property
    def repair_applied(self) -> bool: ...


@runtime_checkable
class ValidationReport(Protocol):
    @property
    def validator_name(self) -> str: ...

    @property
    def findings(self) -> Sequence[Finding]: ...

    @property
    def passed(self) -> bool:
        """True iff no FATAL finding remains after declared repairs."""
        ...


@runtime_checkable
class Validator(Protocol):
    """Public plugin interface: consumes the context, returns a structured report."""

    @property
    def name(self) -> str: ...

    def validate(self, ctx: PipelineContext) -> ValidationReport: ...


from mysterycbn.validate.fidelity import validate_fidelity  # noqa: E402
from mysterycbn.validate.output_validity import validate_output_validity  # noqa: E402
from mysterycbn.validate.palette import validate_palette  # noqa: E402
from mysterycbn.validate.printability import validate_printability  # noqa: E402
from mysterycbn.validate.quality_metrics import compute_quality_metrics  # noqa: E402
from mysterycbn.validate.report import (  # noqa: E402
    ValidationSettings,
    run_output_validity,
    run_validation,
)
from mysterycbn.validate.topology import validate_topology  # noqa: E402

__all__ += [
    "ValidationSettings",
    "compute_quality_metrics",
    "run_output_validity",
    "run_validation",
    "validate_fidelity",
    "validate_output_validity",
    "validate_palette",
    "validate_printability",
    "validate_topology",
]
