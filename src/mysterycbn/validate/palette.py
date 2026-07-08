"""Palette Validator (ENGINE_SPEC.md §25.4; QM-16 Palette Separation).

Min pairwise ΔE00 must clear ``quantize.merge_delta_e`` (a violation here
indicates a §7 quantize-stage bug -- construction re-check, FATAL). A
separation below ``palette_warn_delta_e`` (default 12) is only a WARNING
("colors hard to distinguish for young solvers") -- except for preset
``easy``, where it is promoted to FATAL.
"""

from __future__ import annotations

import numpy as np

from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Palette
from mysterycbn.model.reports import Finding, Severity, ValidationReport

VALIDATOR_NAME = "palette"
PALETTE_WARN_DELTA_E_DEFAULT = 12.0


def validate_palette(
    ctx: PipelineContext,
    *,
    merge_delta_e: float = 7.0,
    palette_warn_delta_e: float = PALETTE_WARN_DELTA_E_DEFAULT,
    warn_is_fatal: bool = False,
) -> ValidationReport:
    """Run the QM-16 separation checks against the bound ``palette``."""
    palette = ctx.get("palette")
    assert isinstance(palette, Palette)

    table = palette.delta_e_table
    k = palette.size
    off_diag = table[~np.eye(k, dtype=bool)]
    min_delta_e = float(off_diag.min()) if off_diag.size else float("inf")

    findings: list[Finding] = []
    if min_delta_e < merge_delta_e:
        findings.append(
            Finding(
                severity=Severity.FATAL,
                invariant="palette",
                message=(
                    f"min pairwise ΔE00 {min_delta_e:.2f} below construction floor "
                    f"{merge_delta_e} (indicates a quantize-stage bug)"
                ),
                location="palette",
            )
        )
    elif min_delta_e < palette_warn_delta_e:
        severity = Severity.FATAL if warn_is_fatal else Severity.WARNING
        findings.append(
            Finding(
                severity=severity,
                invariant="palette",
                message=(
                    f"min pairwise ΔE00 {min_delta_e:.2f} below warn threshold "
                    f"{palette_warn_delta_e} -- colors may be hard to distinguish"
                ),
                location="palette",
            )
        )

    metrics = {"min_delta_e": min_delta_e}
    return ValidationReport(
        validator_name=VALIDATOR_NAME, findings=tuple(findings), metrics=metrics
    )
