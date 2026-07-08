"""Output Validity Validator: SVG/PDF structural conformance
(ENGINE_SPEC.md §22-23, QUALITY_SPEC.md QM-26..28).

Wraps the render module's own structural checks (``render.svg.validate_svg``,
``render.pdf.validate_pdf``) as ``Finding``s rather than raw exceptions, so a
render defect surfaces through the same ``ValidationReport`` channel as every
other invariant instead of aborting the run with an undifferentiated
``StageError``.
"""

from __future__ import annotations

from mysterycbn.foundation.errors import StageError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.reports import Finding, Severity, ValidationReport
from mysterycbn.model.vector import CurveSet
from mysterycbn.render.pdf import validate_pdf
from mysterycbn.render.svg import validate_svg

VALIDATOR_NAME = "output_validity"


def _bytes_of(ctx: PipelineContext, artifact_name: str) -> bytes:
    """The ``.data`` payload of a rendered-output artifact (SvgDocument /
    PdfDocument both carry it; the pipeline context is typed generically as
    ``Artifact``, so this narrows the access to one place)."""
    doc = ctx.get(artifact_name)
    return bytes(doc.data)  # type: ignore[attr-defined]


def validate_output_validity(ctx: PipelineContext) -> ValidationReport:
    """Run QM-26 (SVG) and QM-28 (PDF) structural checks against whatever
    output artifacts are currently bound (``svg`` is required; ``pdf`` is
    optional, matching ``OutputBundle.pdf``'s optionality)."""
    findings: list[Finding] = []
    curve_set_artifact = ctx.get("curve_set") if ctx.has("curve_set") else None
    curve_set = curve_set_artifact if isinstance(curve_set_artifact, CurveSet) else None

    if ctx.has("svg"):
        try:
            validate_svg(_bytes_of(ctx, "svg"), curve_set)
        except StageError as exc:
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I2",
                    message=f"SVG structural validity: {exc}",
                    location="svg",
                )
            )
    else:
        findings.append(
            Finding(
                severity=Severity.FATAL,
                invariant="I2",
                message="no svg artifact bound",
                location="svg",
            )
        )

    if ctx.has("pdf"):
        try:
            validate_pdf(_bytes_of(ctx, "pdf"))
        except StageError as exc:
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I2",
                    message=f"PDF structural validity: {exc}",
                    location="pdf",
                )
            )

    metrics = {"svg_bytes": float(len(_bytes_of(ctx, "svg"))) if ctx.has("svg") else 0.0}
    return ValidationReport(
        validator_name=VALIDATOR_NAME, findings=tuple(findings), metrics=metrics
    )
