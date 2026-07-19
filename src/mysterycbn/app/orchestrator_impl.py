"""Concrete Orchestrator: the Sprint 19 orchestration layer.

Fills the gap the Sprint 18 architecture audit identified as the sole
blocker to end-to-end conversion: ``app/orchestrator.py``'s ``Orchestrator``
was an ``ABC`` with a single ``@abstractmethod`` and no concrete subclass
anywhere in the codebase; ``adapters/cli``/``adapters/api`` were empty
docstring-only files; no code path registered any stage into
``InMemoryStageRegistry``; ``SequentialExecutor``/``DefaultPlanResolver``
were never instantiated by any caller or test.

This module changes none of that infrastructure -- it *uses* it. The
14-stage sequential pipeline (Load through PNG) runs entirely through the
existing ``DefaultPlanResolver`` + ``SequentialExecutor`` (ARCHITECTURE.md
§4.2), with real progress events, real cancellation checks, and real
per-stage tracing. Two steps do not fit the Stage protocol (a validator
raises ``QualityError`` instead of providing an artifact; final bundle
assembly combines several artifacts into one atomic object) and are
therefore orchestrated directly by ``ConcreteOrchestrator.convert()``,
exactly at the point ARCHITECTURE.md §1.1 places them: *between* stage
execution and rendering for the four canonical gates, and *after*
rendering for the SVG/PDF structural conformance check.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mysterycbn import __version__ as ENGINE_VERSION
from mysterycbn.app.config_defaults import (
    D_MIN_MM_BY_PRESET,
    DEFAULT_PAGE_MM,
    builtin_defaults,
    difficulty_preset,
)
from mysterycbn.app.jobs import JobSpec
from mysterycbn.app.orchestrator import Orchestrator
from mysterycbn.app.registry_bootstrap import build_registry
from mysterycbn.foundation.config.resolver import LayeredResolver
from mysterycbn.foundation.config.schema import ConfigLayer
from mysterycbn.foundation.errors import EngineError, InputError
from mysterycbn.foundation.tracing import InMemoryTracer
from mysterycbn.kernel.cancellation import CancelToken
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.kernel.pipeline import DefaultPlanResolver, SequentialExecutor
from mysterycbn.kernel.progress import ProgressListener
from mysterycbn.model.reports import OutputBundle, QualityMetricsReport, RunReport
from mysterycbn.render.pdf import PdfDocument
from mysterycbn.render.png import PngPreviews
from mysterycbn.render.svg import SvgDocument
from mysterycbn.stages.raster.load import SourceBytes
from mysterycbn.validate.quality_metrics import compute_quality_metrics
from mysterycbn.validate.report import ValidationSettings, run_output_validity, run_validation


def _read_source(source: Path | bytes) -> bytes:
    """Accept a filesystem path (``str``/``Path``) or raw ``bytes``."""
    if isinstance(source, bytes):
        return source
    path = Path(source)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc


@dataclass(frozen=True)
class ConvertJobSpec:
    """Concrete ``JobSpec`` for a single ``convert()`` call."""

    source: Path | bytes
    preset: str = "medium"
    overrides: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0


class ConcreteOrchestrator(Orchestrator):
    """Runs the full 14-stage pipeline via the kernel executor, then the
    two non-Stage steps (validation gate, output-bundle assembly)."""

    def __init__(self, *, page_mm: tuple[float, float, float] = DEFAULT_PAGE_MM) -> None:
        self._page_mm = page_mm

    def convert(
        self,
        spec: JobSpec,
        *,
        on_progress: ProgressListener | None = None,
        cancel_token: CancelToken | None = None,
    ) -> OutputBundle:
        data = _read_source(spec.source)
        source_bytes = SourceBytes(data)

        layers = {
            ConfigLayer.BUILTIN_DEFAULTS: builtin_defaults(),
            ConfigLayer.DIFFICULTY_PRESET: difficulty_preset(spec.preset),
            ConfigLayer.PROGRAMMATIC: dict(spec.overrides),
        }
        resolved = LayeredResolver().resolve(layers)
        d_min_mm = D_MIN_MM_BY_PRESET[spec.preset]
        quality_section = resolved.stage_section("quality")
        font_min_pt = float(quality_section.get("font_min_pt", 6.0))
        merge_delta_e = float(resolved.stage_section("quantize").get("merge_delta_e", 7.0))

        # Feed each stage its resolved config section so preset/override knobs
        # (quantize.n_colors, merge.enabled, split.enabled, ...) actually reach
        # the stage that reads them -- the factory otherwise builds every stage
        # with an empty section and silently ignores those overlays.
        def _section(name: str) -> dict[str, Any]:
            try:
                return dict(resolved.stage_section(name))
            except EngineError:
                return {}

        stage_sections = {
            name: _section(name)
            for name in (
                "preprocess",
                "analyze",
                "quantize",
                "denoise",
                "merge",
                "organic",
                "split",
                "simplify",
                "bezier",
                "labels",
            )
        }

        registry = build_registry(
            d_min_mm=d_min_mm,
            seed=spec.seed,
            config_hash=resolved.config_hash,
            page_mm=self._page_mm,
            font_min_pt=font_min_pt,
            sections=stage_sections,
        )
        plan = DefaultPlanResolver(registry, initial_artifacts=["source_bytes"]).resolve(resolved)

        ctx = InMemoryContext(seed=spec.seed, config_hash=resolved.config_hash)
        ctx.put("source_bytes", source_bytes)

        tracer = InMemoryTracer()
        executor = SequentialExecutor(tracer=tracer)

        t0 = time.perf_counter()
        try:
            executor.execute(ctx=ctx, plan=plan, on_progress=on_progress, cancel_token=cancel_token)
        except EngineError:
            raise
        except Exception as exc:  # pragma: no cover - defensive; stages already wrap their own
            raise EngineError(f"pipeline execution failed: {exc}") from exc

        # Two steps outside the Stage protocol (see module docstring):
        # 1. The four canonical validators -- a QualityError here means no
        #    OutputBundle exists at all (atomicity, ARCHITECTURE.md §11).
        validate_section = _section("validate")
        validation_settings = ValidationSettings(
            d_min_mm=d_min_mm,
            font_min_pt=font_min_pt,
            merge_delta_e=merge_delta_e,
            fidelity_min_agreement=float(
                validate_section.get(
                    "fidelity_min_agreement", ValidationSettings.fidelity_min_agreement
                )
            ),
            fidelity_min_agreement_filler=float(
                validate_section.get(
                    "fidelity_min_agreement_filler",
                    ValidationSettings.fidelity_min_agreement_filler,
                )
            ),
        )
        validation_reports = run_validation(ctx, validation_settings)

        # 2. SVG/PDF structural conformance (QM-26..28) -- checked but not
        #    embedded in the bundle (DATA_MODEL_SPEC §19 embeds only the 4
        #    canonical reports); a failure here is still fatal to the run.
        output_validity_report = run_output_validity(ctx)
        if not output_validity_report.passed:
            fatal = [f.message for f in output_validity_report.findings if not f.repair_applied]
            raise EngineError(f"output validity check failed: {fatal}")

        svg_doc = ctx.get("svg")
        pdf_doc = ctx.get("pdf")
        png_previews = ctx.get("png_previews")
        if not isinstance(svg_doc, SvgDocument):
            raise EngineError(f"'svg' artifact has unexpected type {type(svg_doc).__name__}")
        if not isinstance(pdf_doc, PdfDocument):
            raise EngineError(f"'pdf' artifact has unexpected type {type(pdf_doc).__name__}")
        if not isinstance(png_previews, PngPreviews):
            raise EngineError(
                f"'png_previews' artifact has unexpected type {type(png_previews).__name__}"
            )

        elapsed = time.perf_counter() - t0
        snapshot = tracer.snapshot()
        raw_timings = snapshot["timings_s"]
        assert isinstance(raw_timings, Mapping)
        stage_timings: dict[str, float] = {k: float(v) for k, v in raw_timings.items()}
        stage_timings["_total_s"] = elapsed

        report = RunReport(
            resolved_config=resolved.as_mapping(),
            engine_version=ENGINE_VERSION,
            input_hash=source_bytes.provenance.source_hash,
            seed=spec.seed,
            warnings=(),
            stage_timings_s=stage_timings,
            validation=validation_reports,
            renumber_map=(),
        )

        # 3. Sprint 23 quality metrics -- purely observational (never
        #    blocks the bundle), computed after the canonical validators so
        #    ``label_plan`` reflects any printability repair. Reuses
        #    ``printability``'s own tiny-region measurement rather than
        #    re-deriving it a second time (see quality_metrics.py docstring).
        printability_report = next(
            (r for r in validation_reports if r.validator_name == "printability"), None
        )
        quality_metrics = compute_quality_metrics(
            ctx,
            printability_metrics=dict(printability_report.metrics)
            if printability_report is not None
            else None,
        )
        quality = QualityMetricsReport(metrics=quality_metrics)

        return OutputBundle(
            svg=svg_doc.data,
            pdf=pdf_doc.data,
            previews=dict(png_previews.previews),
            report=report,
            quality=quality,
        )
