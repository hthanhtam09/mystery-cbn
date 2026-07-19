"""Drives the real 14-stage pipeline (identical wiring to
``mysterycbn.app.orchestrator_impl.ConcreteOrchestrator.convert()``) but,
unlike ``convert()``, keeps the populated ``InMemoryContext`` and the
executed stage list around afterward so every intermediate artifact can be
inspected -- the one thing the public ``convert()`` API deliberately
doesn't expose (ARCHITECTURE.md §11 atomicity: the public entry point
returns only the validated ``OutputBundle``, nothing partial).

This is a developer tool, not engine code: it duplicates
``ConcreteOrchestrator``'s plan-building glue rather than modifying the
orchestrator to add a debug hook, so no engine file changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mysterycbn.app.config_defaults import (
    D_MIN_MM_BY_PRESET,
    DEFAULT_PAGE_MM,
    builtin_defaults,
    difficulty_preset,
)
from mysterycbn.app.orchestrator_impl import ConvertJobSpec
from mysterycbn.app.registry_bootstrap import build_registry
from mysterycbn.foundation.config.resolver import LayeredResolver
from mysterycbn.foundation.config.schema import ConfigLayer
from mysterycbn.foundation.errors import EngineError, InputError
from mysterycbn.foundation.tracing import InMemoryTracer
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.kernel.pipeline import DefaultPlanResolver, SequentialExecutor
from mysterycbn.kernel.stage import Stage
from mysterycbn.stages.raster.load import SourceBytes
from mysterycbn.validate.report import ValidationSettings, run_output_validity, run_validation


def _read_source(source: Path | bytes) -> bytes:
    if isinstance(source, bytes):
        return source
    path = Path(source)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc


@dataclass(frozen=True)
class DebugRun:
    """Everything the debugger's report needs: the populated context, the
    stages that ran (in execution order, each with its ``provides`` list),
    and per-stage wall time from the same tracer the real orchestrator
    uses."""

    ctx: InMemoryContext
    stages: tuple[Stage, ...]
    stage_timings_s: dict[str, float]
    validation_passed: bool


def run_pipeline_for_debug(
    source: Path | bytes,
    *,
    preset: str = "medium",
    seed: int = 0,
    page_mm: tuple[float, float, float] = DEFAULT_PAGE_MM,
) -> DebugRun:
    """Run the full pipeline exactly as ``convert()`` does, returning the
    populated context instead of collapsing it into an ``OutputBundle``.
    Raises whatever the pipeline raises -- a debugger must show the real
    failure, not paper over it."""
    data = _read_source(source)
    source_bytes = SourceBytes(data)

    layers = {
        ConfigLayer.BUILTIN_DEFAULTS: builtin_defaults(),
        ConfigLayer.DIFFICULTY_PRESET: difficulty_preset(preset),
        ConfigLayer.PROGRAMMATIC: {},
    }
    resolved = LayeredResolver().resolve(layers)
    d_min_mm = D_MIN_MM_BY_PRESET[preset]
    quality_section = resolved.stage_section("quality")
    font_min_pt = float(quality_section.get("font_min_pt", 6.0))
    merge_delta_e = float(resolved.stage_section("quantize").get("merge_delta_e", 7.0))

    # Mirror ConcreteOrchestrator.convert(): each stage must receive its
    # resolved config section, otherwise preset overlays (organic.enabled,
    # split.enabled, quantize.merge_delta_e, ...) silently never reach the
    # stages and the debugger does not reproduce the real pipeline.
    def _section(name: str) -> dict[str, object]:
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
        seed=seed,
        config_hash=resolved.config_hash,
        page_mm=page_mm,
        font_min_pt=font_min_pt,
        sections=stage_sections,
    )
    plan = DefaultPlanResolver(registry, initial_artifacts=["source_bytes"]).resolve(resolved)

    ctx = InMemoryContext(seed=seed, config_hash=resolved.config_hash)
    ctx.put("source_bytes", source_bytes)

    tracer = InMemoryTracer()
    executor = SequentialExecutor(tracer=tracer)
    executor.execute(ctx=ctx, plan=plan, on_progress=None, cancel_token=None)

    validation_settings = ValidationSettings(
        d_min_mm=d_min_mm, font_min_pt=font_min_pt, merge_delta_e=merge_delta_e
    )
    validation_reports = run_validation(ctx, validation_settings)
    output_validity_report = run_output_validity(ctx)
    validation_passed = all(r.passed for r in validation_reports) and output_validity_report.passed

    snapshot = tracer.snapshot()
    timings = snapshot["timings_s"]
    assert isinstance(timings, Mapping)
    stage_timings = {str(k): float(v) for k, v in timings.items()}

    return DebugRun(
        ctx=ctx,
        stages=plan.stages,
        stage_timings_s=stage_timings,
        validation_passed=validation_passed,
    )


def debug_convert_job_spec(
    spec: ConvertJobSpec, *, page_mm: tuple[float, float, float]
) -> DebugRun:
    return run_pipeline_for_debug(spec.source, preset=spec.preset, seed=spec.seed, page_mm=page_mm)
