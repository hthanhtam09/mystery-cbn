"""The engine's public entry point (ARCHITECTURE.md §5, public interface #1;
Sprint 19: "This becomes the only entry point for the engine").

    from mysterycbn.app.api import convert
    bundle = convert("examples/flower.jpg", preset="medium")
    bundle.svg, bundle.pdf, bundle.previews["lineart"], bundle.previews["solved"]
    bundle.report  # RunReport: resolved_config, stage_timings_s, validation, ...

``convert()`` is a thin functional wrapper over ``ConcreteOrchestrator`` --
kept separate from ``orchestrator_impl.py`` so the class (stateful only in
its ``page_mm`` choice) and the single free function developers actually
call are independently importable/testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mysterycbn.app.config_defaults import DEFAULT_PAGE_MM
from mysterycbn.app.orchestrator_impl import ConcreteOrchestrator, ConvertJobSpec
from mysterycbn.kernel.cancellation import CancelToken
from mysterycbn.kernel.progress import ProgressListener
from mysterycbn.model.reports import OutputBundle


def convert(
    source: str | Path | bytes,
    *,
    preset: str = "medium",
    overrides: Mapping[str, Any] | None = None,
    seed: int = 0,
    page_mm: tuple[float, float, float] = DEFAULT_PAGE_MM,
    on_progress: ProgressListener | None = None,
    cancel_token: CancelToken | None = None,
) -> OutputBundle:
    """Run the full pipeline: image bytes/path in, ``OutputBundle`` out.

    Atomic: either every artifact validated and an ``OutputBundle`` is
    returned, or an ``EngineError`` subclass is raised and nothing partial
    is exposed (ARCHITECTURE.md §11).

    Parameters mirror ``JobSpec`` (``app/jobs.py``); ``source`` may be a
    filesystem path (``str``/``Path``) or raw file bytes.
    """
    resolved_source = source if isinstance(source, bytes) else Path(source)
    spec = ConvertJobSpec(
        source=resolved_source, preset=preset, overrides=overrides or {}, seed=seed
    )
    orchestrator = ConcreteOrchestrator(page_mm=page_mm)
    return orchestrator.convert(spec, on_progress=on_progress, cancel_token=cancel_token)
