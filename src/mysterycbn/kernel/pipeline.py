"""Plan resolution and execution (ARCHITECTURE.md §4.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mysterycbn.foundation.config.schema import ResolvedConfig
from mysterycbn.foundation.errors import CancelledError, ConfigError, EngineError, StageError
from mysterycbn.foundation.tracing import Tracer
from mysterycbn.kernel.cache import ArtifactCache, chain_hash, section_hash, stage_cache_key
from mysterycbn.kernel.cancellation import CancelToken
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.kernel.progress import ProgressKind, ProgressListener, ProgressUpdate
from mysterycbn.kernel.registry import StageRegistry
from mysterycbn.kernel.stage import Stage
from mysterycbn.model.context import PipelineContext


@runtime_checkable
class Plan(Protocol):
    """An ordered stage list whose requires/provides chain has been statically validated."""

    @property
    def stages(self) -> Sequence[Stage]: ...

    @property
    def config(self) -> ResolvedConfig: ...


class PlanResolver(ABC):
    """Builds a Plan from config + registry; fails fast on an unsatisfiable artifact chain."""

    @abstractmethod
    def resolve(self, config: ResolvedConfig) -> Plan:
        """Raises ConfigError if the requires/provides chain cannot be satisfied."""


class PipelineExecutor(ABC):
    """Runs a Plan sequentially and deterministically, checking cancellation between stages."""

    @abstractmethod
    def execute(
        self,
        plan: Plan,
        ctx: PipelineContext,
        *,
        on_progress: ProgressListener | None,
        cancel_token: CancelToken | None,
    ) -> PipelineContext:
        """Return the context with all provided artifacts bound.

        Raises StageError or CancelledError.
        """


@dataclass(frozen=True)
class ResolvedPlan:
    """Concrete Plan: an ordered stage list with a statically valid artifact chain."""

    stages: tuple[Stage, ...]
    config: ResolvedConfig


class DefaultPlanResolver(PlanResolver):
    """Builds a plan from the config's ``pipeline.stages`` slot list.

    Implementation per slot comes from that stage's own config section key
    ``impl`` (default ``"default"``) — selection by configuration, never by
    import (ARCHITECTURE.md §8). The requires/provides chain is validated
    statically against ``initial_artifacts`` (bound by the orchestrator before
    execution, e.g. the source bytes).
    """

    def __init__(self, registry: StageRegistry, *, initial_artifacts: Sequence[str] = ()) -> None:
        self._registry = registry
        self._initial = tuple(initial_artifacts)

    def _impl_name(self, config: ResolvedConfig, slot: str) -> str:
        try:
            section = config.stage_section(slot)
        except ConfigError:
            return "default"
        impl = section.get("impl", "default")
        if not isinstance(impl, str):
            raise ConfigError(f"stage {slot!r}: 'impl' must be a string, got {impl!r}")
        return impl

    def resolve(self, config: ResolvedConfig) -> ResolvedPlan:
        pipeline = config.stage_section("pipeline")
        slots = pipeline.get("stages")
        if not isinstance(slots, Sequence) or not slots or isinstance(slots, str):
            raise ConfigError("config section 'pipeline' must define a non-empty 'stages' list")

        stages: list[Stage] = []
        available = set(self._initial)
        for slot in slots:
            if not isinstance(slot, str):
                raise ConfigError(f"pipeline stage names must be strings, got {slot!r}")
            stage = self._registry.lookup(slot, self._impl_name(config, slot))
            missing = [name for name in stage.requires if name not in available]
            if missing:
                raise ConfigError(
                    f"stage {slot!r} requires artifacts {missing} that no earlier "
                    f"stage provides (available: {sorted(available)})"
                )
            available.update(stage.provides)
            stages.append(stage)
        return ResolvedPlan(stages=tuple(stages), config=config)


class SequentialExecutor(PipelineExecutor):
    """Sequential, deterministic executor (ARCHITECTURE.md §4.2, §13.5).

    Between stages it checks the cancellation token and emits progress
    events. With a cache attached, a stage whose full ``provides`` set is
    cached under its content key is skipped entirely; the cached and computed
    paths are required to be indistinguishable (I2).
    """

    def __init__(self, *, tracer: Tracer | None = None, cache: ArtifactCache | None = None) -> None:
        self._tracer = tracer
        self._cache = cache

    def execute(
        self,
        plan: Plan,
        ctx: PipelineContext,
        *,
        on_progress: ProgressListener | None = None,
        cancel_token: CancelToken | None = None,
    ) -> PipelineContext:
        total = len(plan.stages)
        upstream = ""
        for index, stage in enumerate(plan.stages):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise CancelledError(f"cancelled before stage {stage.name!r}")
            if isinstance(ctx, InMemoryContext):
                ctx.enter_stage(stage.name)
            self._emit(on_progress, ProgressKind.STAGE_STARTED, stage.name, index / total)

            upstream = chain_hash(
                upstream, stage.name, stage.version, self._section_hash(plan, stage)
            )
            key = stage_cache_key(ctx.seed, self._source_hash(ctx), upstream)

            if not self._restore_from_cache(ctx, stage, key):
                self._run_stage(plan, ctx, stage)
                if self._cache is not None:
                    for name in stage.provides:
                        self._cache.put(key, name, ctx.get(name))
            self._emit(on_progress, ProgressKind.STAGE_FINISHED, stage.name, (index + 1) / total)
        return ctx

    def _run_stage(self, plan: Plan, ctx: PipelineContext, stage: Stage) -> None:
        """Run one stage with fail-fast contract checks and error wrapping."""
        config_hash = plan.config.config_hash
        for name in stage.requires:
            if not ctx.has(name):
                raise StageError(
                    f"required artifact {name!r} missing at runtime",
                    stage_name=stage.name,
                    config_hash=config_hash,
                )
        try:
            if self._tracer is not None:
                with self._tracer.span(stage.name):
                    stage.run(ctx)
            else:
                stage.run(ctx)
        except EngineError:
            raise
        except Exception as exc:
            raise StageError(
                f"stage failed: {exc}", stage_name=stage.name, config_hash=config_hash
            ) from exc
        for name in stage.provides:
            if not ctx.has(name):
                raise StageError(
                    f"stage completed without binding promised artifact {name!r}",
                    stage_name=stage.name,
                    config_hash=config_hash,
                )

    def _restore_from_cache(self, ctx: PipelineContext, stage: Stage, key: str) -> bool:
        """Bind all provided artifacts from cache; False on any miss."""
        if self._cache is None or not stage.provides:
            return False
        cached = [(name, self._cache.get(key, name)) for name in stage.provides]
        if any(artifact is None for _, artifact in cached):
            return False
        for name, artifact in cached:
            assert artifact is not None  # narrowed by the check above
            ctx.put(name, artifact)
        return True

    @staticmethod
    def _section_hash(plan: Plan, stage: Stage) -> str:
        try:
            return section_hash(plan.config.stage_section(stage.config_section))
        except ConfigError:
            return section_hash({})

    @staticmethod
    def _source_hash(ctx: PipelineContext) -> str:
        """Source identity for cache keys: the first bound artifact's provenance."""
        for name in ctx.names():
            return ctx.get(name).provenance.source_hash
        return "0" * 64

    @staticmethod
    def _emit(
        listener: ProgressListener | None,
        kind: ProgressKind,
        stage_name: str,
        fraction: float,
    ) -> None:
        if listener is not None:
            listener.on_progress(ProgressUpdate(kind, stage_name, fraction))
