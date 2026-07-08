"""Unit tests for the pipeline kernel: context, plan resolution, execution,
progress, cancellation, and caching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from mysterycbn.foundation.config.resolver import LayeredResolver
from mysterycbn.foundation.config.schema import ConfigLayer
from mysterycbn.foundation.errors import CancelledError, ConfigError, StageError
from mysterycbn.foundation.tracing import InMemoryTracer
from mysterycbn.kernel.cache import InMemoryArtifactCache
from mysterycbn.kernel.cancellation import ManualCancelToken
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.kernel.pipeline import DefaultPlanResolver, SequentialExecutor
from mysterycbn.kernel.progress import ProgressEvent, ProgressKind
from mysterycbn.kernel.registry import InMemoryStageRegistry
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Provenance

PROV = Provenance("fake", "1.0.0", "ab" * 32, "cd" * 32)


@dataclass(frozen=True)
class _Payload:
    """Minimal artifact: a value plus provenance."""

    value: int
    provenance: Provenance = PROV


@dataclass
class _FakeStage:
    """Configurable structural Stage: adds 1 to its input artifact."""

    name: str
    requires_: tuple[str, ...]
    provides_: tuple[str, ...]
    version: str = "1.0.0"
    runs: int = 0
    fail: bool = False
    forget_provides: bool = False

    @property
    def requires(self) -> Sequence[str]:
        return self.requires_

    @property
    def provides(self) -> Sequence[str]:
        return self.provides_

    @property
    def config_section(self) -> str:
        return self.name

    def run(self, ctx: PipelineContext) -> None:
        self.runs += 1
        if self.fail:
            raise RuntimeError("synthetic failure")
        if self.forget_provides:
            return
        base = 0
        for name in self.requires_:
            payload = ctx.get(name)
            assert isinstance(payload, _Payload)
            base = payload.value
        for name in self.provides_:
            ctx.put(name, _Payload(base + 1))


@dataclass
class _Recorder:
    events: list[ProgressEvent] = field(default_factory=list)

    def on_progress(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _config(stages: list[str], extra: dict | None = None):  # type: ignore[type-arg]
    tree: dict = {"pipeline": {"stages": stages}}  # type: ignore[type-arg]
    if extra:
        tree.update(extra)
    return LayeredResolver().resolve({ConfigLayer.PROGRAMMATIC: tree})


def _registry(*stages: _FakeStage, impl: str = "default") -> InMemoryStageRegistry:
    registry = InMemoryStageRegistry()
    for stage in stages:
        registry.register(stage.name, impl, stage)
    return registry


def _ctx() -> InMemoryContext:
    ctx = InMemoryContext(seed=0)
    ctx.put("source", _Payload(0))
    return ctx


def _chain() -> tuple[_FakeStage, _FakeStage]:
    return (
        _FakeStage("alpha", ("source",), ("a",)),
        _FakeStage("beta", ("a",), ("b",)),
    )


# ------------------------------------------------------------------- context


def test_context_binding_and_error_attribution() -> None:
    ctx = InMemoryContext(seed=7)
    assert ctx.seed == 7
    assert not ctx.has("x")
    ctx.put("x", _Payload(1))
    assert ctx.names() == ("x",)
    ctx.enter_stage("quantize")
    with pytest.raises(StageError) as err:
        ctx.get("missing")
    assert err.value.stage_name == "quantize"
    with pytest.raises(ValueError):
        InMemoryContext(seed=-1)


# ------------------------------------------------------------------ resolver


def test_resolver_validates_artifact_chain() -> None:
    alpha, beta = _chain()
    resolver = DefaultPlanResolver(_registry(alpha, beta), initial_artifacts=("source",))
    plan = resolver.resolve(_config(["alpha", "beta"]))
    assert tuple(s.name for s in plan.stages) == ("alpha", "beta")

    with pytest.raises(ConfigError, match="requires artifacts"):
        resolver.resolve(_config(["beta", "alpha"]))  # beta needs alpha's output


def test_resolver_impl_selection_and_unknown_slot() -> None:
    octree = _FakeStage("quantize", ("source",), ("a",))
    resolver = DefaultPlanResolver(_registry(octree, impl="octree"), initial_artifacts=("source",))
    plan = resolver.resolve(_config(["quantize"], {"quantize": {"impl": "octree"}}))
    assert plan.stages[0] is octree
    with pytest.raises(ConfigError, match="no implementation"):
        resolver.resolve(_config(["quantize"]))  # 'default' impl not registered
    with pytest.raises(ConfigError, match="stages"):
        DefaultPlanResolver(_registry()).resolve(
            LayeredResolver().resolve({ConfigLayer.PROGRAMMATIC: {"pipeline": {}}})
        )


# ------------------------------------------------------------------ executor


def test_executor_runs_chain_and_emits_progress() -> None:
    alpha, beta = _chain()
    plan = DefaultPlanResolver(_registry(alpha, beta), initial_artifacts=("source",)).resolve(
        _config(["alpha", "beta"])
    )
    recorder = _Recorder()
    tracer = InMemoryTracer()
    ctx = SequentialExecutor(tracer=tracer).execute(
        plan, _ctx(), on_progress=recorder, cancel_token=None
    )
    payload = ctx.get("b")
    assert isinstance(payload, _Payload) and payload.value == 2
    kinds = [(e.kind, e.stage_name, e.fraction_complete) for e in recorder.events]
    assert kinds == [
        (ProgressKind.STAGE_STARTED, "alpha", 0.0),
        (ProgressKind.STAGE_FINISHED, "alpha", 0.5),
        (ProgressKind.STAGE_STARTED, "beta", 0.5),
        (ProgressKind.STAGE_FINISHED, "beta", 1.0),
    ]
    assert set(tracer.snapshot()["timings_s"]) == {"alpha", "beta"}


def test_executor_cancellation_between_stages() -> None:
    alpha, beta = _chain()
    token = ManualCancelToken()

    class _CancelAfterAlpha:
        def on_progress(self, event: ProgressEvent) -> None:
            if event.kind is ProgressKind.STAGE_FINISHED and event.stage_name == "alpha":
                token.cancel()

    plan = DefaultPlanResolver(_registry(alpha, beta), initial_artifacts=("source",)).resolve(
        _config(["alpha", "beta"])
    )
    with pytest.raises(CancelledError, match="beta"):
        SequentialExecutor().execute(
            plan, _ctx(), on_progress=_CancelAfterAlpha(), cancel_token=token
        )
    assert alpha.runs == 1
    assert beta.runs == 0


def test_executor_wraps_failures_and_checks_provides() -> None:
    failing = _FakeStage("alpha", ("source",), ("a",), fail=True)
    plan = DefaultPlanResolver(_registry(failing), initial_artifacts=("source",)).resolve(
        _config(["alpha"])
    )
    with pytest.raises(StageError) as err:
        SequentialExecutor().execute(plan, _ctx(), on_progress=None, cancel_token=None)
    assert err.value.stage_name == "alpha"
    assert isinstance(err.value.__cause__, RuntimeError)

    lazy = _FakeStage("alpha", ("source",), ("a",), forget_provides=True)
    plan = DefaultPlanResolver(_registry(lazy), initial_artifacts=("source",)).resolve(
        _config(["alpha"])
    )
    with pytest.raises(StageError, match="without binding"):
        SequentialExecutor().execute(plan, _ctx(), on_progress=None, cancel_token=None)


# ------------------------------------------------------------------- caching


def test_cache_skips_rerun_and_invalidates_on_config_change() -> None:
    cache = InMemoryArtifactCache()
    alpha, beta = _chain()
    registry = _registry(alpha, beta)
    resolver = DefaultPlanResolver(registry, initial_artifacts=("source",))
    executor = SequentialExecutor(cache=cache)

    plan = resolver.resolve(_config(["alpha", "beta"]))
    executor.execute(plan, _ctx(), on_progress=None, cancel_token=None)
    assert (alpha.runs, beta.runs) == (1, 1)
    assert len(cache) == 2

    # Same config: both stages restored from cache, artifacts identical.
    ctx2 = executor.execute(plan, _ctx(), on_progress=None, cancel_token=None)
    assert (alpha.runs, beta.runs) == (1, 1)
    payload = ctx2.get("b")
    assert isinstance(payload, _Payload) and payload.value == 2

    # Changing beta's section re-runs beta but NOT alpha (ARCHITECTURE §13.4).
    plan2 = resolver.resolve(_config(["alpha", "beta"], {"beta": {"knob": 3}}))
    executor.execute(plan2, _ctx(), on_progress=None, cancel_token=None)
    assert (alpha.runs, beta.runs) == (1, 2)

    # Different seed misses everything.
    ctx_seed = InMemoryContext(seed=1)
    ctx_seed.put("source", _Payload(0))
    executor.execute(plan, ctx_seed, on_progress=None, cancel_token=None)
    assert (alpha.runs, beta.runs) == (2, 3)


def test_manual_cancel_token_is_idempotent() -> None:
    token = ManualCancelToken()
    assert not token.is_cancelled()
    token.cancel()
    token.cancel()
    assert token.is_cancelled()
