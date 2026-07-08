"""Pipeline framework.

A pipeline is an ordered list of :class:`Stage` objects. Each stage declares
the context fields it requires and provides, which gives us:

- fail-fast validation of stage ordering before any work runs,
- per-stage timing collected into ``ctx.trace``,
- uniform error wrapping (any exception becomes a ``StageError``).

Stages are objects (not bare functions) so alternative implementations of a
step can be swapped in by name — the modularity requirement of the design.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from .errors import ConfigError, EngineError, StageError
from .types import PipelineContext, StageTiming


class Stage(Protocol):
    """A pipeline step. Implementations mutate the context in place."""

    name: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]

    def run(self, ctx: PipelineContext) -> None: ...


class FunctionStage:
    """Adapter turning a plain function into a Stage."""

    def __init__(
        self,
        name: str,
        fn: Callable[[PipelineContext], None],
        requires: Sequence[str] = (),
        provides: Sequence[str] = (),
    ) -> None:
        self.name = name
        self._fn = fn
        self.requires = tuple(requires)
        self.provides = tuple(provides)

    def run(self, ctx: PipelineContext) -> None:
        self._fn(ctx)


class Pipeline:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self.stages = list(stages)
        self._validate_order()

    def _validate_order(self) -> None:
        """Every stage's requirements must be provided by an earlier stage."""
        available: set[str] = {"config", "trace"}
        for stage in self.stages:
            missing = set(stage.requires) - available
            if missing:
                raise ConfigError(
                    f"stage '{stage.name}' requires {sorted(missing)} but no "
                    f"earlier stage provides them"
                )
            available.update(stage.provides)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self.stages:
            ctx.require(*(f for f in stage.requires if f not in ("config", "trace")))
            start = time.perf_counter()
            try:
                stage.run(ctx)
            except EngineError:
                raise
            except Exception as exc:
                raise StageError(stage.name, str(exc)) from exc
            ctx.trace.timings.append(StageTiming(stage.name, time.perf_counter() - start))
            unset = [f for f in stage.provides if getattr(ctx, f, None) is None]
            if unset:
                raise StageError(stage.name, f"declared but did not provide: {', '.join(unset)}")
        return ctx
