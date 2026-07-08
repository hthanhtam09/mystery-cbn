"""Stage registry: the kernel discovers concrete stages here, never by import."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.stage import Stage


class StageRegistry(ABC):
    """Maps (stage name, implementation name) to Stage factories."""

    @abstractmethod
    def register(self, stage_name: str, impl_name: str, stage: Stage) -> None:
        """Register an implementation for a named pipeline slot."""

    @abstractmethod
    def lookup(self, stage_name: str, impl_name: str) -> Stage:
        """Return the implementation selected by config (e.g. ``stages.quantize.impl``)."""

    @abstractmethod
    def implementations(self, stage_name: str) -> Sequence[str]:
        """All registered implementation names for a pipeline slot."""


class InMemoryStageRegistry(StageRegistry):
    """Default registry: in-process dict keyed by (stage slot, implementation name).

    Built-in stages register here at import time; plugin stages are registered
    by the plugin loader. Selection is by configuration
    (``stages.<slot>.impl``), never by import (ARCHITECTURE.md §3, §8).
    """

    def __init__(self) -> None:
        self._stages: dict[tuple[str, str], Stage] = {}

    def register(self, stage_name: str, impl_name: str, stage: Stage) -> None:
        key = (stage_name, impl_name)
        if key in self._stages:
            raise ConfigError(
                f"implementation {impl_name!r} already registered for stage {stage_name!r}"
            )
        self._stages[key] = stage

    def lookup(self, stage_name: str, impl_name: str) -> Stage:
        try:
            return self._stages[(stage_name, impl_name)]
        except KeyError:
            raise ConfigError(
                f"no implementation {impl_name!r} for stage {stage_name!r}; "
                f"available: {self.implementations(stage_name)}"
            ) from None

    def implementations(self, stage_name: str) -> Sequence[str]:
        return tuple(sorted(impl for slot, impl in self._stages if slot == stage_name))
