"""The Stage protocol — public plugin interface #4 (ARCHITECTURE.md §6, §8).

Stages must be deterministic given (artifacts, config, seed), side-effect-free
outside the context, and single-purpose. No stage imports another stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mysterycbn.model.context import PipelineContext


@runtime_checkable
class Stage(Protocol):
    """One pipeline step, discovered via the registry and selected by config."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def requires(self) -> Sequence[str]:
        """Artifact names that must be bound before this stage runs."""
        ...

    @property
    def provides(self) -> Sequence[str]:
        """Artifact names this stage binds on success."""
        ...

    @property
    def config_section(self) -> str:
        """The single config section this stage may read."""
        ...

    def run(self, ctx: PipelineContext) -> None:
        """Consume required artifacts from ``ctx`` and bind provided ones."""
        ...
