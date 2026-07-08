"""Concrete pipeline context: the typed artifact store stages communicate through.

Artifacts are immutable; the context's only mutability is name binding, and it
is confined to the executing thread (sequential pipeline, ARCHITECTURE.md §13.5).
"""

from __future__ import annotations

from mysterycbn.foundation.errors import StageError
from mysterycbn.model.artifacts import Artifact
from mysterycbn.model.context import PipelineContext

_UNSET_HASH = "0" * 64


class InMemoryContext(PipelineContext):
    """Default context. ``enter_stage`` is called by the executor so that
    missing-artifact errors are attributed to the stage that asked."""

    def __init__(self, *, seed: int, config_hash: str = _UNSET_HASH) -> None:
        if seed < 0:
            raise ValueError(f"seed must be ≥ 0, got {seed}")
        self._seed = seed
        self._config_hash = config_hash
        self._artifacts: dict[str, Artifact] = {}
        self._current_stage = "<pre-pipeline>"

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def config_hash(self) -> str:
        """Resolved-config hash used for error context and cache keys."""
        return self._config_hash

    def enter_stage(self, stage_name: str) -> None:
        """Record the currently executing stage (error attribution)."""
        self._current_stage = stage_name

    def get(self, artifact_name: str) -> Artifact:
        try:
            return self._artifacts[artifact_name]
        except KeyError:
            raise StageError(
                f"artifact {artifact_name!r} is not bound (bound: {sorted(self._artifacts)})",
                stage_name=self._current_stage,
                config_hash=self._config_hash,
            ) from None

    def put(self, artifact_name: str, artifact: Artifact) -> None:
        if not artifact_name:
            raise ValueError("artifact_name must be non-empty")
        self._artifacts[artifact_name] = artifact

    def has(self, artifact_name: str) -> bool:
        return artifact_name in self._artifacts

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._artifacts))
