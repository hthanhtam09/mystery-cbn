"""Pipeline context: the typed artifact store stages communicate through."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from mysterycbn.model.artifacts import Artifact


class PipelineContext(ABC):
    """Holds immutable artifacts by name; a stage replaces, never mutates in place."""

    @abstractmethod
    def get(self, artifact_name: str) -> Artifact:
        """Return the named artifact. Raises StageError if absent."""

    @abstractmethod
    def put(self, artifact_name: str, artifact: Artifact) -> None:
        """Bind an artifact, replacing any previous binding of the same name."""

    @abstractmethod
    def has(self, artifact_name: str) -> bool:
        """Whether the named artifact is currently bound."""

    @abstractmethod
    def names(self) -> Sequence[str]:
        """All currently bound artifact names."""

    @property
    @abstractmethod
    def seed(self) -> int:
        """The run's RNG seed (determinism invariant I2)."""
