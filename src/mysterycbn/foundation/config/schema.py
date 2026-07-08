"""Config contracts: frozen resolved config, layered resolution, forward migration."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


class ConfigLayer(enum.Enum):
    """Resolution layers, later wins; auto-tune may only fill values the user left unset."""

    BUILTIN_DEFAULTS = enum.auto()
    DIFFICULTY_PRESET = enum.auto()
    USER_FILE = enum.auto()
    PROGRAMMATIC = enum.auto()
    AUTO_TUNE = enum.auto()


@runtime_checkable
class ResolvedConfig(Protocol):
    """Typed, frozen, fully-resolved engine configuration."""

    @property
    def schema_version(self) -> int: ...

    @property
    def config_hash(self) -> str:
        """Stable hash of the fully resolved document (reproducibility record)."""
        ...

    def stage_section(self, stage_name: str) -> Mapping[str, Any]:
        """Return the one section this stage may read — and nothing else."""
        ...


class ConfigResolver(ABC):
    """Merges the five layers into a validated, frozen ResolvedConfig."""

    @abstractmethod
    def resolve(self, layers: Mapping[ConfigLayer, Mapping[str, Any]]) -> ResolvedConfig:
        """Merge, validate (including cross-field rules), and freeze. Raises ConfigError."""


class ConfigMigrator(ABC):
    """Applies ordered migrations so any historical config document loads today."""

    @abstractmethod
    def migrate(self, document: Mapping[str, Any]) -> Mapping[str, Any]:
        """Upgrade a document from its embedded schema_version to the current one."""
