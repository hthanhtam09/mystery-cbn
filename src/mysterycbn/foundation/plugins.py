"""Plugin discovery, version gating, and registry (ARCHITECTURE.md §8)."""

from __future__ import annotations

import importlib.metadata
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mysterycbn.foundation.errors import ConfigError


@runtime_checkable
class PluginDescriptor(Protocol):
    """Identity and compatibility declaration of one discovered plugin."""

    @property
    def name(self) -> str: ...

    @property
    def api_version(self) -> str: ...

    @property
    def extension_point(self) -> str: ...

    def factory(self) -> Any:
        """Return the callable that constructs the plugin's implementation."""
        ...


class PluginLoader(ABC):
    """Discovers plugins via entry points and explicit registration; gates by api_version."""

    @abstractmethod
    def discover(self) -> Sequence[PluginDescriptor]:
        """Scan the ``mysterycbn.plugins`` entry-point group and return compatible plugins."""

    @abstractmethod
    def register(self, descriptor: PluginDescriptor) -> None:
        """Explicitly register a plugin (embedded use). Raises on incompatible api_version."""

    @abstractmethod
    def resolve(self, extension_point: str, name: str) -> PluginDescriptor:
        """Return the registered plugin for ``extension_point`` selected by ``name``."""


PLUGIN_API_VERSION = "1.0"
ENTRY_POINT_GROUP = "mysterycbn.plugins"


@dataclass(frozen=True)
class PluginRecord:
    """Value-type plugin descriptor for explicit (embedded) registration."""

    name: str
    api_version: str
    extension_point: str
    factory_callable: Callable[[], Any]

    def factory(self) -> Any:
        return self.factory_callable()


def _compatible(api_version: str) -> bool:
    """A plugin is compatible iff its declared major version matches the engine's."""
    return api_version.split(".", 1)[0] == PLUGIN_API_VERSION.split(".", 1)[0]


class DefaultPluginLoader(PluginLoader):
    """Entry-point discovery plus explicit registration, with api_version gating.

    Incompatible plugins are refused at load time with a clear message rather
    than failing mid-pipeline (ARCHITECTURE.md §8).
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[str, str], PluginDescriptor] = {}

    def discover(self) -> Sequence[PluginDescriptor]:
        found: list[PluginDescriptor] = []
        for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            descriptor = entry.load()
            if not isinstance(descriptor, PluginDescriptor):
                raise ConfigError(
                    f"entry point {entry.name!r} in {ENTRY_POINT_GROUP!r} does not "
                    "provide a PluginDescriptor"
                )
            self.register(descriptor)
            found.append(descriptor)
        return tuple(found)

    def register(self, descriptor: PluginDescriptor) -> None:
        if not _compatible(descriptor.api_version):
            raise ConfigError(
                f"plugin {descriptor.name!r} declares api_version "
                f"{descriptor.api_version!r}, incompatible with engine plugin API "
                f"{PLUGIN_API_VERSION!r}"
            )
        key = (descriptor.extension_point, descriptor.name)
        if key in self._registry:
            raise ConfigError(
                f"plugin {descriptor.name!r} already registered for extension point "
                f"{descriptor.extension_point!r}"
            )
        self._registry[key] = descriptor

    def resolve(self, extension_point: str, name: str) -> PluginDescriptor:
        try:
            return self._registry[(extension_point, name)]
        except KeyError:
            available = sorted(n for ep, n in self._registry if ep == extension_point)
            raise ConfigError(
                f"no plugin named {name!r} for extension point {extension_point!r}; "
                f"available: {available}"
            ) from None
