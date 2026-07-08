"""Concrete five-layer configuration resolution (ARCHITECTURE.md §7, ENGINE_SPEC.md §3).

Implements the layering, freezing, hashing, and migration contracts of
:mod:`mysterycbn.foundation.config.schema` over plain nested mappings. The
engine's knob schema (per-stage keys) is added by the stages that own each
section; this module is deliberately schema-agnostic so the layering rules
never couple to individual knobs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from mysterycbn.foundation.config.schema import (
    ConfigLayer,
    ConfigMigrator,
    ConfigResolver,
    ResolvedConfig,
)
from mysterycbn.foundation.errors import ConfigError

CURRENT_SCHEMA_VERSION = 1

# Merge precedence, later wins; AUTO_TUNE is handled specially (fill-only).
_ORDER = (
    ConfigLayer.BUILTIN_DEFAULTS,
    ConfigLayer.DIFFICULTY_PRESET,
    ConfigLayer.USER_FILE,
    ConfigLayer.PROGRAMMATIC,
)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overlay`` merged in; nested mappings merge recursively."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _fill_missing(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` into ``base`` writing only keys absent from ``base``.

    Implements the auto-tune rule: proposals may never override explicit
    human intent (ARCHITECTURE.md §7).
    """
    out = dict(base)
    for key, value in overlay.items():
        if key not in out:
            out[key] = value if not isinstance(value, Mapping) else dict(value)
        elif isinstance(value, Mapping) and isinstance(out[key], Mapping):
            out[key] = _fill_missing(dict(out[key]), value)
    return out


def _freeze(tree: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively wrap mappings in read-only proxies."""
    return MappingProxyType(
        {k: _freeze(v) if isinstance(v, Mapping) else v for k, v in tree.items()}
    )


def _canonical_hash(tree: Mapping[str, Any], schema_version: int) -> str:
    """SHA-256 over the canonical JSON form (sorted keys, no whitespace)."""
    payload = json.dumps(
        {"schema_version": schema_version, "config": tree},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FrozenConfig(ResolvedConfig):
    """Immutable resolved configuration with a stable content hash."""

    def __init__(self, tree: Mapping[str, Any], schema_version: int) -> None:
        plain = json.loads(json.dumps(tree, default=str))
        self._tree = _freeze(plain)
        self._schema_version = schema_version
        self._hash = _canonical_hash(plain, schema_version)

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def config_hash(self) -> str:
        return self._hash

    def stage_section(self, stage_name: str) -> Mapping[str, Any]:
        section = self._tree.get(stage_name)
        if not isinstance(section, Mapping):
            raise ConfigError(f"no config section named {stage_name!r}")
        return section

    def as_mapping(self) -> Mapping[str, Any]:
        """The full frozen tree (reproducibility record embedding)."""
        return self._tree


class LayeredResolver(ConfigResolver):
    """Merges the five layers in precedence order and freezes the result."""

    def resolve(self, layers: Mapping[ConfigLayer, Mapping[str, Any]]) -> FrozenConfig:
        unknown = set(layers) - set(ConfigLayer)
        if unknown:
            raise ConfigError(f"unknown config layers: {sorted(u.name for u in unknown)}")
        merged: dict[str, Any] = {}
        for layer in _ORDER:
            merged = _deep_merge(merged, layers.get(layer, {}))
        merged = _fill_missing(merged, layers.get(ConfigLayer.AUTO_TUNE, {}))
        return FrozenConfig(merged, CURRENT_SCHEMA_VERSION)


class MigrationChain(ConfigMigrator):
    """Ordered ``schema_version → schema_version + 1`` migration registry.

    A document declares its ``schema_version``; :meth:`migrate` applies every
    registered step from that version up to :data:`CURRENT_SCHEMA_VERSION`
    (ARCHITECTURE.md §7: a 2028 config must load in 2033).
    """

    def __init__(self) -> None:
        self._steps: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def register(self, from_version: int, step: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Register the migration taking ``from_version`` to ``from_version + 1``."""
        if from_version in self._steps:
            raise ConfigError(f"duplicate migration registered for version {from_version}")
        self._steps[from_version] = step

    def migrate(self, document: Mapping[str, Any]) -> Mapping[str, Any]:
        doc = dict(document)
        version = doc.pop("schema_version", None)
        if not isinstance(version, int):
            raise ConfigError("config document missing integer 'schema_version'")
        if version > CURRENT_SCHEMA_VERSION:
            raise ConfigError(
                f"config schema_version {version} is newer than supported {CURRENT_SCHEMA_VERSION}"
            )
        while version < CURRENT_SCHEMA_VERSION:
            step = self._steps.get(version)
            if step is None:
                raise ConfigError(f"no migration registered from schema_version {version}")
            doc = step(doc)
            version += 1
        return {**doc, "schema_version": version}
