"""Unit tests for the five-layer config resolver and migration chain."""

from __future__ import annotations

import pytest

from mysterycbn.foundation.config.resolver import (
    CURRENT_SCHEMA_VERSION,
    FrozenConfig,
    LayeredResolver,
    MigrationChain,
)
from mysterycbn.foundation.config.schema import ConfigLayer
from mysterycbn.foundation.errors import ConfigError


def test_later_layers_win() -> None:
    resolver = LayeredResolver()
    cfg = resolver.resolve(
        {
            ConfigLayer.BUILTIN_DEFAULTS: {"quantize": {"n_colors": 16, "max_iter": 50}},
            ConfigLayer.DIFFICULTY_PRESET: {"quantize": {"n_colors": 8}},
            ConfigLayer.USER_FILE: {"quantize": {"n_colors": 20}},
        }
    )
    section = cfg.stage_section("quantize")
    assert section["n_colors"] == 20
    assert section["max_iter"] == 50  # untouched defaults survive the merge


def test_auto_tune_fills_only_unset_values() -> None:
    resolver = LayeredResolver()
    cfg = resolver.resolve(
        {
            ConfigLayer.USER_FILE: {"quantize": {"n_colors": 20}},
            ConfigLayer.AUTO_TUNE: {"quantize": {"n_colors": 12, "sample_px": 50_000}},
        }
    )
    section = cfg.stage_section("quantize")
    assert section["n_colors"] == 20  # explicit human intent wins
    assert section["sample_px"] == 50_000  # unset key filled by the proposal


def test_config_hash_is_content_stable() -> None:
    tree = {"a": {"x": 1}, "b": 2}
    assert FrozenConfig(tree, 1).config_hash == FrozenConfig(dict(tree), 1).config_hash
    assert FrozenConfig(tree, 1).config_hash != FrozenConfig({"a": {"x": 2}, "b": 2}, 1).config_hash
    assert FrozenConfig(tree, 1).config_hash != FrozenConfig(tree, 2).config_hash


def test_resolved_tree_is_read_only() -> None:
    cfg = LayeredResolver().resolve({ConfigLayer.USER_FILE: {"page": {"dpi": 300}}})
    with pytest.raises(TypeError):
        cfg.stage_section("page")["dpi"] = 600  # type: ignore[index]


def test_missing_section_raises_config_error() -> None:
    cfg = LayeredResolver().resolve({})
    with pytest.raises(ConfigError):
        cfg.stage_section("quantize")


def test_migration_chain_applies_steps_in_order() -> None:
    chain = MigrationChain()
    applied: list[int] = []

    def make_step(version: int):
        def step(doc: dict) -> dict:  # type: ignore[type-arg]
            applied.append(version)
            return {**doc, f"migrated_{version}": True}

        return step

    for v in range(CURRENT_SCHEMA_VERSION):
        chain.register(v, make_step(v))
    migrated = chain.migrate({"schema_version": 0, "keep": 1})
    assert applied == list(range(CURRENT_SCHEMA_VERSION))
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert migrated["keep"] == 1


def test_migration_rejects_future_and_unbridged_versions() -> None:
    chain = MigrationChain()
    with pytest.raises(ConfigError):
        chain.migrate({"schema_version": CURRENT_SCHEMA_VERSION + 1})
    with pytest.raises(ConfigError):
        chain.migrate({"schema_version": CURRENT_SCHEMA_VERSION - 1})  # no step registered
    with pytest.raises(ConfigError):
        chain.migrate({})  # missing version field
