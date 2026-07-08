"""Unit tests for the in-memory stage registry."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.kernel.registry import InMemoryStageRegistry
from mysterycbn.model.context import PipelineContext


class _FakeStage:
    """Minimal structural Stage for registry tests (no pipeline behavior)."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def requires(self) -> Sequence[str]:
        return ()

    @property
    def provides(self) -> Sequence[str]:
        return ()

    @property
    def config_section(self) -> str:
        return self._name

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("registry tests never execute stages")


def test_register_lookup_roundtrip() -> None:
    registry = InMemoryStageRegistry()
    stage = _FakeStage("quantize")
    registry.register("quantize", "labkmeans", stage)
    assert registry.lookup("quantize", "labkmeans") is stage


def test_implementations_sorted_per_slot() -> None:
    registry = InMemoryStageRegistry()
    registry.register("quantize", "octree", _FakeStage("quantize"))
    registry.register("quantize", "labkmeans", _FakeStage("quantize"))
    registry.register("denoise", "modal", _FakeStage("denoise"))
    assert registry.implementations("quantize") == ("labkmeans", "octree")
    assert registry.implementations("denoise") == ("modal",)
    assert registry.implementations("smooth") == ()


def test_duplicate_and_unknown_raise_config_error() -> None:
    registry = InMemoryStageRegistry()
    registry.register("quantize", "labkmeans", _FakeStage("quantize"))
    with pytest.raises(ConfigError, match="already registered"):
        registry.register("quantize", "labkmeans", _FakeStage("quantize"))
    with pytest.raises(ConfigError, match="available"):
        registry.lookup("quantize", "octree")
