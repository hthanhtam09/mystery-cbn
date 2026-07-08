"""Unit tests for logging, tracing, plugin loader, and the exception hierarchy."""

from __future__ import annotations

import logging

import pytest

from mysterycbn.foundation.errors import (
    CancelledError,
    ConfigError,
    EngineError,
    InputError,
    QualityError,
    StageError,
)
from mysterycbn.foundation.logging import CorrelatedLoggerFactory
from mysterycbn.foundation.plugins import (
    PLUGIN_API_VERSION,
    DefaultPluginLoader,
    PluginRecord,
)
from mysterycbn.foundation.tracing import InMemoryTracer


def test_exception_hierarchy_roots_at_engine_error() -> None:
    for exc_type in (InputError, ConfigError, StageError, QualityError, CancelledError):
        assert issubclass(exc_type, EngineError)
    err = StageError("boom", stage_name="quantize", config_hash="ab" * 32)
    assert err.stage_name == "quantize"
    assert err.config_hash == "ab" * 32


def test_logger_factory_namespaces_and_correlation_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    factory = CorrelatedLoggerFactory()
    log = factory.get_logger("stages.quantize", correlation_id="run-42")
    assert log.logger.name == "mysterycbn.stages.quantize"
    with caplog.at_level(logging.INFO, logger="mysterycbn.stages.quantize"):
        log.info("palette ready")
    assert caplog.records[0].correlation_id == "run-42"  # type: ignore[attr-defined]


def test_tracer_spans_metrics_and_snapshot_immutability() -> None:
    tracer = InMemoryTracer()
    with tracer.span("quantize"):
        tracer.record_metric("quantize", "k_final", 14.0)
    tracer.record_artifact_size("label_map", 1024)
    snap = tracer.snapshot()
    assert snap["timings_s"]["quantize"] >= 0.0
    assert snap["metrics"]["quantize"]["k_final"] == 14.0
    assert snap["artifact_sizes"]["label_map"] == 1024
    with pytest.raises(TypeError):
        snap["timings_s"]["quantize"] = 0.0  # type: ignore[index]


def _record(name: str = "octree", api: str = PLUGIN_API_VERSION) -> PluginRecord:
    return PluginRecord(
        name=name,
        api_version=api,
        extension_point="stage:quantize",
        factory_callable=lambda: object(),
    )


def test_plugin_register_and_resolve() -> None:
    loader = DefaultPluginLoader()
    record = _record()
    loader.register(record)
    assert loader.resolve("stage:quantize", "octree") is record
    assert record.factory() is not None


def test_plugin_incompatible_api_version_refused() -> None:
    loader = DefaultPluginLoader()
    with pytest.raises(ConfigError, match="incompatible"):
        loader.register(_record(api="2.0"))


def test_plugin_duplicate_and_unknown_are_config_errors() -> None:
    loader = DefaultPluginLoader()
    loader.register(_record())
    with pytest.raises(ConfigError, match="already registered"):
        loader.register(_record())
    with pytest.raises(ConfigError, match="available"):
        loader.resolve("stage:quantize", "neural")
