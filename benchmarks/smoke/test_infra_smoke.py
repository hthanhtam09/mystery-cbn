"""Smoke benchmarks for the run-critical infrastructure paths.

Algorithm-stage benchmarks (BENCHMARK_SPEC.md §5-§6) arrive with their stages;
until then this suite keeps the harness, CI wiring, and report plumbing
exercised on real code paths.
"""

from __future__ import annotations

from typing import Any

from mysterycbn.foundation.config.resolver import LayeredResolver
from mysterycbn.foundation.config.schema import ConfigLayer
from mysterycbn.foundation.tracing import InMemoryTracer

_LAYERS = {
    ConfigLayer.BUILTIN_DEFAULTS: {
        "page": {"width_mm": 215.9, "height_mm": 279.4, "margin_mm": 12.7, "dpi": 300},
        "quantize": {"n_colors": 16, "merge_delta_e": 7.0, "sample_px": 100_000},
        "quality": {"d_min_mm": 3.5, "font_min_pt": 6.0, "font_max_pt": 14.0},
    },
    ConfigLayer.DIFFICULTY_PRESET: {"quantize": {"n_colors": 8}, "quality": {"d_min_mm": 5.0}},
    ConfigLayer.USER_FILE: {"quantize": {"n_colors": 20}},
    ConfigLayer.AUTO_TUNE: {"preprocess": {"smooth_passes": 3}},
}


def test_config_resolution_smoke(benchmark: Any) -> None:
    resolver = LayeredResolver()
    cfg = benchmark(resolver.resolve, _LAYERS)
    assert cfg.stage_section("quantize")["n_colors"] == 20
    assert len(cfg.config_hash) == 64


def test_tracer_span_overhead_smoke(benchmark: Any) -> None:
    tracer = InMemoryTracer()

    def traced_noop() -> None:
        with tracer.span("noop"):
            pass

    benchmark(traced_noop)
    assert "noop" in tracer.snapshot()["timings_s"]
