"""Benchmark Framework: runs the engine on synthetic fixtures, measures
performance + quality, compares against baselines and goldens, and emits
JSON/CSV/HTML reports (BENCHMARK_SPEC.md).

This package is deliberately outside ``src/mysterycbn`` — it is tooling that
drives the engine from the outside, not a layer in the engine's own
dependency graph (ARCHITECTURE.md §2-3).
"""

from __future__ import annotations
