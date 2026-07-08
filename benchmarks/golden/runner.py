"""Runs the real engine pipeline over every fixture in the Sprint 20
categorized dataset (docs/GOLDEN_TEST_STANDARDS.md §3).

Adapts ``benchmarks.datasets.DatasetFixture`` to ``benchmarks.framework
.fixtures.Fixture`` so ``benchmarks/framework/pipeline.run_pipeline`` -- the
harness's one, already-reviewed, non-reimplemented path through the engine
(BENCHMARK_SPEC.md §6) -- is reused verbatim rather than duplicated.
"""

from __future__ import annotations

from collections.abc import Iterable

from benchmarks.datasets.loaders import DatasetFixture, load_all, load_golden
from benchmarks.framework.fixtures import Fixture as _BenchFixture
from benchmarks.framework.pipeline import PipelineRun, run_pipeline


def _as_bench_fixture(fx: DatasetFixture) -> _BenchFixture:
    return _BenchFixture(
        fixture_id=fx.fixture_id,
        category=fx.category,
        labels=fx.labels,
        n_colors=fx.metadata.palette_count,
    )


def run_dataset_fixture(fx: DatasetFixture) -> PipelineRun:
    """convert() -> SVG (+ PDF preview) for one dataset fixture."""
    return run_pipeline(_as_bench_fixture(fx))


def run_golden_ladder() -> Iterable[tuple[DatasetFixture, PipelineRun]]:
    """The frozen one-per-category golden subset, run through the pipeline."""
    for fx in load_golden():
        yield fx, run_dataset_fixture(fx)


def run_full_ladder() -> Iterable[tuple[DatasetFixture, PipelineRun]]:
    """Every dataset fixture (all categories x tiers x difficulties), run
    through the pipeline."""
    for fx in load_all():
        yield fx, run_dataset_fixture(fx)
