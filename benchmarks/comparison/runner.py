"""Runs one dataset fixture under every difficulty preset
(docs/TECHNICAL_QUALITY_COMPARISON.md §3), reusing the same non-reimplemented
pipeline path every other benchmark package uses
(``benchmarks/framework/pipeline.run_pipeline``, BENCHMARK_SPEC.md §6).

Presets come from the production config layer (``app/config_defaults``),
not a comparison-specific copy -- ``n_colors``/``d_min_mm`` per preset are
read from the same dicts ``ConcreteOrchestrator.convert()`` itself resolves
against, so a preset comparison reports on the real preset definitions, not
an approximation of them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from benchmarks.datasets.loaders import DatasetFixture, load_all, load_category, load_examples
from benchmarks.framework.fixtures import Fixture as _BenchFixture
from benchmarks.framework.pipeline import PAGE_MM, PipelineRun, run_pipeline
from mysterycbn.app.config_defaults import D_MIN_MM_BY_PRESET, N_COLORS_BY_PRESET
from mysterycbn.model.records import Palette, PaletteColor, Provenance

PRESETS: tuple[str, ...] = ("easy", "medium", "hard")

_PROV = Provenance("comparison", "1.0.0", "0" * 64, "1" * 64)
_L_BANDS = (35.0, 55.0, 75.0)


def _palette_for_preset(n_colors: int) -> Palette:
    """A well-separated palette across the full preset range (up to 24
    colors for ``hard``) -- ``benchmarks/framework/pipeline._palette_for``'s
    single-radius hue wheel only clears the QM-16 warn floor up to ~10
    entries (documented limitation, see ``fixtures.py``); this varies L*
    across three bands as well as hue, which clears the merge_delta_e=7.0
    FATAL floor through all three preset sizes (verified: min ΔE00 stays
    >= 8.89 at n=16, the worst case across easy/medium/hard)."""
    n_colors = max(n_colors, 2)
    colors = []
    for i in range(n_colors):
        band = _L_BANDS[i % len(_L_BANDS)]
        angle = 2.0 * np.pi * i / n_colors
        colors.append(
            PaletteColor.from_lab(i, (band, 45.0 * np.cos(angle), 45.0 * np.sin(angle)), 1000)
        )
    return Palette(colors=tuple(colors), provenance=_PROV)


@dataclass(frozen=True)
class PresetRun:
    """One (fixture, preset) pipeline run plus the preset params used."""

    fixture: DatasetFixture
    preset: str
    n_colors: int
    d_min_mm: float
    run: PipelineRun


def _as_bench_fixture(fx: DatasetFixture, *, n_colors: int) -> _BenchFixture:
    """Adapts a dataset fixture to the benchmark harness's ``Fixture``
    shape, overriding palette size to the preset's ``n_colors`` -- the
    dataset's own declared ``palette_count`` is a *dataset* property (how
    many colors the synthetic generator used), while this is a *pipeline*
    property (how many the quantize/merge stages are configured to target),
    and the two are deliberately allowed to differ across presets.

    ``n_colors`` is floored at the label map's own max label + 1: a preset
    with fewer colors than the fixture's label map actually uses would
    violate ``LabelMap.validate_against`` (every label must index the
    palette) before the comparison even gets to measure anything -- this
    keeps every preset runnable on every fixture rather than silently
    skipping the ones a smaller preset can't represent."""
    min_colors = int(fx.labels.max()) + 1
    return _BenchFixture(
        fixture_id=fx.fixture_id,
        category=fx.category,
        labels=fx.labels,
        n_colors=max(n_colors, min_colors),
    )


def run_fixture_under_preset(
    fx: DatasetFixture, preset: str, *, page_mm: tuple[float, float, float] = PAGE_MM
) -> PresetRun:
    """Run one fixture through the real pipeline configured as ``preset``
    would configure it in production (``d_min_mm``, ``n_colors``)."""
    if preset not in D_MIN_MM_BY_PRESET:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(D_MIN_MM_BY_PRESET)}")
    d_min_mm = D_MIN_MM_BY_PRESET[preset]
    bench_fixture = _as_bench_fixture(fx, n_colors=N_COLORS_BY_PRESET[preset])
    n_colors = bench_fixture.n_colors
    run = run_pipeline(
        bench_fixture, page_mm=page_mm, d_min_mm=d_min_mm, palette_factory=_palette_for_preset
    )
    return PresetRun(fixture=fx, preset=preset, n_colors=n_colors, d_min_mm=d_min_mm, run=run)


def run_fixture_across_presets(
    fx: DatasetFixture, *, presets: tuple[str, ...] = PRESETS
) -> tuple[PresetRun, ...]:
    """One fixture, run under every preset in ``presets``, in order."""
    return tuple(run_fixture_under_preset(fx, preset) for preset in presets)


def run_examples_across_presets(
    *, presets: tuple[str, ...] = PRESETS
) -> Iterable[tuple[DatasetFixture, tuple[PresetRun, ...]]]:
    """The small one-per-category example ladder, each run under every preset."""
    for fx in load_examples():
        yield fx, run_fixture_across_presets(fx, presets=presets)


def run_category_across_presets(
    category: str, *, presets: tuple[str, ...] = PRESETS
) -> Iterable[tuple[DatasetFixture, tuple[PresetRun, ...]]]:
    """Every tier/difficulty fixture for one category, each run under every preset."""
    for fx in load_category(category):
        yield fx, run_fixture_across_presets(fx, presets=presets)


def run_full_dataset_across_presets(
    *, presets: tuple[str, ...] = PRESETS
) -> Iterable[tuple[DatasetFixture, tuple[PresetRun, ...]]]:
    """Every registered dataset fixture, each run under every preset."""
    for fx in load_all():
        yield fx, run_fixture_across_presets(fx, presets=presets)
