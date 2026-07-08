"""Golden update ("bless") workflow (docs/GOLDEN_TEST_STANDARDS.md §6).

Regenerates goldens from the current engine output. Explicit and manual
only -- never invoked implicitly by a comparison run (BENCHMARK_SPEC.md
§4.3: goldens regenerate only via an explicit, reviewed call; this module
is the one place golden files are written).
"""

from __future__ import annotations

from collections.abc import Iterable

from benchmarks.datasets.loaders import DatasetFixture, load_fixture
from benchmarks.datasets.registry import DATASET_VERSION
from benchmarks.framework.pipeline import PipelineRun
from benchmarks.framework.visual import _encode_png, _rasterize_preview
from benchmarks.golden import storage
from benchmarks.golden.runner import run_dataset_fixture, run_full_ladder, run_golden_ladder
from benchmarks.golden.topology_compare import fingerprint_run
from mysterycbn import __version__ as ENGINE_VERSION


def _bless_one(category: str, run: PipelineRun) -> None:
    preview_png: bytes | None = None
    if run.pdf_bytes is not None:
        preview_array = _rasterize_preview(run.pdf_bytes)
        if preview_array is not None:
            preview_png = _encode_png(preview_array)

    storage.write_golden(
        run.fixture_id,
        svg_bytes=run.svg_bytes,
        preview_png_bytes=preview_png,
        topology=fingerprint_run(run),
        engine_version=ENGINE_VERSION,
        dataset_version=DATASET_VERSION,
        category=category,
    )


def _bless(pairs: Iterable[tuple[DatasetFixture, PipelineRun]]) -> list[str]:
    blessed = []
    for fx, run in pairs:
        _bless_one(fx.category, run)
        blessed.append(fx.fixture_id)
    return blessed


def bless_golden_ladder() -> list[str]:
    """Regenerate goldens for the frozen one-per-category subset."""
    return _bless(run_golden_ladder())


def bless_full_dataset() -> list[str]:
    """Regenerate goldens for every fixture in the categorized dataset."""
    return _bless(run_full_ladder())


def bless_fixture_ids(fixture_ids: Iterable[str]) -> list[str]:
    """Regenerate goldens for a specific set of fixture ids."""
    blessed = []
    for fixture_id in fixture_ids:
        fx = load_fixture(fixture_id)
        run = run_dataset_fixture(fx)
        _bless_one(fx.category, run)
        blessed.append(fixture_id)
    return blessed
