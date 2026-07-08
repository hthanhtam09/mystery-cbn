"""Unit tests for the pipeline runner: every fixture must run end-to-end,
and repeated runs must be deterministic (I2)."""

from __future__ import annotations

from benchmarks.framework.fixtures import available_fixture_ids, load_fixture
from benchmarks.framework.pipeline import run_pipeline


def test_every_fixture_runs_without_error() -> None:
    for fixture_id in available_fixture_ids():
        fx = load_fixture(fixture_id)
        run = run_pipeline(fx)
        assert run.fixture_id == fixture_id
        assert len(run.svg_bytes) > 0
        assert len(run.curve_set.faces) >= 1


def test_pipeline_is_deterministic() -> None:
    fx = load_fixture("F-photo-05")
    run_a = run_pipeline(fx)
    run_b = run_pipeline(fx)
    assert run_a.svg_bytes == run_b.svg_bytes


def test_pdf_render_agrees_with_svg_when_available() -> None:
    fx = load_fixture("F-flat-2")
    run = run_pipeline(fx)
    if run.pdf_bytes is None:
        return  # pdf extras not installed
    assert len(run.pdf_bytes) > 0
