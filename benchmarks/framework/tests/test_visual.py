"""Unit tests for golden/visual comparison: no-golden case, identical case,
and a genuine structural-change case (proves the diff actually fires)."""

from __future__ import annotations

from benchmarks.framework.fixtures import Fixture, load_fixture
from benchmarks.framework.pipeline import run_pipeline
from benchmarks.framework.visual import compare_to_golden, has_golden, write_golden
from mysterycbn.model.reports import GoldenOutcome


def test_no_golden_reports_incompatible_not_a_silent_pass(tmp_path, monkeypatch) -> None:
    import benchmarks.framework.visual as visual_mod

    monkeypatch.setattr(visual_mod, "GOLDENS_ROOT", tmp_path)
    fx = load_fixture("F-flat-2")
    run = run_pipeline(fx)
    assert not has_golden(fx.fixture_id)
    comparison = compare_to_golden(run)
    assert comparison.svg_outcome is GoldenOutcome.INCOMPATIBLE
    assert "no golden" in str(comparison.details.get("reason", ""))


def test_identical_rerun_matches_golden_exactly(tmp_path, monkeypatch) -> None:
    import benchmarks.framework.visual as visual_mod

    monkeypatch.setattr(visual_mod, "GOLDENS_ROOT", tmp_path)
    fx = load_fixture("F-flat-2")
    run = run_pipeline(fx)
    write_golden(run, engine_version="0.1.0", config_hash="0" * 64)

    run_again = run_pipeline(fx)
    comparison = compare_to_golden(run_again)
    assert comparison.svg_outcome is GoldenOutcome.IDENTICAL


def test_structurally_different_output_is_flagged_incompatible(tmp_path, monkeypatch) -> None:
    import benchmarks.framework.visual as visual_mod

    monkeypatch.setattr(visual_mod, "GOLDENS_ROOT", tmp_path)
    fx = load_fixture("F-flat-2")
    run = run_pipeline(fx)
    write_golden(run, engine_version="0.1.0", config_hash="0" * 64)

    other = load_fixture("F-illu-2")
    swapped = Fixture(
        fixture_id="F-flat-2", category=other.category, labels=other.labels, n_colors=other.n_colors
    )
    changed_run = run_pipeline(swapped)
    comparison = compare_to_golden(changed_run)
    assert comparison.svg_outcome is GoldenOutcome.INCOMPATIBLE
