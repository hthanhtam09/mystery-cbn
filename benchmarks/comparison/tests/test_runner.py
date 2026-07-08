"""Tests for the cross-preset pipeline runner."""

from __future__ import annotations

import pytest

from benchmarks.comparison.runner import (
    PRESETS,
    run_fixture_across_presets,
    run_fixture_under_preset,
)
from benchmarks.datasets.loaders import load_all, load_fixture


def test_every_preset_is_runnable_on_the_example_ladder() -> None:
    for fixture_id in (
        "D-animals-examples-01",
        "D-flowers-examples-01",
        "D-cartoons-examples-01",
    ):
        fx = load_fixture(fixture_id)
        for preset in PRESETS:
            pr = run_fixture_under_preset(fx, preset)
            assert pr.preset == preset
            assert len(pr.run.curve_set.faces) > 0


def test_unknown_preset_raises() -> None:
    fx = load_fixture("D-animals-examples-01")
    with pytest.raises(ValueError, match="unknown preset"):
        run_fixture_under_preset(fx, "impossible")


def test_run_fixture_across_presets_covers_every_preset_in_order() -> None:
    fx = load_fixture("D-animals-examples-01")
    results = run_fixture_across_presets(fx)
    assert [r.preset for r in results] == list(PRESETS)


def test_n_colors_never_drops_below_the_fixtures_own_label_count() -> None:
    """A preset with fewer colors than a fixture's label map actually uses
    must not violate LabelMap.validate_against -- regression guard for the
    'hard-tier fixture, easy preset' case that originally crashed."""
    fx = load_fixture("D-cartoons-datasets-hard")
    min_required = int(fx.labels.max()) + 1
    for preset in PRESETS:
        pr = run_fixture_under_preset(fx, preset)
        assert pr.n_colors >= min_required


@pytest.mark.parametrize("fixture_id", [fx.fixture_id for fx in load_all()])
def test_every_dataset_fixture_runs_under_every_preset(fixture_id: str) -> None:
    """Full-dataset smoke test: every registered fixture (all 8 categories x
    3 difficulties) must run under every preset without raising."""
    fx = load_fixture(fixture_id)
    for preset in PRESETS:
        pr = run_fixture_under_preset(fx, preset)
        assert len(pr.run.curve_set.faces) > 0
