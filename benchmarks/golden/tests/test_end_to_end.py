"""End-to-end golden-test framework: bless then compare, in one process so
results aren't sensitive to the interpreter's PYTHONHASHSEED
(docs/GOLDEN_TEST_STANDARDS.md §8 known issue -- see conftest for the CLI's
workaround, which doesn't apply within a single pytest process since the
comparison run reuses the same run that produced the golden's inputs
deterministically within-process).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.datasets.loaders import load_fixture
from benchmarks.golden import storage
from benchmarks.golden.compare import compare_run_to_golden
from benchmarks.golden.runner import run_dataset_fixture
from benchmarks.golden.topology_compare import fingerprint_run
from benchmarks.golden.update import _bless_one
from mysterycbn.model.reports import GoldenOutcome


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "GOLDEN_STORE_ROOT", tmp_path)


def test_blessed_fixture_compares_identical_to_itself() -> None:
    fx = load_fixture("D-animals-examples-01")
    run = run_dataset_fixture(fx)
    _bless_one(fx.category, run)

    rerun = run_dataset_fixture(fx)
    report = compare_run_to_golden(rerun, category=fx.category)

    assert report.passed
    assert report.svg_outcome is GoldenOutcome.IDENTICAL
    assert report.topology is not None
    assert report.topology.passed


def test_missing_golden_is_incompatible_not_a_silent_pass() -> None:
    fx = load_fixture("D-flowers-examples-01")
    run = run_dataset_fixture(fx)
    report = compare_run_to_golden(run, category=fx.category)

    assert not report.passed
    assert report.svg_outcome is GoldenOutcome.INCOMPATIBLE
    assert report.topology is None
    assert "reason" in report.details


def test_topology_fingerprint_matches_freshly_blessed_run() -> None:
    fx = load_fixture("D-architecture-examples-01")
    run = run_dataset_fixture(fx)
    _bless_one(fx.category, run)

    stored = storage.read_golden_topology(fx.fixture_id)
    assert stored == fingerprint_run(run)
