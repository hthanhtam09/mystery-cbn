"""Unit tests for topology fingerprint comparison."""

from __future__ import annotations

from benchmarks.golden.tolerances import GoldenTolerances
from benchmarks.golden.topology_compare import (
    TopologyFingerprint,
    _within_tolerance,
)


def test_within_tolerance_exact_match() -> None:
    assert _within_tolerance(10, 10, 0.0)


def test_within_tolerance_zero_golden_requires_zero_candidate() -> None:
    assert _within_tolerance(0, 0, 0.0)
    assert not _within_tolerance(0, 1, 0.5)


def test_within_tolerance_relative_band() -> None:
    assert _within_tolerance(100, 105, 0.10)
    assert not _within_tolerance(100, 120, 0.10)


def test_fingerprint_dict_round_trip() -> None:
    fp = TopologyFingerprint(region_count=4, arc_count=10, face_count=4)
    restored = TopologyFingerprint.from_dict(fp.to_dict())
    assert restored == fp


def test_zero_tolerance_rejects_any_delta() -> None:
    tolerances = GoldenTolerances(
        topology_region_count_tolerance=0.0, topology_arc_count_tolerance=0.0
    )
    assert tolerances.topology_region_count_tolerance == 0.0
