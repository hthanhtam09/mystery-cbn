"""Unit tests for golden-comparison tolerance configuration."""

from __future__ import annotations

from benchmarks.golden.tolerances import DEFAULT_TOLERANCES, GoldenTolerances


def test_default_tolerances_round_trip_to_dict() -> None:
    d = DEFAULT_TOLERANCES.to_dict()
    assert d["ssim_min"] == 0.97
    assert d["topology_region_count_tolerance"] == 0.0


def test_custom_tolerances_are_independent() -> None:
    custom = GoldenTolerances(ssim_min=0.9, topology_region_count_tolerance=0.05)
    assert custom.ssim_min == 0.9
    assert custom.topology_region_count_tolerance == 0.05
    assert DEFAULT_TOLERANCES.ssim_min == 0.97
