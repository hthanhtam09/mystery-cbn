"""Unit tests for the net-new Sprint 24 metrics: average edge length,
region size distribution, label density."""

from __future__ import annotations

import pytest

from benchmarks.comparison.metrics import (
    average_edge_length_mm,
    label_density_per_cm2,
    region_size_distribution,
)
from benchmarks.comparison.runner import run_fixture_under_preset
from benchmarks.datasets.loaders import load_fixture
from benchmarks.framework.pipeline import PAGE_MM


def _curve_set_and_labels(fixture_id: str, preset: str = "medium"):
    pr = run_fixture_under_preset(load_fixture(fixture_id), preset)
    ctx = pr.run.ctx
    return ctx.get("curve_set"), ctx.get("label_plan")


def test_average_edge_length_is_positive_for_a_real_fixture() -> None:
    curve_set, _ = _curve_set_and_labels("D-animals-examples-01")
    assert average_edge_length_mm(curve_set) > 0.0


def test_average_edge_length_is_deterministic() -> None:
    curve_set, _ = _curve_set_and_labels("D-animals-examples-01")
    a = average_edge_length_mm(curve_set)
    b = average_edge_length_mm(curve_set)
    assert a == b


def test_region_size_distribution_count_matches_face_count() -> None:
    curve_set, _ = _curve_set_and_labels("D-animals-examples-01")
    dist = region_size_distribution(curve_set)
    assert dist.count == len(curve_set.faces)


def test_region_size_distribution_ordering_invariants() -> None:
    curve_set, _ = _curve_set_and_labels("D-cartoons-datasets-hard")
    dist = region_size_distribution(curve_set)
    assert dist.min_mm2 <= dist.p10_mm2 <= dist.median_mm2 <= dist.p90_mm2 <= dist.max_mm2
    assert dist.mean_mm2 > 0.0
    assert dist.stdev_mm2 >= 0.0


def test_region_size_distribution_empty_case() -> None:
    from dataclasses import replace

    curve_set, _ = _curve_set_and_labels("D-animals-examples-01")
    empty = replace(curve_set, faces=())
    dist = region_size_distribution(empty)
    assert dist.count == 0
    assert dist.mean_mm2 == 0.0


def test_label_density_is_positive() -> None:
    _, labels = _curve_set_and_labels("D-animals-examples-01")
    density = label_density_per_cm2(labels, page_mm=PAGE_MM)
    assert density > 0.0


def test_label_density_matches_label_count_over_printable_area() -> None:
    from mysterycbn.stages.vector.arcgraph import content_box_pt

    _, labels = _curve_set_and_labels("D-animals-examples-01")
    density = label_density_per_cm2(labels, page_mm=PAGE_MM)
    _, _, width_pt, height_pt = content_box_pt(PAGE_MM)
    pt_to_mm = 25.4 / 72.0
    area_cm2 = (width_pt * pt_to_mm) * (height_pt * pt_to_mm) / 100.0
    expected = len(labels.labels) / area_cm2
    assert density == pytest.approx(expected)


def test_label_density_uses_printable_area_not_full_page() -> None:
    _, labels = _curve_set_and_labels("D-animals-examples-01")
    full_page = (215.9, 279.4, 0.0)
    with_margin = PAGE_MM
    density_full = label_density_per_cm2(labels, page_mm=full_page)
    density_margin = label_density_per_cm2(labels, page_mm=with_margin)
    # Smaller printable area (larger margin) -> higher density for the same label count.
    assert density_margin > density_full
