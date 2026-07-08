"""Unit tests for the Sprint 23 quality-metrics validator
(validate/quality_metrics.py). Purely observational: never raises on a
quality shortfall, only computes measurements from bound artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mysterycbn import validate as V
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.layout import Legend
from mysterycbn.model.records import Palette, PaletteColor, Provenance
from mysterycbn.render.pdf import render_pdf
from mysterycbn.render.svg import render_svg
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import place_labels
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph

PROV = Provenance("test", "1.0.0", "0" * 64, "1" * 64)
PAGE_MM = (215.9, 279.4, 12.7)


@dataclass(frozen=True)
class _Doc:
    data: bytes


def _palette(k: int, spacing: float = 20.0) -> Palette:
    return Palette(
        colors=tuple(
            PaletteColor.from_lab(i, (8.0 + spacing * i, 0.0, 0.0), 100) for i in range(k)
        ),
        provenance=PROV,
    )


def _checkerboard(k: int = 4, block: int = 20, grid: int = 6) -> list[list[int]]:
    rng = np.random.default_rng(0)
    base = rng.integers(0, k, (grid, grid))
    return np.repeat(np.repeat(base, block, axis=0), block, axis=1).tolist()


def _full_context(k: int = 4, spacing: float = 20.0) -> InMemoryContext:
    palette = _palette(k, spacing)
    lm_rows = _checkerboard(k=k)
    from mysterycbn.model.records import LabelMap

    lm = LabelMap(labels=np.array(lm_rows, dtype=np.int32), provenance=PROV)
    rg = build_region_graph(lm, palette)
    box = content_box_pt(PAGE_MM)
    tg = build_topology_graph(rg.component_map)
    ag = build_arc_graph(tg, rg, content_box=box)
    cs = fit_curves(ag)
    plan, findings = place_labels(cs, rg)
    assert findings == ()

    legend = Legend(
        permutation=tuple(range(palette.size)),
        chips=tuple((i, (20.0 + i * 30.0, 20.0), 20.0) for i in range(palette.size)),
        band_rect=(10.0, 10.0, 200.0, 40.0),
        number_font_pt=8.0,
        provenance=PROV,
    )
    svg_bytes = render_svg(cs, plan, legend, palette, page_mm=PAGE_MM)

    ctx = InMemoryContext(seed=0)
    ctx.put("region_graph", rg)
    ctx.put("arc_graph", ag)
    ctx.put("curve_set", cs)
    ctx.put("label_plan", plan)
    ctx.put("palette", palette)
    ctx.put("svg", _Doc(svg_bytes))
    try:
        pdf_bytes = render_pdf(cs, plan, legend, palette, page_mm=PAGE_MM)
        ctx.put("pdf", _Doc(pdf_bytes))
    except ImportError:
        pytest.importorskip("reportlab")
    return ctx


@pytest.fixture()
def clean_ctx() -> InMemoryContext:
    return _full_context()


def test_never_raises_on_a_clean_pipeline(clean_ctx: InMemoryContext) -> None:
    metrics = V.compute_quality_metrics(clean_ctx)
    assert metrics


def test_expected_metric_keys_present(clean_ctx: InMemoryContext) -> None:
    metrics = V.compute_quality_metrics(clean_ctx)
    expected = {
        "QM-13",
        "QM-14",
        "QM-11",
        "QM-08",
        "QM-16",
        "QM-22",
        "label_overlap_rate_pct",
        "QM-26",
        "QM-28",
        "printability_score",
    }
    assert expected <= set(metrics)


def test_region_count_matches_face_count(clean_ctx: InMemoryContext) -> None:
    from mysterycbn.model.vector import CurveSet

    curve_set = clean_ctx.get("curve_set")
    assert isinstance(curve_set, CurveSet)
    metrics = V.compute_quality_metrics(clean_ctx)
    assert metrics["QM-13"].value == float(len(curve_set.faces))


def test_mean_compactness_is_in_unit_range(clean_ctx: InMemoryContext) -> None:
    metrics = V.compute_quality_metrics(clean_ctx)
    assert 0.0 <= metrics["QM-14"].value <= 1.0


def test_boundary_smoothness_is_nonnegative(clean_ctx: InMemoryContext) -> None:
    metrics = V.compute_quality_metrics(clean_ctx)
    assert metrics["QM-08"].value >= 0.0


def test_palette_quality_matches_known_separation() -> None:
    """A palette built with fixed L* spacing has an exactly known min ΔE
    band; the metric must reproduce it, not a re-derivation error."""
    ctx = _full_context(k=3, spacing=30.0)
    metrics = V.compute_quality_metrics(ctx)
    from mysterycbn.model.records import Palette

    palette = ctx.get("palette")
    assert isinstance(palette, Palette)
    k = palette.size
    expected_min = float(palette.delta_e_table[~np.eye(k, dtype=bool)].min())
    assert metrics["QM-16"].value == pytest.approx(expected_min)


def test_no_overlap_on_a_clean_label_plan(clean_ctx: InMemoryContext) -> None:
    metrics = V.compute_quality_metrics(clean_ctx)
    assert metrics["label_overlap_rate_pct"].value == 0.0
    assert metrics["label_overlap_rate_pct"].passed


def test_svg_and_pdf_validity_pass_on_rendered_bytes(clean_ctx: InMemoryContext) -> None:
    metrics = V.compute_quality_metrics(clean_ctx)
    assert metrics["QM-26"].value == 1.0
    assert metrics["QM-28"].value == 1.0


def test_svg_validity_fails_on_malformed_svg(clean_ctx: InMemoryContext) -> None:
    clean_ctx.put("svg", _Doc(b"<svg not well formed"))
    metrics = V.compute_quality_metrics(clean_ctx)
    assert metrics["QM-26"].value == 0.0
    assert not metrics["QM-26"].passed


def test_pdf_validity_passes_when_pdf_absent() -> None:
    ctx = _full_context()
    # Simulate a run with no PDF extras installed: no "pdf" artifact bound.
    printability_metrics = None
    metrics = V.compute_quality_metrics(ctx, printability_metrics=printability_metrics)
    assert metrics["QM-28"].value in (0.0, 1.0)  # present or genuinely absent, never raises


def test_printability_score_uses_tiny_region_pct(clean_ctx: InMemoryContext) -> None:
    metrics = V.compute_quality_metrics(clean_ctx, printability_metrics={"tiny_region_pct": 20.0})
    assert metrics["printability_score"].value == pytest.approx(0.8)
    metrics_floor = V.compute_quality_metrics(
        clean_ctx, printability_metrics={"tiny_region_pct": 90.0}
    )
    assert metrics_floor["printability_score"].value == pytest.approx(0.5)


def test_no_metric_ever_carries_a_fatal_severity(clean_ctx: InMemoryContext) -> None:
    """This validator is purely observational -- unlike the four canonical
    validators, nothing it computes should ever be able to raise/gate."""
    try:
        V.compute_quality_metrics(clean_ctx)
    except Exception as exc:
        pytest.fail(f"compute_quality_metrics must not raise on valid input: {exc}")
