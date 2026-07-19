"""Unit tests for the Validation Engine (ENGINE_SPEC.md §25; QUALITY_SPEC.md
QM-01/02/10/11/16/18/21/24/26/28). Each check is exercised on a
purpose-built broken artifact, plus the repair loop and OutputBundle wiring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mysterycbn import validate as V
from mysterycbn.foundation.errors import QualityError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.layout import Label, LabelMode, LabelPlan, Legend
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.model.reports import Severity
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
    """Minimal stand-in for SvgDocument/PdfDocument (only ``.data`` used)."""

    data: bytes


def _palette(k: int, spacing: float = 20.0) -> Palette:
    return Palette(
        colors=tuple(
            PaletteColor.from_lab(i, (8.0 + spacing * i, 0.0, 0.0), 100) for i in range(k)
        ),
        provenance=PROV,
    )


def _pipeline(
    rows: list[list[int]], palette: Palette, page_mm: tuple[float, float, float] = PAGE_MM
):
    lm = LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV)
    rg = build_region_graph(lm, palette)
    box = content_box_pt(page_mm)
    tg = build_topology_graph(rg.component_map)
    ag = build_arc_graph(tg, rg, content_box=box)
    cs = fit_curves(ag)
    plan, findings = place_labels(cs, rg)
    assert findings == ()
    return rg, ag, cs, plan


def _context(rg, ag, cs, plan, palette: Palette) -> InMemoryContext:
    ctx = InMemoryContext(seed=0)
    ctx.put("region_graph", rg)
    ctx.put("arc_graph", ag)
    ctx.put("curve_set", cs)
    ctx.put("label_plan", plan)
    ctx.put("palette", palette)
    return ctx


def _checkerboard(k: int = 4, block: int = 20, grid: int = 6) -> list[list[int]]:
    rng = np.random.default_rng(0)
    base = rng.integers(0, k, (grid, grid))
    return np.repeat(np.repeat(base, block, axis=0), block, axis=1).tolist()


# --------------------------------------------------------------- fixtures --


@pytest.fixture()
def clean_pipeline():
    palette = _palette(4)
    rg, ag, cs, plan = _pipeline(_checkerboard(), palette)
    return rg, ag, cs, plan, palette


# ------------------------------------------------------------- full gate ---


def test_full_gate_passes_on_a_clean_pipeline(clean_pipeline) -> None:
    rg, ag, cs, plan, palette = clean_pipeline
    ctx = _context(rg, ag, cs, plan, palette)
    reports = V.run_validation(ctx)
    assert [r.validator_name for r in reports] == [
        "fidelity",
        "topology",
        "printability",
        "palette",
    ]
    assert all(r.passed for r in reports)


def test_output_validity_passes_on_rendered_bytes(clean_pipeline) -> None:
    rg, ag, cs, plan, palette = clean_pipeline
    legend = Legend(
        permutation=tuple(range(palette.size)),
        chips=tuple((i, (20.0 + i * 30.0, 20.0), 20.0) for i in range(palette.size)),
        band_rect=(10.0, 10.0, 200.0, 40.0),
        number_font_pt=8.0,
        provenance=PROV,
    )
    svg_bytes = render_svg(cs, plan, legend, palette, page_mm=PAGE_MM)
    pdf_bytes = render_pdf(cs, plan, legend, palette, page_mm=PAGE_MM)
    ctx = _context(rg, ag, cs, plan, palette)
    ctx.put("svg", _Doc(svg_bytes))
    ctx.put("pdf", _Doc(pdf_bytes))
    report = V.run_output_validity(ctx)
    assert report.passed
    assert report.findings == ()


def test_output_validity_fatal_on_missing_svg(clean_pipeline) -> None:
    rg, ag, cs, plan, palette = clean_pipeline
    ctx = _context(rg, ag, cs, plan, palette)
    report = V.run_output_validity(ctx)
    assert not report.passed
    assert any(f.severity is Severity.FATAL for f in report.findings)


def test_output_validity_fatal_on_malformed_svg(clean_pipeline) -> None:
    rg, ag, cs, plan, palette = clean_pipeline
    ctx = _context(rg, ag, cs, plan, palette)
    ctx.put("svg", _Doc(b"<svg not well formed"))
    report = V.run_output_validity(ctx)
    assert not report.passed
    assert any("not well-formed" in f.message for f in report.findings)


# ----------------------------------------------------------------- topology


def test_topology_passes_and_finds_no_errors_on_real_geometry(clean_pipeline) -> None:
    rg, ag, cs, _plan, _palette = clean_pipeline
    ctx = InMemoryContext(seed=0)
    ctx.put("curve_set", cs)
    ctx.put("arc_graph", ag)
    ctx.put("region_graph", rg)
    report = V.validate_topology(ctx)
    assert report.passed
    assert report.metrics["topology_errors"] == 0.0


def test_topology_fatal_on_corrupted_face_geometry() -> None:
    """A face whose outer walk is shrunk (gap injected) must FATAL on
    watertightness -- this never triggers a repair (§25.2)."""
    from mysterycbn.model.records import Region, RegionGraph
    from mysterycbn.model.vector import ArcGraph, BezierSegment, Curve, CurveSet, Face

    def seg(p0: tuple[float, float], p1: tuple[float, float]) -> BezierSegment:
        a, b = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
        return BezierSegment(control=np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]))

    def curve(arc_id: int, p0: tuple[float, float], p1: tuple[float, float]) -> Curve:
        return Curve(
            arc_id=arc_id, segments=(seg(p0, p1),), corner_indices=(), max_fit_error_pt=0.0
        )

    # Correct winding for outer-positive (y-down, region-on-left): CCW in (x,y) math.
    good = (
        curve(0, (0, 0), (0, 10)),
        curve(1, (0, 10), (10, 10)),
        curve(2, (10, 10), (10, 0)),
        curve(3, (10, 0), (0, 0)),
    )
    face = Face(face_id=0, label=0, outer_walk=((0, False), (1, False), (2, False), (3, False)))
    good_cs = CurveSet(curves=good, faces=(face,), provenance=PROV)

    region = Region(0, 0, 100, (0, 0, 9, 9), (0, 0), 40, (5.0, 5.0))
    cmap = np.zeros((10, 10), dtype=np.int32)
    rg = RegionGraph(regions=(region,), component_map=cmap, edges=(), provenance=PROV)
    ag = ArcGraph(arcs=(), faces=(), work_scale=1.0, provenance=PROV)

    ctx_good = InMemoryContext(seed=0)
    ctx_good.put("curve_set", good_cs)
    ctx_good.put("arc_graph", ag)
    ctx_good.put("region_graph", rg)
    good_report = V.validate_topology(ctx_good)
    assert good_report.passed, good_report.metrics

    # Corrupt: cut a corner off arc 3, shrinking the face (gap injected).
    bad3 = curve(3, (5, 0), (0, 0))
    bad_cs = CurveSet(curves=(*good[:3], bad3), faces=(face,), provenance=PROV)
    ctx_bad = InMemoryContext(seed=0)
    ctx_bad.put("curve_set", bad_cs)
    ctx_bad.put("arc_graph", ag)
    ctx_bad.put("region_graph", rg)
    bad_report = V.validate_topology(ctx_bad)
    assert not bad_report.passed
    assert all(f.severity is Severity.FATAL for f in bad_report.findings)
    assert all(not f.repair_applied for f in bad_report.findings)


def test_topology_fatal_on_arc_not_bordering_two_faces() -> None:
    """An arc referenced by 0 walk-sides is neither an interior arc (2) nor
    a legal exterior-boundary arc (1) -- FATAL."""
    from mysterycbn.model.records import Region, RegionGraph
    from mysterycbn.model.vector import ArcGraph, BezierSegment, Curve, CurveSet, Face

    def seg(p0: tuple[float, float], p1: tuple[float, float]) -> BezierSegment:
        a, b = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
        return BezierSegment(control=np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]))

    def curve(arc_id: int, p0: tuple[float, float], p1: tuple[float, float]) -> Curve:
        return Curve(
            arc_id=arc_id, segments=(seg(p0, p1),), corner_indices=(), max_fit_error_pt=0.0
        )

    orphan = curve(4, (100, 100), (110, 110))  # never referenced by any face walk
    good = (
        curve(0, (0, 0), (0, 10)),
        curve(1, (0, 10), (10, 10)),
        curve(2, (10, 10), (10, 0)),
        curve(3, (10, 0), (0, 0)),
    )
    face = Face(face_id=0, label=0, outer_walk=((0, False), (1, False), (2, False), (3, False)))
    cs = CurveSet(curves=(*good, orphan), faces=(face,), provenance=PROV)

    region = Region(0, 0, 100, (0, 0, 9, 9), (0, 0), 40, (5.0, 5.0))
    cmap = np.zeros((10, 10), dtype=np.int32)
    rg = RegionGraph(regions=(region,), component_map=cmap, edges=(), provenance=PROV)
    ag = ArcGraph(arcs=(), faces=(), work_scale=1.0, provenance=PROV)

    ctx = InMemoryContext(seed=0)
    ctx.put("curve_set", cs)
    ctx.put("arc_graph", ag)
    ctx.put("region_graph", rg)
    report = V.validate_topology(ctx)
    assert not report.passed
    assert any("arc 4" in f.location for f in report.findings)


# ------------------------------------------------------------- printability


def test_printability_passes_on_a_well_formed_page(clean_pipeline) -> None:
    rg, ag, cs, plan, palette = clean_pipeline
    ctx = _context(rg, ag, cs, plan, palette)
    report = V.validate_printability(ctx)
    assert report.passed
    assert report.metrics["tiny_region_pct"] == 0.0
    assert report.metrics["label_coverage_pct"] == 100.0


def test_printability_repairs_a_sub_floor_region_to_a_leader() -> None:
    """A ~1mm sliver below d_min_mm gets demoted to a leader (REPAIRED, not
    FATAL); the repair mutates the bound label_plan in place."""
    palette = _palette(2, spacing=60.0)
    rows = [[0] * 60 for _ in range(60)]
    for c in range(10, 50):
        rows[29][c] = 1
    rg, ag, cs, plan = _pipeline(rows, palette)
    ctx = _context(rg, ag, cs, plan, palette)

    report = V.validate_printability(ctx)
    assert report.passed
    assert any(f.severity is Severity.REPAIRED and f.repair_applied for f in report.findings)
    assert report.metrics["tiny_region_pct"] > 0.0

    repaired_plan = ctx.get("label_plan")
    sliver = next(lb for lb in repaired_plan.labels if lb.region_id == 1)
    assert sliver.mode is LabelMode.LEADER


def test_printability_fatal_when_face_has_no_label_entry(clean_pipeline) -> None:
    rg, ag, cs, plan, palette = clean_pipeline
    truncated = LabelPlan(labels=plan.labels[1:], provenance=plan.provenance)
    ctx = _context(rg, ag, cs, truncated, palette)
    report = V.validate_printability(ctx)
    assert not report.passed
    assert any("no label plan entry" in f.message for f in report.findings)
    assert report.metrics["label_coverage_pct"] < 100.0


def test_printability_fatal_when_font_below_floor(clean_pipeline) -> None:
    rg, ag, cs, plan, palette = clean_pipeline
    tampered = LabelPlan(
        labels=tuple(
            Label(
                region_id=lb.region_id,
                printed_number=lb.printed_number,
                anchor=lb.anchor,
                font_size_pt=1.0,
                mode=lb.mode,
                clearance_pt=lb.clearance_pt,
                leader=lb.leader,
            )
            for lb in plan.labels
        ),
        provenance=plan.provenance,
    )
    ctx = _context(rg, ag, cs, tampered, palette)
    report = V.validate_printability(ctx, font_min_pt=6.0)
    assert not report.passed
    assert all(f.severity is Severity.FATAL for f in report.findings)


# ------------------------------------------------------------------ palette


def test_palette_passes_when_well_separated() -> None:
    palette = _palette(4, spacing=25.0)
    ctx = InMemoryContext(seed=0)
    ctx.put("palette", palette)
    report = V.validate_palette(ctx, merge_delta_e=7.0, palette_warn_delta_e=12.0)
    assert report.passed
    assert report.findings == ()


def test_palette_fatal_on_construction_floor_violation() -> None:
    palette = Palette(
        colors=(
            PaletteColor.from_lab(0, (50.0, 0.0, 0.0), 10),
            PaletteColor.from_lab(1, (50.3, 0.0, 0.0), 10),
        ),
        provenance=PROV,
    )
    ctx = InMemoryContext(seed=0)
    ctx.put("palette", palette)
    report = V.validate_palette(ctx, merge_delta_e=7.0)
    assert not report.passed
    assert report.findings[0].severity is Severity.FATAL
    assert "quantize-stage bug" in report.findings[0].message


def test_palette_warning_vs_fatal_by_preset() -> None:
    palette = Palette(
        colors=(
            PaletteColor.from_lab(0, (50.0, 0.0, 0.0), 10),
            PaletteColor.from_lab(1, (58.0, 0.0, 0.0), 10),
        ),
        provenance=PROV,
    )
    ctx = InMemoryContext(seed=0)
    ctx.put("palette", palette)
    warn = V.validate_palette(ctx, merge_delta_e=7.0, palette_warn_delta_e=12.0)
    assert warn.passed
    assert warn.findings[0].severity is Severity.WARNING

    fatal = V.validate_palette(
        ctx, merge_delta_e=7.0, palette_warn_delta_e=12.0, warn_is_fatal=True
    )
    assert not fatal.passed
    assert fatal.findings[0].severity is Severity.FATAL


# ----------------------------------------------------------------- fidelity


def test_fidelity_passes_with_perfect_correspondence(clean_pipeline) -> None:
    rg, ag, cs, _plan, _palette = clean_pipeline
    ctx = InMemoryContext(seed=0)
    ctx.put("curve_set", cs)
    ctx.put("arc_graph", ag)
    ctx.put("region_graph", rg)
    report = V.validate_fidelity(ctx)
    assert report.passed
    assert report.metrics["min_face_label_agreement"] >= 0.99


def test_fidelity_fatal_on_corrupted_face_label() -> None:
    """A face whose ``.label`` disagrees with its region's authoritative
    label must fail the correspondence audit."""
    from mysterycbn.model.records import Region, RegionGraph
    from mysterycbn.model.vector import ArcGraph, BezierSegment, Curve, CurveSet, Face

    def seg(p0: tuple[float, float], p1: tuple[float, float]) -> BezierSegment:
        a, b = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
        return BezierSegment(control=np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]))

    def curve(arc_id: int, p0: tuple[float, float], p1: tuple[float, float]) -> Curve:
        return Curve(
            arc_id=arc_id, segments=(seg(p0, p1),), corner_indices=(), max_fit_error_pt=0.0
        )

    good = (
        curve(0, (0, 0), (0, 10)),
        curve(1, (0, 10), (10, 10)),
        curve(2, (10, 10), (10, 0)),
        curve(3, (10, 0), (0, 0)),
    )
    corrupted_face = Face(
        face_id=0, label=7, outer_walk=((0, False), (1, False), (2, False), (3, False))
    )
    cs = CurveSet(curves=good, faces=(corrupted_face,), provenance=PROV)

    region = Region(0, 3, 100, (0, 0, 9, 9), (0, 0), 40, (5.0, 5.0))  # region's real label is 3
    cmap = np.zeros((10, 10), dtype=np.int32)
    rg = RegionGraph(regions=(region,), component_map=cmap, edges=(), provenance=PROV)
    ag = ArcGraph(arcs=(), faces=(), work_scale=1.0, provenance=PROV)

    ctx = InMemoryContext(seed=0)
    ctx.put("curve_set", cs)
    ctx.put("arc_graph", ag)
    ctx.put("region_graph", rg)
    report = V.validate_fidelity(ctx)
    assert not report.passed
    assert report.metrics["min_face_label_agreement"] == 0.0


def _partial_agreement_ctx(mismatch_px: int) -> InMemoryContext:
    """A single 10x10 face whose component_map has ``mismatch_px`` pixels
    belonging to a differently-labelled region -> agreement (100-n)/100."""
    from mysterycbn.model.records import Region, RegionGraph
    from mysterycbn.model.vector import ArcGraph, BezierSegment, Curve, CurveSet, Face

    def seg(p0: tuple[float, float], p1: tuple[float, float]) -> BezierSegment:
        a, b = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
        return BezierSegment(control=np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]))

    def curve(arc_id: int, p0: tuple[float, float], p1: tuple[float, float]) -> Curve:
        return Curve(
            arc_id=arc_id, segments=(seg(p0, p1),), corner_indices=(), max_fit_error_pt=0.0
        )

    curves = (
        curve(0, (0, 0), (0, 10)),
        curve(1, (0, 10), (10, 10)),
        curve(2, (10, 10), (10, 0)),
        curve(3, (10, 0), (0, 0)),
    )
    face = Face(face_id=0, label=3, outer_walk=((0, False), (1, False), (2, False), (3, False)))
    cs = CurveSet(curves=curves, faces=(face,), provenance=PROV)

    cmap = np.zeros((10, 10), dtype=np.int32)
    cmap.ravel()[:mismatch_px] = 1  # region 1 carries a different label
    regions = (
        Region(0, 3, 100 - mismatch_px, (0, 0, 9, 9), (0, 0), 40, (5.0, 5.0)),
        Region(1, 7, mismatch_px, (0, 0, 9, 9), (0, 0), 10, (2.0, 1.0)),
    )
    rg = RegionGraph(regions=regions, component_map=cmap, edges=(), provenance=PROV)
    ag = ArcGraph(arcs=(), faces=(), work_scale=1.0, provenance=PROV)

    ctx = InMemoryContext(seed=0)
    ctx.put("curve_set", cs)
    ctx.put("arc_graph", ag)
    ctx.put("region_graph", rg)
    return ctx


def test_fidelity_filler_face_uses_lower_floor() -> None:
    ctx = _partial_agreement_ctx(mismatch_px=5)  # agreement 0.95

    # Non-filler: 0.95 < 0.99 -> FATAL.
    assert not V.validate_fidelity(ctx).passed

    # Filler: 0.95 >= 0.90 filler floor -> passes.
    ctx.put("filler_region_ids", frozenset({0}))
    assert V.validate_fidelity(ctx).passed


def test_fidelity_filler_face_below_filler_floor_still_fatal() -> None:
    ctx = _partial_agreement_ctx(mismatch_px=15)  # agreement 0.85
    ctx.put("filler_region_ids", frozenset({0}))
    report = V.validate_fidelity(ctx)
    assert not report.passed
    assert "below floor 0.9" in report.findings[0].message


def test_fidelity_filler_floor_is_configurable() -> None:
    ctx = _partial_agreement_ctx(mismatch_px=5)  # agreement 0.95
    ctx.put("filler_region_ids", frozenset({0}))
    assert not V.validate_fidelity(ctx, fidelity_min_agreement_filler=0.97).passed
    assert V.validate_fidelity(ctx, fidelity_min_agreement_filler=0.9).passed


# ------------------------------------------------------------- repair loop


def test_run_validation_raises_quality_error_when_unrepairable() -> None:
    """A fatal palette violation has no declared repair path -- QualityError."""
    palette = _palette(4)
    rg, ag, cs, plan = _pipeline(_checkerboard(), palette)
    bad_palette = Palette(
        colors=(
            PaletteColor.from_lab(0, (50.0, 0.0, 0.0), 10),
            PaletteColor.from_lab(1, (50.2, 0.0, 0.0), 10),
            PaletteColor.from_lab(2, (60.0, 0.0, 0.0), 10),
            PaletteColor.from_lab(3, (70.0, 0.0, 0.0), 10),
        ),
        provenance=PROV,
    )
    ctx = _context(rg, ag, cs, plan, bad_palette)
    with pytest.raises(QualityError, match="palette"):
        V.run_validation(ctx)


def test_run_validation_returns_exactly_four_reports_for_output_bundle(clean_pipeline) -> None:
    """OutputBundle requires exactly these 4, in this order (DATA_MODEL_SPEC §19)."""
    rg, ag, cs, plan, palette = clean_pipeline
    ctx = _context(rg, ag, cs, plan, palette)
    reports = V.run_validation(ctx)
    assert len(reports) == 4
    assert [r.validator_name for r in reports] == [
        "fidelity",
        "topology",
        "printability",
        "palette",
    ]
