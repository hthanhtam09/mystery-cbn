"""Drives the real engine stages (region graph -> topology -> arc graph ->
curves -> labels -> render -> validate) on a fixture, collecting per-stage
timings via the engine's own ``InMemoryTracer`` (ARCHITECTURE.md §12) so the
benchmark harness measures the same code path production runs use --
never a re-implementation (BENCHMARK_SPEC.md §6).

Starts from a synthetic ``LabelMap`` (post-quantize artifact) rather than
raw pixels: no raster fixture assets exist yet (see ``fixtures.py``), and
the raster-domain stages (load/preprocess/quantize) already have their own
per-stage benchmarks in ``benchmarks/perf/``. This harness exercises the
graph/vector/layout/render/validate stages, which is where BENCHMARK_SPEC's
full QM battery and most of the §26 stage-budget table lives.
"""

from __future__ import annotations

import resource
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from benchmarks.framework.fixtures import Fixture
from mysterycbn.foundation.tracing import InMemoryTracer
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.layout import Legend
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.model.vector import ArcGraph, CurveSet
from mysterycbn.render.svg import render_svg
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import place_labels
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph
from mysterycbn.validate.report import ValidationSettings, run_validation

PAGE_MM = (215.9, 279.4, 12.7)
_PROV = Provenance("benchmark", "1.0.0", "0" * 64, "1" * 64)

try:
    from mysterycbn.render.pdf import render_pdf

    _HAS_PDF = True
except ImportError:  # pdf extras (reportlab/PyMuPDF) not installed
    _HAS_PDF = False


def _palette_for(n_colors: int) -> Palette:
    """A well-separated palette on an LAB hue wheel (min pairwise ΔE00 stays
    >= 12 -- the QM-16 warn floor -- up to 10 entries at this radius; see
    ``test_validate_perf.py`` for the same construction and rationale).

    The Palette model requires >= 2 entries (DATA_MODEL_SPEC's palette
    floor); a genuinely degenerate 1-color fixture (F-degen-1) still needs a
    2-entry palette here even though only index 0 is ever painted -- the
    same accommodation the real quantize stage would make for an all-flat
    input (index 1 simply has zero coverage).
    """
    n_colors = max(n_colors, 2)
    return Palette(
        colors=tuple(
            PaletteColor.from_lab(
                i,
                (
                    55.0,
                    45.0 * np.cos(2 * np.pi * i / n_colors),
                    45.0 * np.sin(2 * np.pi * i / n_colors),
                ),
                1000,
            )
            for i in range(n_colors)
        ),
        provenance=_PROV,
    )


def _identity_legend(palette: Palette) -> Legend:
    k = palette.size
    chips = tuple((i, (20.0 + i * 26.0, 20.0), 20.0) for i in range(k))
    return Legend(
        permutation=tuple(range(k)),
        chips=chips,
        band_rect=(10.0, 10.0, max(30.0, k * 26.0 + 10.0), 40.0),
        number_font_pt=8.0,
        provenance=_PROV,
    )


@dataclass(frozen=True)
class PipelineRun:
    """Everything one fixture run measured: artifacts, tracer snapshot,
    peak RSS delta, and (if built) validation reports."""

    fixture_id: str
    tracer_snapshot: dict[str, object]
    peak_rss_delta_mib: float
    curve_set: CurveSet
    arc_graph: ArcGraph
    region_graph: object
    svg_bytes: bytes
    pdf_bytes: bytes | None
    ctx: InMemoryContext

    @property
    def stage_wall_s(self) -> dict[str, float]:
        """Per-stage wall time from the tracer snapshot, typed (the
        snapshot itself is ``Mapping[str, Any]`` -- ``InMemoryTracer``'s
        own return type, ARCHITECTURE.md §12; a ``MappingProxyType``, not a
        plain ``dict``, hence the ``Mapping`` isinstance check)."""
        timings = self.tracer_snapshot["timings_s"]
        assert isinstance(timings, Mapping)
        return {str(k): float(v) for k, v in timings.items()}


def _rss_mib() -> float:
    """Current process peak RSS in MiB (``ru_maxrss`` is KiB on Linux,
    bytes on macOS -- normalized here since this runs on both in CI)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys

    return peak / 1024.0 if sys.platform == "linux" else peak / (1024.0 * 1024.0)


def run_pipeline(
    fixture: Fixture,
    *,
    page_mm: tuple[float, float, float] = PAGE_MM,
    d_min_mm: float | None = None,
    palette_factory: Callable[[int], Palette] | None = None,
) -> PipelineRun:
    """Run the region->render->validate pipeline on ``fixture``, tracing
    every stage. Raises whatever the stages raise (a benchmark run must not
    silently swallow a broken pipeline).

    ``d_min_mm`` overrides the printability floor (``ValidationSettings``
    default otherwise) -- lets callers exercise a specific difficulty
    preset's floor (``app/config_defaults.D_MIN_MM_BY_PRESET``) without
    duplicating this function (Sprint 24 cross-preset comparison).

    ``palette_factory`` overrides ``_palette_for`` -- this module's own
    single-radius hue wheel only clears the QM-16 merge_delta_e FATAL floor
    up to ~10-16 colors (documented limitation above); a caller exercising
    a preset with more colors (``hard`` = 24) needs a differently-built
    palette, without duplicating the rest of this function."""
    tracer = InMemoryTracer()
    rss_before = _rss_mib()

    build_palette = palette_factory if palette_factory is not None else _palette_for
    palette = build_palette(fixture.n_colors)
    label_map = LabelMap(labels=fixture.labels, provenance=_PROV)

    with tracer.span("regions"):
        region_graph = build_region_graph(label_map, palette)

    box = content_box_pt(page_mm)
    with tracer.span("topology"):
        topology = build_topology_graph(region_graph.component_map)
    with tracer.span("arcgraph"):
        arc_graph = build_arc_graph(topology, region_graph, content_box=box)
    with tracer.span("curves"):
        curve_set = fit_curves(arc_graph)
    with tracer.span("labels"):
        label_plan, findings = place_labels(curve_set, region_graph)
    if findings:
        raise RuntimeError(f"{fixture.fixture_id}: label placement had FATAL findings: {findings}")

    legend = _identity_legend(palette)
    with tracer.span("svg"):
        svg_bytes = render_svg(curve_set, label_plan, legend, palette, page_mm=page_mm)

    pdf_bytes: bytes | None = None
    if _HAS_PDF:
        with tracer.span("pdf"):
            pdf_bytes = render_pdf(curve_set, label_plan, legend, palette, page_mm=page_mm)

    ctx = InMemoryContext(seed=0)
    ctx.put("region_graph", region_graph)
    ctx.put("arc_graph", arc_graph)
    ctx.put("curve_set", curve_set)
    ctx.put("label_plan", label_plan)
    ctx.put("palette", palette)

    validation_settings = (
        ValidationSettings() if d_min_mm is None else ValidationSettings(d_min_mm=d_min_mm)
    )
    with tracer.span("validate"):
        run_validation(ctx, validation_settings)

    tracer.record_artifact_size("svg", len(svg_bytes))
    if pdf_bytes is not None:
        tracer.record_artifact_size("pdf", len(pdf_bytes))

    peak_delta = max(_rss_mib() - rss_before, 0.0)

    return PipelineRun(
        fixture_id=fixture.fixture_id,
        tracer_snapshot=dict(tracer.snapshot()),
        peak_rss_delta_mib=peak_delta,
        curve_set=curve_set,
        arc_graph=arc_graph,
        region_graph=region_graph,
        svg_bytes=svg_bytes,
        pdf_bytes=pdf_bytes,
        ctx=ctx,
    )
