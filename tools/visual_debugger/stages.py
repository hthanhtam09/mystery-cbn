"""Maps the engine's internal artifact names to the Sprint 22 stage labels
and dispatches each bound artifact to its ``ArtifactView`` renderer
(docs/VISUAL_DEBUGGER.md §2).

The 11 labeled stages the brief calls out (Original -> Working Resolution
-> Quantized -> Label Map -> Region Graph -> Arc Graph -> Curves -> Labels
-> Legend -> SVG -> Preview) map onto artifact names as follows; "Quantized"
and "Label Map" are the same underlying ``label_map`` artifact viewed twice
(once with the just-quantized palette, once after denoise/merge refine it)
since ENGINE_SPEC's pipeline doesn't introduce a second distinct raster
artifact between them.
"""

from __future__ import annotations

from dataclasses import dataclass

from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.layout import LabelPlan, Legend
from mysterycbn.model.records import ImageStats, LabelMap, Palette, RasterImage, RegionGraph
from mysterycbn.model.vector import ArcGraph, CurveSet
from mysterycbn.render.png import PngPreviews
from mysterycbn.render.svg import SvgDocument
from mysterycbn.stages.raster.load import SourceBytes
from tools.visual_debugger.render_artifact import (
    ArtifactView,
    view_arc_graph,
    view_curve_set,
    view_image_stats,
    view_label_map,
    view_label_plan,
    view_legend,
    view_palette,
    view_png_previews,
    view_raster_source,
    view_raster_working,
    view_region_graph,
    view_source_bytes,
    view_svg_document,
)

# Report stage label -> the artifact name(s) it displays, in report order.
STAGE_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Original", ("source_bytes",)),
    ("Working Resolution", ("raster_working",)),
    ("Image Stats", ("image_stats",)),
    ("Quantized", ("label_map", "palette")),
    ("Label Map (post-merge)", ("label_map",)),
    ("Region Graph", ("region_graph",)),
    ("Arc Graph", ("arc_graph",)),
    ("Curves", ("curve_set",)),
    ("Labels", ("label_plan",)),
    ("Legend", ("legend",)),
    ("SVG", ("svg",)),
    ("Preview", ("png_previews",)),
)


@dataclass(frozen=True)
class StageViews:
    """One report stage: its label, the artifacts it needed, and their
    rendered views (multiple named views for multi-artifact/multi-file
    stages, e.g. ``png_previews`` yields both "lineart" and "solved")."""

    label: str
    artifact_names: tuple[str, ...]
    views: dict[str, ArtifactView]
    available: bool


def _render_one(ctx: InMemoryContext, name: str) -> dict[str, ArtifactView]:
    if not ctx.has(name):
        return {}
    artifact = ctx.get(name)
    if name == "source_bytes":
        assert isinstance(artifact, SourceBytes)
        return {name: view_source_bytes(artifact)}
    if name == "raster_source":
        assert isinstance(artifact, RasterImage)
        return {name: view_raster_source(artifact)}
    if name == "raster_working":
        assert isinstance(artifact, RasterImage)
        return {name: view_raster_working(artifact)}
    if name == "image_stats":
        assert isinstance(artifact, ImageStats)
        return {name: view_image_stats(artifact)}
    if name == "label_map":
        assert isinstance(artifact, LabelMap)
        palette = ctx.get("palette") if ctx.has("palette") else None
        assert palette is None or isinstance(palette, Palette)
        return {name: view_label_map(artifact, palette=palette)}
    if name == "palette":
        assert isinstance(artifact, Palette)
        return {name: view_palette(artifact)}
    if name == "region_graph":
        assert isinstance(artifact, RegionGraph)
        return {name: view_region_graph(artifact)}
    if name == "arc_graph":
        assert isinstance(artifact, ArcGraph)
        return {name: view_arc_graph(artifact)}
    if name == "curve_set":
        assert isinstance(artifact, CurveSet)
        return {name: view_curve_set(artifact)}
    if name == "label_plan":
        assert isinstance(artifact, LabelPlan)
        return {name: view_label_plan(artifact)}
    if name == "legend":
        assert isinstance(artifact, Legend)
        return {name: view_legend(artifact)}
    if name == "svg":
        assert isinstance(artifact, SvgDocument)
        return {name: view_svg_document(artifact)}
    if name == "png_previews":
        assert isinstance(artifact, PngPreviews)
        return view_png_previews(artifact)
    raise KeyError(f"no renderer registered for artifact {name!r}")


def build_stage_views(ctx: InMemoryContext) -> tuple[StageViews, ...]:
    """One ``StageViews`` per labeled report stage, in pipeline order."""
    result = []
    for label, artifact_names in STAGE_LABELS:
        views: dict[str, ArtifactView] = {}
        for name in artifact_names:
            views.update(_render_one(ctx, name))
        result.append(
            StageViews(
                label=label,
                artifact_names=artifact_names,
                views=views,
                available=bool(views),
            )
        )
    return tuple(result)
