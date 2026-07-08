"""Stage registry bootstrap: the missing wiring identified by the Sprint 18
architecture audit ("no code path registers any concrete stage class into
InMemoryStageRegistry"; ``grep -rn "InMemoryStageRegistry" src/`` previously
matched only the class's own definition).

This module registers every concrete Stage implementation this session's
Sprint 19 pipeline needs, under the default ("default") implementation
name for each pipeline slot. It performs no computation itself -- it is
pure wiring, matching ARCHITECTURE.md §8's "selection is by configuration,
never by import" plugin discovery contract.
"""

from __future__ import annotations

from mysterycbn.kernel.registry import InMemoryStageRegistry
from mysterycbn.render.pdf import PdfExportStage
from mysterycbn.render.png import PngPreviewStage
from mysterycbn.render.svg import SvgExportStage
from mysterycbn.stages.graph.components import ConnectedComponentsStage
from mysterycbn.stages.graph.merge import MergeTinyStage
from mysterycbn.stages.layout.labels import LabelPlacementStage
from mysterycbn.stages.layout.legend import LegendStage
from mysterycbn.stages.raster.analyze import AnalyzeStage
from mysterycbn.stages.raster.denoise import DenoiseStage
from mysterycbn.stages.raster.load import LoadStage
from mysterycbn.stages.raster.preprocess import PreprocessStage
from mysterycbn.stages.raster.quantize import QuantizeStage
from mysterycbn.stages.vector.arcgraph import ArcGraphStage
from mysterycbn.stages.vector.curves import CurveFitStage
from mysterycbn.stages.vector.simplify import SimplifyStage
from mysterycbn.stages.vector.topology import TopologyStage


# Pipeline slot name -> Stage instance, in the Sprint 19 declared order
# (Load -> Preprocess -> Analyze -> Quantize -> Denoise -> Region Graph ->
# Merge Tiny Regions -> Contour Extraction -> Simplify -> Curve Smoothing ->
# Label Placement -> Legend -> Validation -> SVG -> PDF -> PNG).
#
# "Contour Extraction" maps to two existing stages run back to back
# (topology graph junction/arc decomposition, then arc-graph face assembly
# + the Φ page-scale application) -- ENGINE_SPEC's own module numbering
# already splits this into two stages (§14 Topology Graph, §15 Arc Graph);
# Sprint 19 does not introduce a third "contours" stage, it reuses both.
#
# "Curve Smoothing" maps to CurveFitStage: Bézier fitting already performs
# G1-continuous smoothing as part of curve fitting (ENGINE_SPEC §18) -- no
# separate smoothing stage exists in the architecture's own implementation
# (confirmed by the Sprint 18 audit), and Sprint 19's brief is orchestration
# only, not new algorithm design, so no new smoothing stage is introduced.
def build_stage_factories(
    *,
    d_min_mm: float,
    seed: int,
    config_hash: str,
    page_mm: tuple[float, float, float],
) -> dict[str, object]:
    """Construct one instance of every stage this pipeline uses.

    Stages are stateless with respect to a single run's config (each reads
    its own frozen section once, in ``__init__``), so building fresh
    instances per ``convert()`` call is the simplest correct approach
    consistent with the Stage protocol's determinism requirement.

    ``page_mm`` is threaded through every stage that touches page geometry
    (arcgraph's Φ letterbox, legend band width, svg/pdf/png page canvas) so
    a single value is the source of truth -- avoiding the class of bug where
    ArcGraphStage and SvgExportStage silently disagree on page size.
    """
    width_mm, height_mm, margin_mm = page_mm
    page_section = {"width_mm": width_mm, "height_mm": height_mm, "margin_mm": margin_mm}
    return {
        "load": LoadStage(config_hash=config_hash),
        "preprocess": PreprocessStage({}, config_hash=config_hash),
        "analyze": AnalyzeStage({}, config_hash=config_hash),
        "quantize": QuantizeStage({}, seed=seed, config_hash=config_hash),
        "denoise": DenoiseStage({}, d_min_mm=d_min_mm, config_hash=config_hash),
        "regions": ConnectedComponentsStage({}),
        "merge_tiny": MergeTinyStage({}, d_min_mm=d_min_mm, config_hash=config_hash),
        "topology": TopologyStage(),
        "arcgraph": ArcGraphStage(page_section, config_hash=config_hash),
        "simplify": SimplifyStage(config_hash=config_hash),
        "bezier": CurveFitStage(config_hash=config_hash),
        "labels": LabelPlacementStage(config_hash=config_hash),
        "legend": LegendStage(page_width_mm=width_mm, margin_mm=margin_mm, config_hash=config_hash),
        "svg": SvgExportStage(page_mm=page_mm, config_hash=config_hash),
        "pdf": PdfExportStage(page_mm=page_mm, config_hash=config_hash),
        "png": PngPreviewStage(page_mm=page_mm),
    }


def build_registry(
    *,
    d_min_mm: float,
    seed: int,
    config_hash: str,
    page_mm: tuple[float, float, float],
) -> InMemoryStageRegistry:
    """A fully populated ``InMemoryStageRegistry`` for one ``convert()`` run."""
    registry = InMemoryStageRegistry()
    for slot, stage in build_stage_factories(
        d_min_mm=d_min_mm, seed=seed, config_hash=config_hash, page_mm=page_mm
    ).items():
        registry.register(slot, "default", stage)  # type: ignore[arg-type]
    return registry
