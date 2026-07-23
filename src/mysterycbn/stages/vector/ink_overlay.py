"""Ink-line vectorization stage: turn the raster ``InkMask`` into render-only
black centerline polylines in page points (``InkOverlay``).

Runs after ``arcgraph`` so the single raster-px→pt transform Φ is fixed. It
recomputes Φ *identically* to ``arcgraph.build_arc_graph`` (same
``content_box``, same ``scale = arc_graph.work_scale``, same letterbox origin)
so the ink lines register pixel-exactly with the traced faces -- any drift here
shifts the ink off the artwork.

Vectorization skeletonizes the mask to 1-px centerlines, then emits one short
polyline per skeleton edge (each pair of 8-connected skeleton pixels, counted
once in row-major order). Per-edge segments keep the tracer fully deterministic
-- required for the SVG's byte-determinism (I2) -- and render as a continuous
line under round line caps/joins. An empty mask yields an empty overlay.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from skimage.morphology import skeletonize

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.ink import InkMask, InkOverlay
from mysterycbn.model.vector import ArcGraph
from mysterycbn.model.records import Provenance
from mysterycbn.stages.vector.arcgraph import content_box_pt

STAGE_NAME = "ink_overlay"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64
_DEFAULT_PAGE_MM = (215.9, 279.4, 12.7)
# Matches the region-boundary stroke weight (0.3 pt, see render/svg.py's
# STROKE_PT_DEFAULT = 0.3 * MM_PER_INCH / PT_PER_INCH) so ink lines don't
# read as a bolder outline than the rest of the line art -- a heavier ink
# stroke would trace the subject's silhouette clearly enough to give the
# mystery away.
STROKE_MM_DEFAULT = 0.3 * MM_PER_INCH / PT_PER_INCH
# 8-neighbourhood offsets in a fixed order (deterministic traversal).
_NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _trace_skeleton(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Decompose a 1-px skeleton into polylines of pixel coords, tracing each
    edge exactly once. Chains between endpoints/junctions (degree != 2) become
    one polyline; pure loops (all degree 2) are emitted from a deterministic
    start. Fully deterministic: pixels and neighbours are visited in fixed
    (sorted / ``_NEIGHBORS``) order -- required for SVG byte-determinism (I2)."""
    pixels = set(zip(*(a.tolist() for a in np.nonzero(skel)), strict=True))

    def nbrs(p: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = p
        return [(r + dr, c + dc) for dr, dc in _NEIGHBORS if (r + dr, c + dc) in pixels]

    degree = {p: len(nbrs(p)) for p in pixels}
    used: set[frozenset] = set()  # undirected edges consumed
    polylines: list[list[tuple[int, int]]] = []

    def walk(start: tuple[int, int], first: tuple[int, int]) -> None:
        path = [start, first]
        used.add(frozenset((start, first)))
        prev, cur = start, first
        while degree.get(cur, 0) == 2:
            step = next((n for n in nbrs(cur) if n != prev), None)
            if step is None or frozenset((cur, step)) in used:
                break
            used.add(frozenset((cur, step)))
            path.append(step)
            prev, cur = cur, step
        polylines.append(path)

    # Chains anchored at endpoints/junctions first (deterministic pixel order).
    for p in sorted(pixels):
        if degree[p] == 2:
            continue
        for n in nbrs(p):
            if frozenset((p, n)) not in used:
                walk(p, n)
    # Remaining pure loops (every pixel degree 2, no anchor).
    for p in sorted(pixels):
        for n in nbrs(p):
            if frozenset((p, n)) not in used:
                walk(p, n)
    return polylines


def vectorize_ink(
    mask: np.ndarray,
    *,
    scale: float,
    origin_xy: tuple[float, float],
) -> tuple[np.ndarray, ...]:
    """Skeleton-path polylines in pt. ``scale`` is Φ's pt/px; ``origin_xy`` is
    the letterbox origin ``(m_x, m_y)``. Pixel (r, c) center maps to
    ``(m_x + (c+1)·scale, m_y + (r+1)·scale)`` -- the normal-pixel form of Φ's
    doubled-corner mapping in ``arcgraph.build_arc_graph``."""
    if not mask.any():
        return ()
    skel = skeletonize(np.array(mask))
    m_x, m_y = origin_xy
    polylines: list[np.ndarray] = []
    for path in _trace_skeleton(skel):
        if len(path) < 2:
            continue
        polylines.append(
            np.array(
                [[m_x + (c + 1) * scale, m_y + (r + 1) * scale] for r, c in path],
                dtype=np.float64,
            )
        )
    return tuple(polylines)


class InkOverlayStage:
    """Stage wrapper: (``ink_mask``, ``arc_graph``) → ``ink_overlay``."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        page_mm: tuple[float, float, float] = _DEFAULT_PAGE_MM,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        stroke = section.get("stroke_mm", STROKE_MM_DEFAULT)
        if not isinstance(stroke, (int, float)) or float(stroke) <= 0.0:
            raise ConfigError(f"ink config: stroke_mm must be > 0, got {stroke!r}")
        self._stroke_mm = float(stroke)
        self._content_box = content_box_pt(page_mm)
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("ink_mask", "arc_graph")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("ink_overlay",)

    @property
    def config_section(self) -> str:
        return "ink"

    def run(self, ctx: PipelineContext) -> None:
        ink_mask = ctx.get("ink_mask")
        arc_graph = ctx.get("arc_graph")
        if not isinstance(ink_mask, InkMask) or not isinstance(arc_graph, ArcGraph):
            raise ConfigError("ink_overlay requires InkMask + ArcGraph artifacts")
        stroke_pt = self._stroke_mm * PT_PER_INCH / MM_PER_INCH
        provenance = Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=self._config_hash,
            source_hash=arc_graph.provenance.source_hash,
        )
        mask = ink_mask.mask
        h, w = mask.shape
        scale = arc_graph.work_scale
        box_x, box_y, box_w, box_h = self._content_box
        # Identical to arcgraph.build_arc_graph's Φ letterbox origin.
        m_x = box_x + (box_w - scale * w) / 2.0
        m_y = box_y + (box_h - scale * h) / 2.0
        polylines = vectorize_ink(mask, scale=scale, origin_xy=(m_x, m_y))
        ctx.put(
            "ink_overlay",
            InkOverlay(polylines=polylines, stroke_pt=stroke_pt, provenance=provenance),
        )
