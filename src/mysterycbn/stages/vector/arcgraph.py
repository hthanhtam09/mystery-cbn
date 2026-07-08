"""Arc Graph stage: assemble the planar map from the topology graph
(ENGINE_SPEC.md §15; Euler form MATH_SPEC §5.2; frame map Φ MATH_SPEC §1.3).

Faces are built by half-edge face walking over the shared boundary arcs:

1. Every arc yields two directed half-arcs; the forward side carries
   ``left_region`` on its left, the reversed side ``right_region``.
2. At a junction, the continuation of a face walk is the unique outgoing
   half-arc with the same region on its left (uniqueness is a property of
   the 4-connected crack grid: a region's boundary enters and leaves a
   junction exactly once — asserted at build time). Walking "same region on
   left" is equivalent to the textbook rotate-at-twin traversal, but needs
   no angle table on axis-aligned geometry.
3. Each closed walk is one face ring of its left region. The ring with
   positive signed area (outer-positive convention, MATH_SPEC §7.1) is the
   region's **outer walk**; negative rings are its **holes**, sorted by min
   anchor. The exterior (−1) rings are checked and discarded — the exterior
   face is not stored.

**Topology preservation** is verified before any coordinate leaves the exact
doubled-integer frame: Euler identity ``V − A + F = 1 + C`` (closed arcs as
their own components), every half-arc consumed exactly once (each arc borders
exactly two faces counting sides), constant region-on-left per walk, and the
exact partition identity Σ signed face areas = page area (integer shoelace,
holes negative). Failures raise ``StageError`` — a §13/§14 bug, never
repaired silently.

Only then is Φ applied — the single place ``work_scale`` touches coordinates
(aspect-preserving letterbox into the content box). Output arc points are
``(x_pt, y_pt)``; the applied scale is stored on the artifact.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Provenance, RegionGraph
from mysterycbn.model.vector import EXTERIOR_ID, Arc, ArcGraph, Face, TopologyGraph

STAGE_NAME = "arcgraph"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

# US Letter with 12.7 mm margins (core PageConfig defaults).
_DEFAULT_PAGE_MM = (215.9, 279.4, 12.7)

_HalfArc = tuple[int, bool]  # (arc_id, reversed)
_Walk = tuple[_HalfArc, ...]


def _fail(message: str) -> StageError:
    return StageError(message, stage_name=STAGE_NAME, config_hash=_UNSET_HASH)


def _half_arc_ends(arc: Arc, rev: bool) -> tuple[tuple[int, int], tuple[int, int], int]:
    """(tail junction, head junction, region-on-left) of one half-arc."""
    p0 = (int(arc.points[0, 0]), int(arc.points[0, 1]))
    p1 = (int(arc.points[-1, 0]), int(arc.points[-1, 1]))
    if rev:
        return p1, p0, arc.right_region
    return p0, p1, arc.left_region


def _segment_dir(p0: np.ndarray, p1: np.ndarray) -> int:
    """Direction code (E=0, S=1, W=2, N=3 — clockwise, y down) of a unit
    crack segment in doubled coords."""
    if p1[0] == p0[0]:
        return 0 if p1[1] > p0[1] else 2
    return 1 if p1[0] > p0[0] else 3


def _half_arc_dirs(arc: Arc, rev: bool) -> tuple[int, int]:
    """(outgoing direction at tail, incoming direction at head)."""
    pts = arc.points
    if rev:
        return _segment_dir(pts[-1], pts[-2]), _segment_dir(pts[1], pts[0])
    return _segment_dir(pts[0], pts[1]), _segment_dir(pts[-2], pts[-1])


_Continuation = dict[tuple[tuple[int, int], int, int], _HalfArc]


def _outgoing_table(arcs: tuple[Arc, ...]) -> tuple[_Continuation, list[_HalfArc]]:
    """Outgoing half-arcs keyed by (junction, region-on-left, direction).

    A region may pass a junction twice (diagonal self-touch), so direction is
    part of the key; the walk resolves with the turn rule.
    """
    continuation: _Continuation = {}
    open_halves: list[_HalfArc] = []
    for arc in arcs:
        if arc.closed:
            continue
        for rev in (False, True):
            tail, _, left = _half_arc_ends(arc, rev)
            out_dir, _ = _half_arc_dirs(arc, rev)
            key = (tail, left, out_dir)
            if key in continuation:
                raise _fail(f"duplicate outgoing half-arc at {tail} dir {out_dir}")
            continuation[key] = (arc.arc_id, rev)
            open_halves.append((arc.arc_id, rev))
    return continuation, open_halves


def _next_half_arc(
    h: _HalfArc, region: int, arcs: tuple[Arc, ...], continuation: _Continuation
) -> _HalfArc:
    """Continuation of a face walk after ``h``: the sharpest LEFT turn among
    outgoing half-arcs with the same region on the left — the half-edge
    rotation rule specialized to axis-aligned crack geometry."""
    _, head, _ = _half_arc_ends(arcs[h[0]], h[1])
    _, in_dir = _half_arc_dirs(arcs[h[0]], h[1])
    for turn in ((in_dir + 3) & 3, in_dir, (in_dir + 1) & 3):  # L, S, R
        nxt = continuation.get((head, region, turn))
        if nxt is not None:
            return nxt
    raise _fail(f"face walk of region {region} dead-ends at {head}")


def _trace_walk(
    start: _HalfArc,
    arcs: tuple[Arc, ...],
    continuation: _Continuation,
    used: set[_HalfArc],
) -> tuple[int, _Walk]:
    """One closed face walk from ``start``; returns (region-on-left, walk)."""
    _, _, region = _half_arc_ends(arcs[start[0]], start[1])
    walk: list[_HalfArc] = []
    h = start
    while True:
        if h in used:
            raise _fail(f"half-arc {h} visited twice during face walk")
        used.add(h)
        walk.append(h)
        _, _, left = _half_arc_ends(arcs[h[0]], h[1])
        if left != region:
            raise _fail(f"face walk of region {region} strayed onto region {left}")
        h = _next_half_arc(h, region, arcs, continuation)
        if h == start:
            break
    pivot = walk.index(min(walk))
    return region, tuple(walk[pivot:] + walk[:pivot])


def _face_walks(topology: TopologyGraph) -> dict[int, list[_Walk]]:
    """All closed face walks, keyed by region-on-left (exterior included)."""
    arcs = topology.arcs
    continuation, open_halves = _outgoing_table(arcs)
    walks: dict[int, list[_Walk]] = {}
    used: set[_HalfArc] = set()
    for start in open_halves:
        if start not in used:
            region, walk = _trace_walk(start, arcs, continuation, used)
            walks.setdefault(region, []).append(walk)
    for arc in arcs:  # closed arcs: one single-arc ring per side
        if arc.closed:
            walks.setdefault(arc.left_region, []).append(((arc.arc_id, False),))
            walks.setdefault(arc.right_region, []).append(((arc.arc_id, True),))
    return walks


def _walk_ring(walk: _Walk, arcs: tuple[Arc, ...]) -> np.ndarray:
    """Closed ring of doubled-int corners (last point = repeat of first,
    dropped), concatenating walk arcs head-to-tail."""
    parts = []
    for arc_id, rev in walk:
        pts = arcs[arc_id].points.astype(np.int64)
        if rev:
            pts = pts[::-1]
        parts.append(pts[:-1])  # head duplicates the next arc's tail
    return np.concatenate(parts)


def _signed_area_2x(ring: np.ndarray) -> int:
    """2 × signed area of a ring in doubled coords (exact integer; the
    outer-positive convention of MATH_SPEC §7.1: y-down, region-on-left)."""
    y, x = ring[:, 0], ring[:, 1]
    yn, xn = np.roll(y, -1), np.roll(x, -1)
    return -int((x * yn - xn * y).sum())


def content_box_pt(page_mm: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """(origin_x, origin_y, width, height) of the content box in pt."""
    width_mm, height_mm, margin_mm = page_mm
    if width_mm - 2 * margin_mm <= 0 or height_mm - 2 * margin_mm <= 0:
        raise ConfigError("margins leave no printable content area")
    to_pt = PT_PER_INCH / MM_PER_INCH
    return (
        margin_mm * to_pt,
        margin_mm * to_pt,
        (width_mm - 2 * margin_mm) * to_pt,
        (height_mm - 2 * margin_mm) * to_pt,
    )


def _assemble_faces(
    walks: dict[int, list[_Walk]],
    topology: TopologyGraph,
    region_graph: RegionGraph,
    *,
    page_px: tuple[int, int],
) -> tuple[Face, ...]:
    """Faces 1:1 with regions (face_id = region_id): the positive-area ring
    is the outer walk, negative rings are holes (sorted by min anchor); the
    exterior rings are checked and dropped. Asserts the exact partition
    identity Σ signed face areas = page area (no gaps, no overlaps)."""
    n_regions = len(region_graph.regions)
    faces = []
    area_sum_2x = 0
    for region in range(n_regions):
        rings = walks.get(region)
        if not rings:
            raise _fail(f"region {region} has no face walk")
        outer: list[_Walk] = []
        holes: list[tuple[tuple[int, int], _Walk]] = []
        for ring_walk in rings:
            ring = _walk_ring(ring_walk, topology.arcs)
            area2 = _signed_area_2x(ring)
            area_sum_2x += area2
            if area2 > 0:
                outer.append(ring_walk)
            else:
                holes.append(((int(ring[:, 0].min()), int(ring[:, 1].min())), ring_walk))
        if len(outer) != 1:
            raise _fail(f"region {region} has {len(outer)} outer rings (expected 1)")
        holes.sort()
        faces.append(
            Face(
                face_id=region,
                label=region_graph.regions[region].label,
                outer_walk=outer[0],
                hole_walks=tuple(hw for _, hw in holes),
            )
        )
    if not walks.get(EXTERIOR_ID):
        raise _fail("exterior face has no ring")
    if set(walks) - set(range(n_regions)) - {EXTERIOR_ID}:
        raise _fail("face walk references a region outside the graph")
    h, w = page_px  # doubled coords → 2×area is 8× the px² area
    if area_sum_2x != 8 * h * w:
        raise _fail(f"face areas sum to {area_sum_2x / 8} px², page is {h * w} px²")
    return tuple(faces)


def _check_euler(topology: TopologyGraph, n_regions: int) -> None:
    """Euler identity ``V − A + F = 1 + C`` (MATH_SPEC §5.2); closed arcs
    count as their own components with a virtual anchor vertex."""
    n_closed = sum(1 for a in topology.arcs if a.closed)
    parent = {i: i for i in range(len(topology.junctions))}
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(topology.junctions.tolist())}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for arc in topology.arcs:
        if not arc.closed:
            tail, head, _ = _half_arc_ends(arc, False)
            parent[find(index[tail])] = find(index[head])
    components = len({find(i) for i in parent}) + n_closed
    v_count = len(topology.junctions) + n_closed
    if v_count - len(topology.arcs) + (n_regions + 1) != 1 + components:
        raise _fail(
            f"Euler identity violated: V={v_count} A={len(topology.arcs)} "
            f"F={n_regions + 1} C={components}"
        )


def build_arc_graph(
    topology: TopologyGraph,
    region_graph: RegionGraph,
    *,
    content_box: tuple[float, float, float, float],
    config_hash: str = _UNSET_HASH,
) -> ArcGraph:
    """Full §15 assembly: face walks, topology checks, then the single Φ."""
    h, w = region_graph.component_map.shape
    n_regions = len(region_graph.regions)
    walks = _face_walks(topology)
    faces = _assemble_faces(walks, topology, region_graph, page_px=(h, w))
    _check_euler(topology, n_regions)

    # Φ — the single scaling (MATH_SPEC §1.3): doubled corner (r, c) sits at
    # raster (x, y) = (c/2, r/2), so x_pt = m_x + (c + 1)·s/2, y analogous.
    box_x, box_y, box_w, box_h = content_box
    scale = min(box_w / w, box_h / h)
    m_x = box_x + (box_w - scale * w) / 2.0
    m_y = box_y + (box_h - scale * h) / 2.0
    scaled_arcs = tuple(
        Arc(
            arc_id=a.arc_id,
            points=np.stack(
                [
                    m_x + (a.points[:, 1] + 1.0) * (scale / 2.0),
                    m_y + (a.points[:, 0] + 1.0) * (scale / 2.0),
                ],
                axis=1,
            ),
            left_region=a.left_region,
            right_region=a.right_region,
            closed=a.closed,
        )
        for a in topology.arcs
    )
    return ArcGraph(
        arcs=scaled_arcs,
        faces=tuple(faces),
        work_scale=scale,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=topology.provenance.source_hash,
        ),
    )


class ArcGraphStage:
    """Stage wrapper: (``topology_graph``, ``region_graph``) → ``arc_graph``."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        width = section.get("width_mm", _DEFAULT_PAGE_MM[0])
        height = section.get("height_mm", _DEFAULT_PAGE_MM[1])
        margin = section.get("margin_mm", _DEFAULT_PAGE_MM[2])
        if not all(isinstance(v, (int, float)) for v in (width, height, margin)):
            raise ConfigError("page config: width_mm, height_mm, margin_mm must be numbers")
        self._content_box = content_box_pt((float(width), float(height), float(margin)))  # type: ignore[arg-type]
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("topology_graph", "region_graph")

    @property
    def provides(self) -> tuple[str, ...]:
        return ("arc_graph",)

    @property
    def config_section(self) -> str:
        return "page"

    def run(self, ctx: PipelineContext) -> None:
        topology = ctx.get("topology_graph")
        region_graph = ctx.get("region_graph")
        if not isinstance(topology, TopologyGraph) or not isinstance(region_graph, RegionGraph):
            raise ConfigError("arcgraph requires TopologyGraph + RegionGraph artifacts")
        ctx.put(
            "arc_graph",
            build_arc_graph(
                topology,
                region_graph,
                content_box=self._content_box,
                config_hash=self._config_hash,
            ),
        )
