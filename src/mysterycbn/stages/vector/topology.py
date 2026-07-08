"""Topology Graph stage: junction/arc decomposition of the crack boundary
(ENGINE_SPEC.md §14; planar multigraph MATH_SPEC §5.2; frame MATH_SPEC §1.2).

Builds the topological representation of the planar subdivision induced by
the component map, in exact doubled-integer crack coordinates (corner
``(u, v)`` → ``(2u−1, 2v−1)``; no floats touch topology):

1. **Junctions** — a crack-grid corner is a junction iff the 2×2 pixel block
   around it contains ≥ 3 distinct region ids (page exterior = −1 counts),
   or it is one of the 4 page corners.
2. **Arcs** — maximal crack paths with constant (left, right) region pair,
   cut at junctions. Each undirected crack edge lands in exactly ONE arc
   (no gaps, no overlaps — structural: the walk marks edges and the builder
   asserts Σ arc lengths = B). Pair constancy between junctions is a theorem
   (a pair change implies ≥ 3 ids, hence a junction); degree-4 pinch corners
   with only 2 ids are resolved by the §13 turn priority (left, straight,
   right) so crossing strands pair deterministically.
3. **Closed arcs** — junction-free loops (islands) anchored at their
   lexicographically smallest corner, stored with the anchor repeated at the
   end. Arc ids are assigned in (min corner, left, right) order — stable
   across runs; open arcs are stored in their lexicographically smaller
   direction (left/right swapped accordingly).

``validate_topology`` re-proves the guarantees on the finished artifact:
edge coverage (no gaps/overlaps), per-edge (left, right) consistency with
the component map (consistent face adjacency), and the Euler identity
``V − A + F = 1 + C`` with C counted by union-find over arc endpoints and
closed arcs as their own components (MATH_SPEC §5.2). Violations raise
``StageError`` — constructive bugs, never repairable.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.records import Provenance, RegionGraph
from mysterycbn.model.vector import Arc, TopologyGraph

STAGE_NAME = "topology"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64


def _fail(message: str) -> StageError:
    return StageError(message, stage_name=STAGE_NAME, config_hash=_UNSET_HASH)


# Directions on the corner grid, clockwise in the y-down frame.
_E, _S, _W, _N = 0, 1, 2, 3


def _junction_mask(padded: np.ndarray) -> np.ndarray:
    """(H+1, W+1) bool: 2×2 distinct-id ≥ 3 rule plus the 4 page corners."""
    block = np.stack(
        [padded[:-1, :-1], padded[:-1, 1:], padded[1:, :-1], padded[1:, 1:]]
    )  # pixels NW, NE, SW, SE of each corner
    block = np.sort(block, axis=0)
    distinct = (np.diff(block, axis=0) != 0).sum(axis=0) + 1
    mask = distinct >= 3
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = True
    return np.asarray(mask, dtype=bool)


def _crack_edges(padded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean edge grids: ``e_east[u, v]`` = crack (u,v)–(u,v+1) exists,
    ``e_south[u, v]`` = crack (u,v)–(u+1,v) exists (flat, corner-indexed)."""
    h, w = padded.shape[0] - 2, padded.shape[1] - 2
    e_east = np.zeros((h + 1, w + 1), dtype=bool)
    e_south = np.zeros((h + 1, w + 1), dtype=bool)
    e_east[:, :w] = padded[: h + 1, 1 : w + 1] != padded[1 : h + 2, 1 : w + 1]
    e_south[:h, :] = padded[1 : h + 1, : w + 1] != padded[1 : h + 1, 1 : w + 2]
    return e_east, e_south


class _Tracer:
    """Mutable crack-walk state: edge grids double as visited markers."""

    def __init__(self, padded: np.ndarray, junction: np.ndarray, cols: int) -> None:
        e_east, e_south = _crack_edges(padded)
        self.padded = padded
        self.cols = cols
        self.total_cracks = int(e_east.sum()) + int(e_south.sum())
        self.junc = bytearray(junction.ravel().astype(np.uint8).tobytes())
        self.east = bytearray(e_east.ravel().astype(np.uint8).tobytes())
        self.south = bytearray(e_south.ravel().astype(np.uint8).tobytes())
        # Direction tables: corner-index delta, (edge array, owner delta).
        self.step = (1, cols, -1, -cols)  # E, S, W, N
        self.edge_of = ((self.east, 0), (self.south, 0), (self.east, -1), (self.south, -cols))
        # Per incoming direction: (edge array, owner offset, corner step,
        # new dir) for the left/straight/right candidates — the walk's hot
        # loop is tuple lookups and bytearray probes only.
        self.turns = tuple(
            tuple(
                (self.edge_of[cand][0], self.edge_of[cand][1], self.step[cand], cand)
                for cand in ((d + 3) & 3, d, (d + 1) & 3)
            )
            for d in range(4)
        )

    def take(self, c: int, d: int) -> bool:
        """Consume the crack edge leaving corner ``c`` in direction ``d``."""
        arr, off = self.edge_of[d]
        if arr[c + off]:
            arr[c + off] = 0
            return True
        return False

    def walk(self, c0: int, d0: int) -> tuple[list[int], bool]:
        """Trace from corner ``c0`` heading ``d0`` to the next junction (or
        back to ``c0``); returns (corner path, closed)."""
        junc, turns = self.junc, self.turns
        path = [c0]
        c, d = c0 + self.step[d0], d0
        path.append(c)
        append = path.append
        while not junc[c] and c != c0:
            for arr, off, delta, cand in turns[d]:
                i = c + off
                if arr[i]:
                    arr[i] = 0
                    d = cand
                    c += delta
                    append(c)
                    break
            else:  # pragma: no cover - impossible on a valid component map
                raise _fail(f"dead-end crack walk at corner {divmod(c, self.cols)}")
        return path, c == c0 and not junc[c0]

    def pair_of(self, c: int, d: int) -> tuple[int, int]:
        """(left, right) region ids of the first step from ``c`` along ``d``."""
        padded = self.padded
        u, v = divmod(c, self.cols)
        if d == _E:
            return int(padded[u, v + 1]), int(padded[u + 1, v + 1])
        if d == _S:
            return int(padded[u + 1, v + 1]), int(padded[u + 1, v])
        if d == _W:
            return int(padded[u + 1, v]), int(padded[u, v])
        return int(padded[u, v]), int(padded[u, v + 1])

    def trace(self, c0: int, d0: int) -> tuple[list[int], int, int, bool]:
        left, right = self.pair_of(c0, d0)
        path, closed = self.walk(c0, d0)
        return (path, left, right, closed)


def _trace_all_arcs(
    tracer: _Tracer, junction: np.ndarray
) -> list[tuple[list[int], int, int, bool]]:
    """Partition every crack edge into arcs (open first, then closed loops)."""
    raw_arcs = []
    # Open arcs: every unvisited edge incident to a junction, in lex order.
    for c0 in np.flatnonzero(junction.ravel()).tolist():
        for d0 in (_E, _S, _W, _N):
            if tracer.take(c0, d0):
                raw_arcs.append(tracer.trace(c0, d0))
    # Closed arcs: junction-free loops. Scanning corners in lex order makes
    # each loop's first-seen corner its lexicographically smallest anchor.
    leftover = np.flatnonzero(
        np.frombuffer(tracer.east, dtype=np.uint8).astype(bool)
        | np.frombuffer(tracer.south, dtype=np.uint8).astype(bool)
    )
    for c0 in leftover.tolist():
        for d0 in (_E, _S):
            while tracer.take(c0, d0):
                raw_arcs.append(tracer.trace(c0, d0))
    return raw_arcs


def _canonical_arcs(raw_arcs: list[tuple[list[int], int, int, bool]], cols: int) -> tuple[Arc, ...]:
    """Canonical orientation + deterministic id order (ENGINE_SPEC §14.4)."""
    keyed = []
    for path, left, right, closed in raw_arcs:
        if closed:
            body = path[:-1]
            if len(body) > 2 and tuple(body[1:]) > tuple(reversed(body[1:])):
                body = [body[0], *reversed(body[1:])]
                left, right = right, left
            path = [*body, body[0]]
        elif (path[-1], path[-2]) < (path[0], path[1]):
            # First-edge vs last-edge comparison decides full lexicographic
            # order (an arc cannot contain the same undirected edge twice).
            path = list(reversed(path))
            left, right = right, left
        pts = np.empty((len(path), 2), dtype=np.int64)
        arr = np.asarray(path, dtype=np.int64)
        pts[:, 0] = 2 * (arr // cols) - 1
        pts[:, 1] = 2 * (arr % cols) - 1
        min_corner = (int(pts[:, 0].min()), int(pts[:, 1].min()))
        # (first, second) point is unique per arc — each edge lives in one arc.
        keyed.append(((min_corner, left, right, path[0], path[1]), pts, left, right, closed))
    keyed.sort(key=lambda item: item[0])
    return tuple(
        Arc(
            arc_id=i,
            points=pts.astype(np.float64),
            left_region=left,
            right_region=right,
            closed=closed,
        )
        for i, (_, pts, left, right, closed) in enumerate(keyed)
    )


def build_topology_graph(
    component_map: np.ndarray, *, config_hash: str = _UNSET_HASH, source_hash: str = "1" * 64
) -> TopologyGraph:
    """§14 construction: junction detection, loop cutting, arc identity."""
    cmap = np.asarray(component_map, dtype=np.int64)
    h, w = cmap.shape
    padded = np.full((h + 2, w + 2), -1, dtype=np.int64)
    padded[1:-1, 1:-1] = cmap

    junction = _junction_mask(padded)
    tracer = _Tracer(padded, junction, w + 1)
    raw_arcs = _trace_all_arcs(tracer, junction)

    if sum(len(p) - 1 for p, *_ in raw_arcs) != tracer.total_cracks:
        raise _fail("crack edges not partitioned into arcs (gap/overlap)")

    arcs = _canonical_arcs(raw_arcs, w + 1)
    uu, vv = np.nonzero(junction)
    junctions = np.stack([2 * uu - 1, 2 * vv - 1], axis=1).astype(np.int64)
    return TopologyGraph(
        junctions=junctions,
        arcs=arcs,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=source_hash,
        ),
    )


def validate_topology(graph: TopologyGraph, component_map: np.ndarray) -> None:
    """Independent re-proof of the §14 guarantees; raises ``StageError``.

    - **No gaps / no overlaps.** Every undirected crack edge of the component
      map belongs to exactly one arc (multiset equality of edge sets).
    - **Consistent face adjacency.** Every edge's (left, right) pair, derived
      from the component map, equals its arc's annotation; ``left ≠ right``.
    - **Euler identity.** ``V − A + F = 1 + C`` (MATH_SPEC §5.2) with
      ``F = R + 1`` and C via union-find over arc-endpoint junctions, closed
      arcs counting as their own components.
    """
    cmap = np.asarray(component_map, dtype=np.int64)
    h, w = cmap.shape
    padded = np.full((h + 2, w + 2), -1, dtype=np.int64)
    padded[1:-1, 1:-1] = cmap

    # One flat edge array over all arcs, with per-edge expected (left, right).
    p0 = np.concatenate([arc.points[:-1] for arc in graph.arcs]).astype(np.int64)
    p1 = np.concatenate([arc.points[1:] for arc in graph.arcs]).astype(np.int64)
    counts = np.array([len(arc.points) - 1 for arc in graph.arcs], dtype=np.int64)
    exp_left = np.repeat(np.array([a.left_region for a in graph.arcs]), counts)
    exp_right = np.repeat(np.array([a.right_region for a in graph.arcs]), counts)

    d = p1 - p0  # (±2, 0) or (0, ±2)
    left_d = np.stack([-d[:, 1] // 2, d[:, 0] // 2], axis=1)
    mid = (p0 + p1) // 2  # edge midpoint in doubled coords
    left_px = (mid + left_d) // 2 + 1  # doubled pixel centre → padded index
    right_px = (mid - left_d) // 2 + 1
    bad = (padded[left_px[:, 0], left_px[:, 1]] != exp_left) | (
        padded[right_px[:, 0], right_px[:, 1]] != exp_right
    )
    if bool(bad.any()):
        arc_id = int(np.searchsorted(np.cumsum(counts), int(np.flatnonzero(bad)[0]) + 1))
        raise _fail(f"arc {arc_id}: (left, right) pair inconsistent with map")

    lo = np.minimum(p0, p1)
    span = 2 * (padded.shape[1] + 1)
    keys = (lo[:, 0] * span + lo[:, 1]) * 2 + (d[:, 0] != 0)  # min corner + axis
    covered = int(np.unique(keys).size)
    if covered != keys.size:
        raise _fail("crack edge covered twice (overlap)")
    e_east, e_south = _crack_edges(padded)
    if covered != int(e_east.sum()) + int(e_south.sum()):
        raise _fail(
            f"arc edges cover {covered} cracks, map has "
            f"{int(e_east.sum()) + int(e_south.sum())} (gap)"
        )

    # Euler: V − A + F = 1 + C.
    junction_ids = {(int(row[0]), int(row[1])) for row in graph.junctions.tolist()}
    parent: dict[tuple[int, int], tuple[int, int]] = {j: j for j in junction_ids}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_closed = 0
    for arc in graph.arcs:
        if arc.closed:
            n_closed += 1
            continue
        a = find((int(arc.points[0, 0]), int(arc.points[0, 1])))
        b = find((int(arc.points[-1, 0]), int(arc.points[-1, 1])))
        parent[a] = b
    components = len({find(j) for j in junction_ids}) + n_closed
    v_count = len(junction_ids) + n_closed
    faces = int(cmap.max()) + 2  # regions + page exterior
    if v_count - len(graph.arcs) + faces != 1 + components:
        raise _fail(
            f"Euler identity violated: V={v_count} A={len(graph.arcs)} F={faces} C={components}"
        )


class TopologyStage:
    """Stage wrapper: ``region_graph`` → ``topology_graph`` (validated)."""

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        # ENGINE_SPEC §14: no configuration parameters.
        del section
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("region_graph",)

    @property
    def provides(self) -> tuple[str, ...]:
        return ("topology_graph",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        graph = ctx.get("region_graph")
        if not isinstance(graph, RegionGraph):
            raise ConfigError("topology requires a RegionGraph artifact")
        topo = build_topology_graph(
            graph.component_map,
            config_hash=self._config_hash,
            source_hash=graph.provenance.source_hash,
        )
        validate_topology(topo, graph.component_map)
        ctx.put("topology_graph", topo)
