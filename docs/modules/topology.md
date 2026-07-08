# Module Design — Topology Graph (`stages/vector/topology`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §14](../ENGINE_SPEC.md); planar multigraph in [MATH_SPEC.md §5.2](../MATH_SPEC.md), crack frame §1.2; data model in [DATA_MODEL_SPEC.md §9–§10](../DATA_MODEL_SPEC.md).

## Purpose

Impose node/edge structure on the crack boundary of the component map: find **junctions** and cut the boundary into **arcs** — maximal crack paths separating exactly one (left, right) region pair. This is the topological representation of the planar subdivision on which Arc Graph assembly, simplification, and the topology validator operate. Output: `TopologyGraph` (junctions + arcs), the shortest-lived artifact in the pipeline.

All topology runs in **exact doubled-integer crack coordinates** (corner `(u, v)` → `(2u−1, 2v−1)`, MATH_SPEC §1.2) — no floats touch incidence decisions.

## Algorithm

1. **Junction detection** — a crack-grid corner is a junction iff the 2×2 pixel block around it contains ≥ 3 distinct region ids (page exterior counts as id −1), or it is one of the 4 page corners. Local, O(1) per corner, vectorized as a sort-and-count over the four padded shifts.
2. **Arc tracing** — from every junction, each unvisited incident crack edge starts a walk that continues through degree-2 corners until the next junction, consuming edges as it goes (the edge grid doubles as the visited marker). Turn priority left/straight/right (the §13 rule) makes the continuation deterministic. Pair constancy between junctions is a theorem — a pair change at a corner implies ≥ 3 ids, hence a junction.
3. **Closed arcs** — after the junction pass, remaining edges form junction-free loops (islands). Scanning corners in lexicographic order anchors each loop at its smallest corner (stored with the anchor repeated at the end).
4. **Identity** — open arcs are stored in their lexicographically smaller direction (left/right swapped on reversal; comparing first edge vs last edge decides full lexicographic order, since no arc repeats an undirected edge). Ids are dense in `(min corner, left, right)` order — stable across runs.

## Guarantees and validation

The builder asserts Σ arc lengths = B (total crack-edge count) — combined with visited-marking this makes **no gaps / no overlaps** structural. `validate_topology` independently re-proves, vectorized over one flat edge array:

- **No overlaps** — every (min-corner, axis) edge key is unique across arcs.
- **No gaps** — the covered-edge count equals the component map's crack count.
- **Consistent face adjacency** — every edge's (left, right), re-derived from the component map by midpoint/normal lookup, matches its arc's annotation; `left ≠ right` by the arc model.
- **Euler identity** — `V − A + F = 1 + C` (MATH_SPEC §5.2) with `F = R + 1` and C counted by union-find over arc-endpoint junctions, closed arcs as their own components (each contributing V = 1, A = 1).

Any violation raises `StageError` — a constructive bug, never repairable. The stage wrapper always validates before binding `topology_graph`.

## Edge cases

- Single region: 4 page-corner junctions, 4 border arcs.
- Degree-4 corner (four regions meeting): one junction, four incident arcs.
- Island: one closed arc, no interior junctions.
- Loop arc from a junction back to itself: legal open arc with coincident endpoints.
- Degree-4 corners with only 2 ids (pinch) cannot occur on a 4-connected component map (both diagonal pairs connected elsewhere is planar-impossible); the turn rule still handles them deterministically if presented.

## Quality requirements

- Exactness: every crack edge in exactly one arc; endpoints are junctions or the arc is closed; Σ arc lengths = B — property-tested on random maps (Hypothesis) with the validator as the property.
- Determinism: arc ids, orientations, and anchors are reproducible — unit- and property-tested.
- Budget: ≤ 0.3 s at 1600 px (ENGINE_SPEC §26) — measured ≈ 0.16 s build + 0.03 s validate on a post-merge-density fixture (~1000 regions).

## Artifacts

Requires `region_graph` (its component map is the authoritative geometry); provides `topology_graph` (stage `topology` v1.0.0, no configuration parameters). Consumed solely by Arc Graph assembly and dropped from context once `ArcGraph` exists.

## Future

None anticipated — the stage is deliberately minimal (per spec).
