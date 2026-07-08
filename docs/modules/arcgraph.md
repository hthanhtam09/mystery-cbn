# Module Design — Arc Graph (`stages/vector/arcgraph`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §15](../ENGINE_SPEC.md); Euler form [MATH_SPEC.md §5.2](../MATH_SPEC.md), frame map Φ §1.3; data model [DATA_MODEL_SPEC.md §11](../DATA_MODEL_SPEC.md).

## Purpose

Assemble the planar map from the Topology Graph's shared boundary arcs: faces (regions as ordered arc walks with orientation flags), face↔region↔palette-label correspondence, exterior-face handling, topology verification, and the **single** application of `work_scale` — the last stage allowed to know about pixels.

## Algorithm — half-edge face walking

1. Every arc yields two directed half-arcs: the forward side carries `left_region` on its left, the reversed side `right_region` (the **shared-boundary** primitive — one polyline, two faces).
2. Outgoing half-arcs are indexed by `(junction, region-on-left, outgoing direction)`. The continuation of a face walk is the sharpest **left turn** from the incoming direction among same-region-on-left candidates — the textbook rotate-at-twin rule specialized to axis-aligned crack geometry, so all angular comparisons are table lookups (no `atan2`). Direction matters in the key: a region may pass a junction twice (diagonal self-touch), which the Hypothesis property test surfaced during development.
3. Each closed walk is one ring of its left region. Ring classification by exact integer shoelace sign (doubled coordinates, outer-positive convention of MATH_SPEC §7.1): the positive ring is the face's `outer_walk` (exactly one asserted), negative rings are `hole_walks`, sorted by min anchor. Hole attachment is therefore data-driven and handles arbitrary nesting (donut-in-donut) without recursion. Exterior (−1) rings are checked and discarded — the exterior face is not stored.
4. Walks are canonicalized (rotated to start at their minimal `(arc_id, reversed)` entry) — bitwise deterministic output.

## Topology preservation (all checked before scaling, in exact integers)

- **Euler identity** `V − A + F = 1 + C`, closed arcs as their own components with a virtual anchor vertex.
- **Every arc borders exactly two faces** counting sides — structural (each half-arc is consumed by exactly one walk); the unit/property tests additionally verify stored reference counts (border arcs appear once because the exterior face is not stored).
- **Constant region-on-left per walk** — asserted at every step.
- **Exact partition identity** Σ signed face areas = page area (holes negative, integer shoelace pre-scaling; the ±0.01 % float form is subsumed by exactness).

Violations raise `StageError` — a §13/§14 bug surfaced, never repaired silently.

## Scaling (Φ, applied exactly once)

`s = min(C_w/W, C_h/H)` (aspect-preserving letterbox in the content box, margins + centering offsets); doubled corner `(r, c)` maps to `(x, y)_pt = (m_x + (c+1)s/2, m_y + (r+1)s/2)`. Output arc points are `(x_pt, y_pt)`; `s` is stored as `work_scale` on the artifact (provenance of the single scaling — unit-tested against the coordinate ratio).

## Rejected alternatives

Full DCEL library (heavy native dependency for one use); point-in-polygon hole attachment (O(F²), float-fragile — sign classification is exact and O(E)); per-region contour rendering (reintroduces the double-boundary flaw §13 eliminated).

## Quality requirements

- Euler identity, two-faces-per-arc, exact area partition — asserted per run and property-tested over random label maps.
- Determinism — golden test over the full artifact dump.
- Budget: ≤ 0.3 s for A ≤ 20 000 (ENGINE_SPEC §26) — measured ≈ 0.12 s at 1600 px, ~1200 regions.

## Artifacts

Requires `topology_graph` + `region_graph`; provides `arc_graph` (`ArcGraph`: arcs in pt, faces, `work_scale`; stage `arcgraph` v1.0.0). Config section `page` (width/height/margin mm — page geometry, not stage knobs). Consumed by Simplification, Smoothing, Bézier Fitting, and the topology validator.

## Future

None anticipated — frozen with §13/§14 as "the heart" (per spec).
