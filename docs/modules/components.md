# Module Design — Connected Components (`stages/graph/components`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §9–§10](../ENGINE_SPEC.md); formulas in [MATH_SPEC.md §5.1](../MATH_SPEC.md); data model in [DATA_MODEL_SPEC.md §7–§8](../DATA_MODEL_SPEC.md).

## Purpose

Convert the denoised `LabelMap` into discrete regions — maximal **4-connected** sets of equal-label pixels — and build the region adjacency graph (RAG) that every later merge/split/label decision operates on. Output: `RegionGraph` (regions + component map + weighted edges).

Connectivity is an invariant, not a knob (ENGINE_SPEC §1.3): 8-connectivity would break planarity of the crack partition.

## Algorithm

1. **Labeling** — union-find component labeling (`skimage.measure.label`, connectivity=1), then renumbering to raster-scan first-occurrence order: region ids are dense, deterministic, top-left-first. O(N α(N)).
2. **Region records** — one bincount sweep per statistic:

   | Field | Definition |
   |---|---|
   | `label` | palette index of the region's pixels (label-homogeneous by construction) |
   | `area_px` | pixel count |
   | `bbox` | tight `(row_min, col_min, row_max, col_max)`, inclusive |
   | `seed_px` | first pixel in raster order (reconstruction anchor) |
   | `centroid` | mean of pixel centers, `(row, col)` f64 |
   | `perimeter_px` | `Σ_b w_len(a, b) + border_len(a)` — crack-edge count incl. page border |

3. **Adjacency sweep** — for every horizontal/vertical pixel pair `(p, q)` with different region ids, increment `w_len[(min, max)]`; page-border pixels accumulate `border_len`. Edges get `w_col = ΔE00(palette[label(a)], palette[label(b)])` from the palette's cached table. Edges are `(a, b, w_col, w_len)` with `a < b`, sorted lexicographically; `RegionGraph.neighbors(id)` derives the neighbor sets. Exact, O(N), allocation-light.

## Quality requirements

- **Exactness:** output equals the mathematical 4-connected partition — property-tested against a brute-force flood fill (Hypothesis, random rasters).
- **Double-entry identity:** `Σ_e w_len + Σ_r border_len = B` (total crack-edge count) — asserted at construction and property-tested; §25.2 cross-checks the same count against crack tracing.
- **Determinism:** identical ids/edges across runs (no RNG anywhere); id-order stability unit-tested.
- **Budget:** labeling ≤ 0.2 s, graph build ≤ 0.3 s at 1600 px, R ≤ 50 000 (ENGINE_SPEC §26).

## Edge cases

- R = 1 (flat page): one node, zero edges — legal end-to-end.
- Donut topology: hole is a distinct region, adjacent only to the ring.
- Two same-label regions may be diagonal neighbors (never edge-connected: orthogonal equal-label contact merges them); after §8 tie-breaks such pairs can share a crack edge with ΔE00 = 0, and §11 merges them first.
- Tens of thousands of regions pre-merge on noisy input — ids are int32 by contract.

## Artifacts

Requires `label_map` + `palette`; provides `region_graph` (`RegionGraph`, DATA_MODEL_SPEC §8). Provenance: stage `regions` v1.0.0; `source_hash` inherited from the label map.

## Future

Run-length labeling if profiling ever shows this stage on the critical path; incremental edge updates exposed for §11/§12 (currently those stages rebuild bookkeeping themselves).
