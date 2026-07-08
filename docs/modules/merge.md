# Module Design — Tiny Region Merge (`stages/graph/merge`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §11](../ENGINE_SPEC.md); cost function in [MATH_SPEC.md §11](../MATH_SPEC.md); quality gates in [QUALITY_SPEC.md QM-11](../quality/QUALITY_SPEC.md).

## Purpose

Enforce the printability floor (invariant I4's area component): after this stage no region is smaller than a physically colorable size. Input `RegionGraph` + `Palette`; output both replaced (component map re-derived, palette compacted).

## Algorithm — smallest-first greedy merge with perceptual cost

1. **Area constraint** — `A_min = π·(d_min_mm/2)²·ppmm²`, with ppmm derived from the working scale (`foundation/units`). `A_min` exceeding the content area is a `ConfigError`.
2. **Priority queue** — min-heap of sub-floor regions keyed by `(area, region_id)`, with lazy invalidation: an entry is dead if its region was absorbed or its recorded area is stale. A region that grows but stays sub-floor re-enters with its new key.
3. **Merge cost** — fold `r` into `argmin_n C(r, n)`:

   ```
   C(r, n) = ΔE00(μ(r), μ(n)) − λ · w_len(r, n) / perim(r),   λ = 15 (config merge.lambda_boundary, 0–50)
   ```

   The **color-similarity** term decides clear cases; the **boundary-similarity** (hug) term breaks near-ties toward the neighbor the sliver hugs — λ = 15 means a full-perimeter hug is worth 15 ΔE00 of color mismatch. Ties → larger neighbor, then lower id. A single-neighbor region (donut hole) merges regardless of cost.
4. **Update** — absorber inherits edges with boundary lengths summed; `perim(n′) = perim(n) + perim(r) − 2·w_len(r, n)` by inclusion–exclusion. Termination: each merge strictly decreases region count; heap re-entries are bounded by total insertions (≤ R−1 merges).
5. **Compaction** — surviving regions renumber to dense ids preserving relative order; palette entries with zero coverage drop, renumber map returned for provenance. Compaction is skipped (identity map) if it would leave < 2 colors — the model's palette floor; the degenerate one-region page is legal (validator warns).

Records and adjacency of the output graph are re-derived from the merged component map via the Connected Components sweeps, so the double-entry boundary identity is re-asserted structurally. Two surviving same-label regions may touch orthogonally after a merge — the component map, not the label partition, is authoritative in the graph domain.

## Rejected alternatives

Global graph-cut/MRF (nondeterministic solver order, 20×+ cost); pure-ΔE00 merging (slivers jump to distant-colored large fields → print-scale speckle); watershed flooding from large regions (inverts control, harder to bound). See ENGINE_SPEC §11.

## Quality requirements

- **QM-11 (Gate):** 0 regions with `area < A_min` post-stage (degenerate R = 1 exempt) — property-tested and benchmarked.
- **Fidelity guard (Gate):** mean ΔE00 of recolored pixels vs their new palette color ≤ 15 on fixtures — `benchmarks/quality/test_merge_quality.py` (measured 6.8–10.2 on the synthetic ladder).
- **Determinism:** chain merges resolve identically across runs (unit-tested).
- **Budget:** 20 000 → ~800 regions at 1600 px ≤ 1.0 s (measured ≈ 0.33 s).

## Artifacts

Requires `region_graph`, `palette`, `raster_working` (work scale for mm→px); provides `region_graph` + `palette` (both new generations, stage `merge_tiny` v1.0.0). Config section `merge` (`lambda_boundary`); the floor derives from `quality.d_min_mm`.

## Future

Saliency-weighted per-region floors (plugin, ENGINE_SPEC §14.3); learned cost function.
