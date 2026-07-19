# ADR-002: Sprint 36 — Print-Aware Geometry Simplification Stage

**Status:** Proposed (design only — no code in this sprint until approved).
**Date:** Sprint 36.
**Supersedes:** none. **Related:** ADR-001 (orchestration layer this stage plugs into).

## Context

The engine passes topology validation, determinism checks, region-graph validation, and
SVG/PDF generation for every existing fixture and preset. Despite that, images with dense
thin linear detail — ropes, cables, branches, grass blades, hair, decorative borders —
produce pages with far more printable regions than a human colorist can reasonably fill in.

This has been confirmed to be a **geometric** problem, not a chromatic one:

- Lowering palette size (`quantize.n_colors`) does not help — thin features survive
  quantization because they are often a single sharp color run (a black outline, a dark
  branch against sky), not a smoothly-graded area quantization would merge away.
- `merge_tiny` (`stages/graph/merge.py`) already enforces a printability **area** floor
  (`A_min = π·(d_min_mm/2)²·ppmm²`, ENGINE_SPEC §11). A thin, long region — a 40 mm cable
  1 mm wide — can have areas far above `A_min` while remaining physically uncolorable with a
  marker or crayon. Area is the wrong metric for this failure mode; **width** is.
- Nothing in the pipeline today measures or bounds region **width** independent of area.

Sprint 36's mandate is to add a new, dedicated pipeline stage that prunes/simplifies
geometry that is too narrow to print, while leaving color decisions (`quantize`) and area
consolidation (`merge_tiny`) untouched. This is deliberately scoped as *print-readiness*
geometry work, not image simplification, denoising, or stylization.

## Stage position

### Candidates considered

1. **Between `regions` and `merge_tiny`** (chosen).
2. Between `merge_tiny` and `topology`.
3. After `topology`, operating on vector arcs (integer crack coordinates) instead of the
   pixel `component_map`.

### Decision: between `regions` and `merge_tiny`

The prompt's proposed pipeline names a "region graph" stage distinct from "connected
components," but no such stage exists — `regions` (`ConnectedComponentsStage`,
`stages/graph/components.py`) *is* the region-graph-construction stage: it already builds
the `RegionGraph` (regions + `component_map` + adjacency edges) that every later stage
consumes. So the prompt's intended slot — "Region Graph → Print-Aware Geometry
Simplification → Merge Tiny" — maps onto the real pipeline as:

```
regions → geometry_simplify (NEW) → merge_tiny → topology → arcgraph → simplify → bezier
```

Reasons this position is correct, not merely convenient:

- **Everything up through `merge_tiny` is pixel/`component_map`-based; `topology` is the
  first stage to introduce vector geometry** (and even there, only exact-integer doubled
  crack coordinates — floats do not appear until `arcgraph`'s single Φ page-scale
  application). A geometry-simplification stage that edits the *shape* of regions (erasing
  thin peninsulas, snapping thin bridges) belongs in this pixel-array world, where
  raster morphology operations are natural, well-understood, and cheap. Doing this after
  `topology` would mean mutating validated junction/arc structures directly — a much higher
  risk of reintroducing gaps, overlaps, or Euler-identity violations that `topology`'s
  `validate_topology()` (called unconditionally, `stages/vector/topology.py`) would then
  have to re-derive from scratch anyway.
- **It must run before `merge_tiny`, not after.** `merge_tiny` enforces a floor on *area*;
  it has no notion of width and will happily leave a physically unprintable thin shape alone
  if that shape's area already clears `A_min`. If geometry simplification ran after
  `merge_tiny`, its width-pruning could re-fragment or shrink regions merge_tiny had just
  consolidated, potentially dropping some back under the area floor — forcing a second
  merge pass. Running simplification first means merge_tiny's area floor becomes the
  **second** gate (chromatic/area) applied after the **first** gate (geometric/width), which
  composes cleanly: simplification produces a component map with no sub-width features, and
  merge_tiny then may still coarsen small-but-wide leftover regions on top of that.
- **It must run after `regions`, not fold into it**, because it needs the exact same
  `RegionGraph`/`component_map` representation `regions` already produces, and because the
  new stage's own invariants (double-entry boundary identity, deterministic component
  relabeling) are best re-derived by reusing `components.py`'s existing
  `_region_records`/`_adjacency` helpers — exactly the pattern `merge_tiny` already
  establishes (`merge.py:169-170`) for re-deriving records after it mutates the map.

### Consequence for existing invariants

- `merge_tiny`'s area floor is no longer the first geometric gate a region passes through —
  it is the second. This is called out here explicitly (ADR-001 style) rather than left
  implicit. No change to `merge_tiny`'s own code, contract, or config is needed: it still
  requires `("region_graph", "palette", "raster_working")` and still operates on whatever
  `RegionGraph` precedes it in context — it does not know or care that a new stage now sits
  upstream.
- `regions`' own invariants (4-connectivity, double-entry boundary identity, dense
  raster-order ids) must hold on `geometry_simplify`'s **output** component map exactly as
  they hold on its input — the new stage re-derives records/adjacency the same way
  `merge_tiny` does, so this is enforced by construction, not by hope.
- `topology`, `arcgraph`, `simplify`, `bezier` are entirely unaffected: they never see the
  pre-simplification component map, and their own invariant tests
  (`tests/property/test_topology_properties.py`,
  `tests/property/test_arcgraph_properties.py`) require no changes.

## Algorithm

### Candidates considered

| Approach | Pros | Cons |
|---|---|---|
| **Morphological opening + distance-transform pruning** (chosen) | Deterministic, `O(N)` per region via `scipy.ndimage.distance_transform_edt`; structuring-element radius derives directly from `minimum_feature_width_mm` via the same `ppmm` conversion `merge_tiny` already uses; well-understood failure modes (conservative radius ⇒ under-pruning, never new intersections); operates per-label on a binary mask, trivially deterministic and easy to bound. | Naive per-region opening can round corners on legitimate wide shapes if the radius is too large relative to feature scale; needs the distance-transform pass specifically to distinguish "thin protrusion" from "small but chunky region" (opening alone conflates the two only weakly). |
| Skeleton / medial-axis width analysis + graph pruning | Most surgical: prunes exactly the thin skeleton branches below a width threshold while leaving thick trunks untouched, better composition preservation on branching shapes (e.g. tree branches, veins). | Skeletonization (`skimage.morphology.skeletonize` / medial axis) is harder to keep bit-for-bit deterministic across platforms/BLAS versions than a distance transform; reconstructing a clean, injective region map from a pruned skeleton (skeleton → thickened region, re-partitioned to disjoint labels) is a nontrivial extra algorithm with its own invariant risk; higher implementation cost for this sprint. |
| Scale-space / shape-complexity pruning | Most flexible for difficulty scaling (a single complexity score gives a natural knob); can in principle unify silhouette, hole, and thin-feature preservation into one saliency metric. | Least formally boundable of the three; "shape complexity score" is not a single well-defined deterministic algorithm but a family of heuristics, which conflicts directly with this sprint's "no random rules," "think like a commercial publishing engine" mandate. Highest risk, weakest prior art in this codebase. |

### Decision: morphological opening + distance-transform pruning

Per connected region (on `component_map`, one binary mask per region label):

1. Convert `minimum_feature_width_mm` to a working-px radius via the existing
   `ppmm = 1.0 / (work_scale · MM_PER_INCH / PT_PER_INCH)` conversion (`merge.py:62`,
   proposed to be hoisted to `foundation/units.py` as a shared `mm_to_px` helper — see
   Migration Impact) — this reuses the *exact* mm→px pattern `merge_tiny` established rather
   than inventing a second one.
2. Compute the Euclidean distance transform of the region's binary mask
   (`scipy.ndimage.distance_transform_edt`); any pixel whose distance-to-background is below
   `minimum_feature_width_mm / 2` (converted to px) lies in a sub-width feature (a thin
   peninsula, spike, or bridge neck).
3. Binary-open the mask with a disk structuring element of that same radius — this removes
   thin protrusions and narrow bridges while leaving the bulk of wide shapes untouched
   (opening's core property: it removes anything the structuring element cannot fit inside).
4. Reassign pixels removed by opening to the best surviving 4-connected neighbor region
   (nearest by boundary adjacency, ties broken by lower region id — deterministic, mirrors
   `merge_tiny`'s existing tie-break convention) rather than leaving holes: this is what
   keeps the *page area* conserved (every pixel keeps a label) and avoids inventing a "no
   label" pixel value the rest of the pipeline has no representation for.
5. If a region's mask becomes empty after opening (the whole region was a sub-width sliver),
   drop the region and let its pixels be reassigned to neighbors — this is a case that
   `merge_tiny`'s existing degenerate-page handling (R = 1 legal minimum) already covers
   downstream and requires no special-casing here.
6. Re-derive `_region_records`/`_adjacency` from the mutated `component_map` exactly as
   `merge_tiny` does, producing a new `RegionGraph`.

**Silhouette and hole preservation** (`preserve_silhouette`, `preserve_holes` config, see
below) are implemented as a pre-mask: the union of all region masks (the page's outer
silhouette) and enclosed background holes above `minimum_enclosed_hole_mm²` are computed
first and excluded from candidate pixels for reassignment — opening is only ever allowed to
touch pixels *interior* to the silhouette / *outside* enclosed protected holes, never to
erode the outer boundary of the artwork itself. This directly satisfies the sprint's "reduce
thin regions... while preserving silhouette... object recognizability" goal without needing
a separate saliency heuristic.

**Why not skeletonization for v1**: the width-pruning goal here is served just as well by
distance-transform + opening, at materially lower implementation and determinism risk;
skeleton-based branch pruning is recorded as a documented future increment (see Migration
Impact) if opening proves too coarse for high-branching content (grass, hair) in practice.

## Configuration

New, dedicated section — not folded into `quantize`, `merge`, or `quality` — because none of
those sections own width-based, print-oriented thresholds today, and the sprint brief
explicitly calls for a dedicated section:

```python
# app/config_defaults.py — new section, added to PIPELINE_STAGES between "regions" and
# "merge_tiny", and to builtin_defaults()/difficulty_preset() alongside quality/quantize.

"geometry_simplification": {
    "enabled": True,
    "minimum_feature_width_mm": 1.5,   # by preset, see below
    "minimum_bridge_width_mm": 1.0,    # by preset
    "minimum_peninsula_width_mm": 1.2, # by preset
    "minimum_enclosed_hole_mm2": 2.0,
    "preserve_silhouette": True,
    "preserve_holes": True,
}

GEOMETRY_MIN_FEATURE_WIDTH_MM_BY_PRESET = {"easy": 2.5, "medium": 1.5, "hard": 1.0}
GEOMETRY_MIN_BRIDGE_WIDTH_MM_BY_PRESET = {"easy": 1.8, "medium": 1.0, "hard": 0.6}
GEOMETRY_MIN_PENINSULA_WIDTH_MM_BY_PRESET = {"easy": 2.0, "medium": 1.2, "hard": 0.8}
```

All thresholds are in **millimeters**, matching `quality.d_min_mm`, `merge`'s `d_min_mm`,
`simplify`'s `tolerance_mm`, and `bezier`'s `fit_error_mm` — the codebase's established
convention that print-facing geometry parameters are always mm, converted to working px at
the point of use via `work_scale`, never stored or compared in px.

`enabled: False` makes the stage a structural no-op (`RegionGraph` passed through
unmodified, still re-stamped with this stage's `Provenance` for audit-trail consistency) —
this is the escape hatch for A/B benchmarking against the pre-Sprint-36 pipeline without
removing the stage from `PIPELINE_STAGES`.

### Difficulty scaling

Following the existing `D_MIN_MM_BY_PRESET`/`N_COLORS_BY_PRESET` pattern in
`app/config_defaults.py` exactly:

- **Easy**: larger thresholds → remove more thin detail → fewer, chunkier regions, easiest
  to color.
- **Medium**: balanced defaults (1.5 / 1.0 / 1.2 mm) chosen to sit below `quality.d_min_mm`
  medium (3.5 mm) — geometry simplification should not be more aggressive than the area
  floor it precedes; a feature must be *narrower* than what merge_tiny would already
  independently plan to keep as its own colorable disc diameter.
- **Hard**: smaller thresholds → preserve more thin, intricate detail.

Threaded exactly as `difficulty_preset()` already threads `d_min_mm`/`n_colors`: as a
`DIFFICULTY_PRESET`-layer overlay of the `geometry_simplification` section.

## Validation strategy

New validators, added to `src/mysterycbn/validate/` alongside the existing
`topology.py`/`printability.py`/`fidelity.py`/`palette.py`, following the same "independent
re-proof, never silently repair" convention `topology.py` documents:

1. **Minimum printable width** — for every surviving region, distance-transform max ≥
   `minimum_feature_width_mm / 2` somewhere in the region (a region need not be uniformly
   wide, but must contain *some* colorable core) — extends `printability.py`.
2. **Bridge width validation** — no two same-region components connected only through a
   corridor narrower than `minimum_bridge_width_mm` survives (checked via the same distance
   transform used to simplify; a validator re-proof, not a re-run of the simplification
   algorithm).
3. **Geometry complexity score** — count of thin-feature pixels remaining
   pre-/post-simplification, exposed as a `RunReport` metric (not a pass/fail gate) for
   benchmark tracking (see Benchmarks).
4. **Thin-feature statistics** — histogram of region minimum-width, reported for benchmark
   comparison against the golden dataset baseline.
5. **Shape preservation score** — IoU (intersection-over-union) between the pre- and
   post-simplification silhouette masks, must exceed a high fixed threshold (e.g. 0.98) —
   this is the quantitative backstop for "preserving artistic composition" and would fail
   loudly (`StageError`, not a warning) if `preserve_silhouette` is misconfigured or a future
   change accidentally erodes the outer boundary.

These slot into `validate/report.py::run_validation` as a fifth canonical gate alongside the
existing four (fidelity, topology, printability, palette).

## Regression safety

- **Determinism**: `distance_transform_edt` and binary opening with a fixed structuring
  element are both deterministic given a fixed input array and radius (no floating-point
  accumulation order dependency of the kind that would break bit-for-bit reproducibility);
  reassignment tie-breaks are id-ordered exactly like `merge_tiny`'s. A Hypothesis property
  test (`tests/property/test_geometry_simplification_properties.py`, mirroring
  `test_merge_properties.py`'s pattern) asserts calling the stage twice on the same input
  yields identical `component_map` output.
- **No topology corruption / no gaps / no overlaps**: guaranteed by construction, not by
  post-hoc checking, because every pixel keeps exactly one label at all times (opening
  removes a region's *claim* on a pixel, immediately followed by deterministic reassignment
  to a neighbor — never an unlabeled gap) and `_region_records`/`_adjacency` re-derivation
  re-establishes the double-entry boundary identity from scratch, exactly as `merge_tiny`
  already relies on.
- **No self-intersections**: not applicable at this pixel-array stage (self-intersection is
  a vector-geometry concept that only exists from `topology` onward); this stage cannot
  introduce a self-intersection because it never produces vector geometry.
- **No renderer regression**: renderers (`render/svg.py`, `render/pdf.py`,
  `render/png.py`, and the cross-renderer contract test
  `tests/contracts/test_renderer_agreement.py`) consume `CurveSet`, produced downstream of
  `bezier`; they are unaffected by any upstream `RegionGraph` change in shape, only in count,
  which they already handle generically.
- **Existing tests continue to pass**: no existing stage's algorithm, config schema, or
  contract changes. `merge_tiny`, `topology`, etc. are unmodified files. The only touch to
  existing infrastructure is additive: a new entry in `PIPELINE_STAGES`
  (`app/config_defaults.py`) and a new factory registration in
  `app/registry_bootstrap.py::build_stage_factories`.

## Migration impact

- **Additive only.** No existing stage's file changes. `PIPELINE_STAGES` gains one entry;
  `builtin_defaults()`/`difficulty_preset()` gain one section each.
- A shared `mm_to_px`/`px_area_from_mm_diameter`-style helper is proposed to move from being
  duplicated logic (`merge.py::area_floor_px`) into `foundation/units.py`, with `merge.py`
  refactored to call the shared helper instead of keeping its own copy — this is the one
  small, mechanical touch to existing code this sprint's design implies, justified by "don't
  invent a second mm→px conversion" rather than by any behavior change (the formula is
  identical; `area_floor_px` becomes a one-line wrapper preserving its existing public name
  and signature for backward compatibility with `merge.py`'s own tests).
- Golden/benchmark datasets (`benchmarks/golden_store/`) will need one-time re-baselining
  once the stage is implemented and enabled by default, since region counts will change for
  any fixture containing sub-threshold thin features. This is expected and desired — it is
  the metric this sprint is meant to move — but must be called out so benchmark diffs in the
  first implementation PR are not mistaken for regressions.
- Future increment, explicitly deferred: skeleton/medial-axis-based branch pruning as an
  alternate or additional pass for highly-branching content (grass, hair, fur), if
  opening-based pruning proves too coarse in practice on those fixture categories. Tracked
  here rather than attempted now, matching this codebase's convention (see ADR-001 §
  Deviations 3–4) of naming out-of-scope follow-ups explicitly rather than silently omitting
  them.

## Consequences

- The engine gains a fifth canonical validation gate and a sixteenth pipeline stage, both
  purely additive to existing contracts.
- `merge_tiny`'s area floor becomes explicitly the second of two geometric gates, not the
  first — documented here so this is never rediscovered as a surprise.
- A shared mm→px conversion helper reduces future duplication risk between `merge_tiny` and
  the new stage.
- Benchmark golden datasets require one-time re-baselining after implementation.
