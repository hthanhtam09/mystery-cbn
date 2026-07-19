# Module Design — Print-Aware Geometry Simplification (`stages/graph/geometry_simplification`)

**Status:** v0.1 — design proposed, Sprint 36 (see [ADR-002](../adr/002-sprint36-print-aware-geometry-simplification.md)). Not yet implemented.

## Purpose

Reduce region **width**, not area, to a physically printable minimum before `merge_tiny`
runs. Solves a distinct failure mode from `merge_tiny`'s area floor: a long thin region
(cable, branch, hair strand) can exceed `A_min` while remaining too narrow to color with a
marker or crayon. Input `RegionGraph` (from `regions`); output a new `RegionGraph` with
sub-width protrusions/bridges pruned and reassigned to neighbors, no pixel left unlabeled.

Pipeline position: `regions → geometry_simplify → merge_tiny → topology → ...` — the first
of two geometric gates a region passes through before vectorization; see ADR-002 for why
this stage must run pixel-side, before `merge_tiny`.

## Algorithm — distance-transform-guided morphological opening

Operates per region label on `component_map` (pixel domain only — no vector geometry
exists yet at this pipeline position):

1. **mm → px** — convert `minimum_feature_width_mm` (and bridge/peninsula variants) to
   working px via the shared `ppmm` conversion (`foundation/units`, same formula
   `merge.py::area_floor_px` already uses for its area floor).
2. **Silhouette + hole protection** — compute the union mask of all regions (outer
   silhouette) and enclosed background holes ≥ `minimum_enclosed_hole_mm2`; both are excluded
   from candidate pixels for removal when `preserve_silhouette`/`preserve_holes` are set
   (default both `True`). Opening only ever touches pixels interior to the artwork's own
   boundary.
3. **Distance transform** — `scipy.ndimage.distance_transform_edt` on each region's binary
   mask; pixels with distance-to-background below half the relevant mm threshold lie in a
   sub-width feature (thin peninsula, spike, or bridge neck).
4. **Morphological opening** — binary opening with a disk structuring element sized to the
   same radius removes anything the disk cannot fit inside, leaving wide shapes untouched by
   construction (opening never adds pixels, only removes ones too thin for the element).
5. **Reassignment** — every pixel removed by opening is deterministically reassigned to the
   best surviving 4-connected neighbor region (nearest by adjacency, ties → lower region id,
   mirroring `merge_tiny`'s own tie-break convention). No pixel is ever left unlabeled — page
   area is conserved by construction.
6. **Empty-region drop** — a region entirely consumed by opening (a pure sliver) disappears;
   its former pixels were already reassigned in step 5. No special-casing needed —
   `merge_tiny`'s existing R = 1 degenerate-page handling downstream already covers the
   limit case.
7. **Re-derivation** — `_region_records`/`_adjacency` (`stages/graph/components.py`) rebuild
   records and adjacency from the mutated `component_map`, exactly as `merge_tiny` does,
   re-establishing the double-entry boundary identity structurally rather than by patching it.

## Rejected alternatives

- **Skeleton/medial-axis branch pruning** — more surgical (prunes exactly thin skeleton
  branches, leaves trunks untouched), but skeletonization is harder to keep bit-for-bit
  deterministic across platforms, and reconstructing a clean disjoint region partition from
  a pruned skeleton is a second nontrivial algorithm with its own invariant risk. Recorded
  as a future increment for high-branching content (grass, hair, fur) if opening proves too
  coarse in practice — see ADR-002 Migration Impact.
- **Scale-space / shape-complexity saliency pruning** — most flexible for difficulty
  scaling, but not a single well-defined deterministic algorithm; conflicts with "no random
  rules" and this sprint's determinism mandate.
- **Area-based pruning (reuse `merge_tiny`)** — rejected outright: area and width are
  different quantities; a long thin region can have arbitrarily large area while being
  arbitrarily narrow. This was the original limitation motivating this sprint.

## Quality requirements (proposed)

- **Minimum printable width (Gate):** every surviving region contains at least one pixel
  with distance-transform value ≥ `minimum_feature_width_mm / 2` — property-tested and
  checked by a new validator (`validate/printability.py` extension).
- **Bridge width (Gate):** no surviving region contains a corridor narrower than
  `minimum_bridge_width_mm` connecting two wider lobes — re-proved independently from the
  simplification algorithm via a fresh distance-transform pass in the validator, not by
  re-running the stage.
- **Shape preservation (Gate):** IoU between pre- and post-simplification silhouette masks
  ≥ 0.98 — quantitative backstop for "preserving artistic composition."
- **Determinism:** identical `component_map` output across repeated runs on the same input
  (Hypothesis property test, mirrors `tests/property/test_merge_properties.py`).
- **Area conservation:** `Σ area_px` before and after equals the page pixel count (every
  pixel keeps exactly one label throughout).

## Artifacts

Requires `region_graph`, `raster_working` (work scale for mm→px). Provides `region_graph`
(new generation, stage `geometry_simplify` v0.1.0). Config section
`geometry_simplification` (`enabled`, `minimum_feature_width_mm`,
`minimum_bridge_width_mm`, `minimum_peninsula_width_mm`, `minimum_enclosed_hole_mm2`,
`preserve_silhouette`, `preserve_holes`), difficulty-scaled via
`GEOMETRY_MIN_FEATURE_WIDTH_MM_BY_PRESET` and siblings in `app/config_defaults.py`.
`enabled: False` makes the stage an identity pass-through (re-stamped provenance only).

## Future

Skeleton/medial-axis branch pruning for high-branching organic content, if opening-based
pruning proves too coarse in practice on grass/hair/fur fixture categories (see ADR-002).
