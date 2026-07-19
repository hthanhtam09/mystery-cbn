# ADR-003: Organic Region Partition Stage

**Status:** Accepted (implemented).
**Date:** 2026-07-11.
**Supersedes:** none. **Related:** ADR-002 (pixel-vs-vector stage-boundary reasoning this
decision reuses); `stages/graph/split_large.py` (the stage this feature extends the shared
texture machinery from).

## Context

Region *shape* in the engine today is entirely a byproduct of LAB color quantization
(`stages/raster/quantize.py`) followed by 4-connected pixel labeling
(`stages/graph/components.py`). Boundaries trace pixel-grid axes and only receive light
Bézier smoothing downstream (`stages/vector/curves.py`), so they read as blocky/staircased
rather than hand-drafted.

A request surfaced for an alternative partition style: organic topology, flowing spline
boundaries, nested islands, branching/ribbon-like cells, variable density, no repeated
tiling, no sharp corners — closer to a hand-drafted color-by-number sheet than a
computer-quantized one.

`stages/graph/split_large.py` (ENGINE_SPEC §12) already solves an adjacent problem —
breaking oversized flat regions into many smaller same-color cells — using hand-rolled
value-noise, domain-warp, and curl-noise streamline machinery (no external noise/Voronoi
library; the project deliberately avoids that dependency, `noise`/`opensimplex` are absent
from `pyproject.toml`, and `scipy.spatial.Voronoi`/`KDTree` are unused everywhere in favor of
a distance-transform-based nearest-seed assignment). This machinery is the right building
block: organic partitioning is a variation on "subdivide a flat region into many cells," not
a new geometric primitive.

## Decision

### Stage position: new stage, own pipeline slot, between `merge_tiny` and `split_large`

```
regions → merge_tiny → organic_partition (NEW) → split_large → topology → arcgraph → ...
```

**Candidates considered:**

1. An `impl` config switch inside `ConnectedComponentsStage` (`stages/graph/components.py`).
   Rejected: `regions` runs immediately on the raw `LabelMap`, before `merge_tiny` exists —
   organic partitioning needs the *opposite* input shape (a clean, already-merged
   `RegionGraph`), not the raw post-quantization label map. `ConnectedComponentsStage` also
   explicitly documents "no configuration parameters; connectivity is an invariant, not a
   knob" (ENGINE_SPEC §9) — folding a second algorithm in would break that contract.
2. The stage registry's native `(slot, impl)` multi-registration mechanism
   (`kernel/registry.py`'s `InMemoryStageRegistry`). Architecturally present
   (`ARCHITECTURE.md §8`: "selection is by configuration, never by import") but unused by
   every existing slot — every stage registers under `impl_name="default"` only. Adopting it
   here would be the first live use of an unexercised code path, a larger platform bet than
   this feature needs.
3. **New stage, own slot** (chosen), config-gated exactly like `merge_tiny`/`split_large`
   (`enabled: bool`, default `False` → identity passthrough).

**Why between `merge_tiny` and `split_large` specifically:**

- Must run **after `merge_tiny`**: needs clean, consolidated same-color masks to redraw with
  organic boundaries, not pre-merge fragments it would have to fight.
- Must run **before `split_large`**: organic cells are already reasonably sized by
  construction; `split_large` still acts as a safety net for any pathologically large
  leftover cell (its `area > split_factor * a_min` gate is unaffected either way), rather
  than immediately overwriting organic shapes with its own Voronoi splitting if it ran
  first.
- Both stages operate purely on `RegionGraph` + `component_map` in the pixel domain —
  consistent with ADR-002's finding that "everything up through `merge_tiny` is
  pixel/`component_map`-based; `topology` is the first stage to introduce vector geometry."
  `organic_partition` must finish before `topology` for the same reason: mutating
  post-topology vector structures directly is a much higher-risk operation than raster
  mutation, and `topology`'s junction/crack-arc extraction is a pixel-perfect 2×2-neighborhood
  algorithm that requires a dense int32 `component_map` regardless of how regions were
  seeded — there is no way to bypass this contract without also redesigning
  `stages/vector/topology.py`/`arcgraph.py` (out of scope).
- `organic_partition` is classified as a `GraphStage` (`stages/base.py`: "Operates on
  RegionGraph/LabelMap; the label raster remains authoritative geometry") — the same
  classification as `ConnectedComponentsStage`/`MergeTinyStage`/`SplitLargeStage`.

### Algorithm

Per eligible region (`area_px >= min_area_mm2`, converted to working px² via `work_scale`):

1. **Rim/core split**: erode by `rim_mm` (distance-transform-based, matching
   `split_large`'s existing rim-protection reasoning) so an un-partitioned rim strip always
   traces the region's real silhouette — an organic cell must never eat into a subject's own
   detail boundary (face, ear, fur outline).
2. **Seed placement**: target cell count from `seed_density_mm2`; seeded via
   `farthest_point_seeds`/`grid_seeds` (deterministic, shared with `split_large`).
3. **Cell-shape generation**, selectable via `mode`:
   - `"voronoi"`: domain-warped nearest-seed assignment (`voronoi_labels` + `flow_field`) —
     flowing-but-compact cells.
   - `"streamline"` (default): curl-noise stroke tracing (`streamline_labels`/
     `trace_streamlines`) whose enclosed pockets are free-form, non-repeating, and
     ribbon-capable — the more organic of the two mechanisms, and the primary mode for
     meeting the "flowing spline boundaries"/"branching ribbon cells" requirement.
     `ribbon_elongation` biases stroke length up / width down to further favor thin,
     branching pockets.
   - `"mixed"`: streamline first, then a Voronoi overlay re-splits any pocket still oversized.
4. **Nested islands** (deliberately engineered, not an accidental byproduct — neither Voronoi
   nor streamline pockets nest one enclosure inside another by default): for a configurable
   fraction (`island_probability`) of eligible interior cells, a small secondary seed cluster
   is grown within that cell's own mask, producing a genuinely enclosed sub-cell as its own
   `Region` id. Carving happens in a single pass over each region's pre-island cell ids —
   an island can never itself contain a sub-island.
5. **Compact + re-densify**: force single-4-connected-component per id, fold any sub-floor
   cell into a same-label neighbor — reusing `split_large`'s exact machinery.
6. **Label inheritance**: every organic cell — and every island — keeps its parent region's
   palette index unchanged. This stage only ever subdivides *shape* within one color; it
   never reassigns or crosses palette labels, preserving color-by-number semantics (one
   number = one palette color).

### Code sharing: extracted to `stages/graph/_organic_common.py`

`split_large.py`'s private noise/warp/Voronoi/streamline/fold/rebuild helpers were promoted
verbatim into a new shared module, and `split_large.py` refactored to import from it —
avoiding one stage reaching into another's private (`_`-prefixed) internals, and giving both
stages one implementation to maintain. This is a pure extraction: `split_large`'s behavior
and existing outputs are unchanged (verified by the full existing test suite passing
unmodified after the refactor).

### Default config values

Initial defaults (small `seed_density_mm2`, `ribbon_elongation`/`island_probability` at 0)
produced small, confetti-like cells rather than the requested large, sparse "liquid blob"
look. Defaults were re-tuned against a rendered preview
(`seed_density_mm2=400.0`, `warp_strength_mm=10.0`, `noise_scale_mm=30.0`,
`ribbon_elongation=0.7`, `island_probability=0.6`) to visually match the reference style
before landing — `SEED_DENSITY_MM2_MAX`/`NOISE_SCALE_MM_MAX` were widened accordingly to give
headroom above the new defaults. This tuning only affects `organic_partition.py`'s own
constants (still disabled by default at the pipeline level), so it has no effect on any
existing preset's output.

A subsequent real-world test (a rabbit photo through the web UI) surfaced a visible "double
outline" artifact around the subject's silhouette at the then-default `rim_mm=1.5`: the
eroded rim becomes its own `Region`, which introduces a *second* boundary (rim↔core) running
immediately parallel to the region's real silhouette edge (rim↔background) — invisible on
`split_large`'s tiny filler cells deep in a busy background, but glaring on
`organic_partition`'s large cells right at a subject's outline. Fixed by changing
`RIM_MM_DEFAULT` to `0.0` (verified via before/after preview render: single outline
restored, organic cells now reach the true silhouette edge). Callers who want the rim's
silhouette-protection behavior back can still opt in via `rim_mm`.

Two more real-world artifacts surfaced testing the same rabbit-shaped fixture through the
web UI (which always uses the `"dense"` preset):

- **Background shattering**: `organic_partition` was applying to *every* eligible region,
  including the page background — a large flat backdrop got shattered into hundreds of small
  organic cells (visual noise), and the many new region boundaries running close beside a
  subject's real silhouette edge read as a doubled outline even with `rim_mm=0`. Fixed by
  adding `_background_region_id()` (the single region touching the page border with the
  largest border-contact length, mirroring `components.py`'s own border-length concept) and
  a new `skip_background` config (default `True`) that leaves it untouched regardless of
  size.
- **`split_large` double-rim conflict**: the `"dense"` preset enables `split_large` in
  addition to `organic_partition`. `split_large` has its own independent `rim_mm` default
  (`2.0`, unrelated to `organic_partition`'s own now-`0.0` default) — when both stages run,
  `split_large` sees the large background `organic_partition` deliberately left untouched,
  Voronoi-splits it, and wraps it in its *own* rim right next to the subject's silhouette,
  reproducing the doubled-outline artifact one layer downstream. Fixed at the caller level:
  `mystery-cbn-web`'s fixed conversion overrides now include `split: { enabled: false }`
  whenever `organic.enabled` is `true`, since the two stages do the same job (subdivide flat
  areas into cells) and running both is redundant even without the rim conflict.
- **Ribbon-cell near-parallel edges**: with `ribbon_elongation` at its then-default `0.7`, an
  elongated organic cell can end up hugging the inside of a subject's real silhouette (e.g.
  running the length of an ear) — its near-parallel inner edge is a genuine organic cell
  boundary, not a duplicated silhouette, but reads as one. Fixed by changing
  `RIBBON_ELONGATION_DEFAULT` to `0.0`; callers who want the more branching/vein-like look
  can still opt in.

All three fixes were verified via the same before/after rendered-preview method on a
rabbit-shaped synthetic fixture (round head, two long ears, body) through the full
`mysterycbn.app.api.convert()` pipeline at the `"dense"` preset, matching the real web-UI
conversion path.

A fourth, distinct doubled-outline source surfaced only when testing against the user's
**actual** source photo (a hand-drawn-style cartoon rabbit) rather than a synthetic fixture —
the earlier synthetic fixtures could not reproduce it because they used flat, unbroken
silhouettes with no drawn outline stroke of their own:

- **Pre-drawn cartoon outline quantizes into its own ring-shaped region**: a real cartoon
  image's own ink/line-art outline has physical width, not zero width, so `quantize` puts it
  into its own near-black palette color and it becomes a thin, ring-shaped region enclosing
  the subject (its two edges touch the background on one side and the subject fill on the
  other). This region is large enough to clear `min_area_mm2` in a typical photo. The first
  fix attempt was `SKIP_DARK_LAB_L_THRESHOLD` (15.0, LAB L*) as a *skip* — leaving the ring as
  its own untouched region, exempt from partitioning like the background. **This did not
  work**: a ring's two edges are real geometry regardless of what happens inside the ring
  itself, so organic-partitioning the ring's *neighboring* regions still traced both of the
  ring's edges as real silhouette boundaries — the doubled outline persisted unchanged,
  confirmed via rendered preview against the real photo.
- **Fix: fold, not skip**: the correct fix folds the dark ring region into an adjacent region
  *before* partitioning runs, using a new generalized `fold_regions_where(component_map,
  labels, should_fold)` in `_organic_common.py` (`fold_subfloor_regions` is now one line,
  `should_fold=lambda areas, labels: areas < a_min`, specializing the same shared mechanics).
  This removes the ring as a distinct shape entirely, so only one silhouette edge remains.
  **Implementation pitfall hit and fixed during this change**: the first version of
  `fold_regions_where`'s `should_fold` callback received only the current pass's `areas`
  array, not the current pass's `labels` list — a caller wanting to fold by *label* (like the
  dark-color predicate) had no choice but to close over the *original*, pre-fold label array
  and index into it by region id. Since `_fold_regions_where_once`'s internal fold loop
  renumbers every region id on every pass, that stale label array was silently misaligned
  from the second pass onward, folding arbitrary wrong regions together — this corrupted the
  rabbit's shape entirely (ears and body vanished) in local testing and failed 3 existing
  tests, including one asserting the page background stays a single region. Fixed by changing
  `should_fold`'s signature to `Callable[[np.ndarray, list[int]], np.ndarray]` — it now
  receives that pass's freshly-recomputed `labels` alongside `areas`, so a label-based
  predicate is always evaluated against current, correctly-indexed ids. Caught before landing
  by re-running the full test suite immediately after the change (not just the target
  scenario) — the pre-existing property tests were sufficient to catch the corruption despite
  none of them being specifically about dark-region folding.
- Two of the three existing property-test fixtures (`PAL4`) originally used LAB L*=10.0 for
  their background color, which is below `SKIP_DARK_LAB_L_THRESHOLD` — an unrelated
  coincidence that made the background itself look like an "outline" region and get folded
  away, breaking those tests' own assertions. Fixed by raising the fixture's darkest L* to
  30.0 (still visually distinct, safely above the threshold) and adding a dedicated
  `test_organic_partition_folds_dark_outline_region` test with its own fixture built
  specifically to exercise the fold path (a thin near-black strip along one edge of a larger
  subject region).

### Filler-cell printability exemption

`split_large`'s filler cells are exempt from the printability readable-size floor
(`validate/printability.py`'s `filler_region_ids` contract) because they are deliberately
small by construction. Organic-partitioned cells share that same construction, so
`OrganicPartitionStage` produces the identical `filler_region_ids`/`render_filler_region_ids`
contract (`provides`: `region_graph`, `filler_region_ids`, `render_filler_region_ids`).

Because `organic_partition` now runs *before* `split_large` in the pipeline,
`SplitLargeStage` was changed to **merge into** (not overwrite) any incoming
`filler_region_ids`/`render_filler_region_ids` from context — including when `split_large`
itself is disabled, which previously always stamped empty sets regardless of upstream state.
Filler/rim status is threaded through by per-pixel mask, not by region id, since ids are
renumbered by each stage's own component-map rebuild.

### RNG

`OrganicPartitionStage` derives its own `stage_seed()` (`SHA-256(seed‖stage_name)[:8]`,
matching `quantize.py`'s convention) rather than reusing `ctx.seed` directly like
`split_large` does — this stage's RNG surface is larger (seed placement, warp, streamline
tracing, and island sub-seeding all need independent, well-separated random streams).

## Consequence for existing invariants

- **`RegionGraph`/`Region` dataclass invariants**: unaffected by construction — organic
  partitioning re-derives region records/adjacency/edges via the same `_region_records`/
  `_adjacency` closed-form math every graph stage already uses
  (`_organic_common.rebuild_region_graph`), so every existing `__post_init__` check
  (dense `component_map` ids, bbox/seed/centroid containment, `area_px >= 1`,
  `perimeter_px >= 4`) holds automatically.
- **Determinism**: preserved via the per-stage `stage_seed()` convention; a property test
  (`tests/property/test_organic_partition_properties.py`) asserts byte-identical
  `component_map`/region/edge output across repeated calls with the same seed.
- **Topology's pixel-perfect crack-coordinate algorithm**: untouched — `organic_partition`
  still emits a dense int32 `component_map` before `topology` runs, satisfying its
  requirement unchanged.
- **`fold_subfloor_regions` fixpoint bug fix**: a property test for this stage caught a
  pre-existing, unrelated bug in the (now-shared) fold logic — a single fold pass can leave a
  chain of tiny same-label regions, folded only into each other with no larger neighbor
  available, still below the area floor afterward, violating `split_large.py`'s own
  documented invariant ("no sub-floor region is introduced"). Fixed by looping the fold pass
  to a fixpoint (capped at 8 passes) in `_organic_common.fold_subfloor_regions`; this also
  fixes the latent bug for `split_large`'s existing callers, verified via the full existing
  test suite passing unmodified.
- **Existing tests continue to pass**: no existing stage's algorithm, config schema, or
  contract changes beyond the `filler_region_ids` merge fix above (itself covered by the full
  suite passing). `PIPELINE_STAGES` gains one entry (`organic_partition`); `builtin_defaults()`
  gains one config section (`"organic"`, default `{}` → `enabled: False`). No built-in preset
  (including `"dense"`) enables it — opt-in only, so no existing golden fixture output
  changes.

## Verification

```
$ .venv/bin/python -m pytest tests/ -q
391 passed
```

- `tests/property/test_organic_partition_properties.py`: boundary double-entry identity, no
  sub-floor-area regions post-fold, label inheritance (including islands), single-4-connected-
  component-per-id, determinism, and a true-passthrough check for the `min_area_px` gate.
- `tests/golden/test_organic_partition_golden.py`: fixture-driven exact-output regression on
  a small synthetic label map under a fixed seed.
- Full-pipeline smoke test via `mysterycbn.app.api.convert(...)` with
  `overrides={"organic": {"enabled": True, "mode": ...}}` for all three modes
  (`voronoi`/`streamline`/`mixed`), confirming: output differs from the disabled baseline,
  output is deterministic across repeated calls with the same seed, and the printability
  validation gate passes (after the filler-exemption fix above — before it, all three modes
  failed the gate on cells organic partitioning produces at typical density).
- `organic.enabled=False` (default) full-pipeline and golden-test output is unchanged from
  pre-feature baseline — verified via the unmodified existing test suite passing.

## Migration impact

- **Additive only** apart from the two small, behavior-preserving fixes above
  (`fold_subfloor_regions` fixpoint loop; `SplitLargeStage`'s filler-id merge-not-overwrite).
  No existing stage's file changes besides `split_large.py` (pure extraction refactor +
  the filler-merge fix) and `orchestrator_impl.py` (one missing config-section name added
  to the list of sections threaded from resolved config to stage factories — `"organic"` was
  omitted initially and is also the mechanism `split.enabled`/`merge.enabled` overrides
  already rely on).
- No golden/benchmark re-baselining required: the new stage is disabled by default
  everywhere, so no existing fixture's output changes.

## Consequences

- The engine gains an eighteenth pipeline stage and a fourth documented ADR, purely additive
  to existing contracts when disabled (the default).
- `split_large`'s filler-cell exemption becomes a shared two-producer contract
  (`organic_partition` and `split_large` can both mark cells filler, and either or both may
  run in a given configuration) rather than a single-stage concept — documented here so this
  is never rediscovered as a surprise.
- `fold_subfloor_regions`'s fixpoint behavior is now guaranteed (bounded at 8 passes) for
  both current callers (`split_large`, `organic_partition`) and any future caller.
