# Mystery Color-by-Number Engine — Data Model Specification

**Status:** v1.0 — authoritative definition of every object the engine passes between stages or emits. Companion to [ARCHITECTURE.md](ARCHITECTURE.md) §4 (artifact chain), [ENGINE_SPEC.md](ENGINE_SPEC.md) (producing/consuming stages), [MATH_SPEC.md](MATH_SPEC.md) (formula definitions), [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) §11 (report schema).
**Rule:** the data model *is* the inter-stage contract (ARCHITECTURE.md §1.1). Any field change is a semver event: additive optional fields → minor; anything else → major, with an ADR.

---

## 1. Global rules (apply to every object unless overridden)

- **Immutability.** Every artifact is frozen at construction. Stages replace artifacts on the context; they never mutate (enables caching, diffing, snapshots — ARCHITECTURE.md §4.1). Arrays are read-only views (`writeable=False` semantics); handing out a mutable buffer is a contract violation.
- **Provenance.** Every artifact carries a `provenance` record (§2). Constructed exactly once, by the producing stage.
- **Thread safety.** Immutability makes all artifacts safely shareable across threads without locks. The only mutable objects in the system — `PipelineContext` bindings, heaps and hashes *inside* a stage — are confined to a single thread by the sequential-pipeline rule (ARCHITECTURE.md §13.5); stage-internal parallelism must treat artifacts as read-only inputs and produce results via thread-confined buffers frozen before publication.
- **Serialization.** Two regimes: **debug snapshots** (documented below per object; used by diagnostics bundles and golden tooling) and **outputs** (SVG/PDF/PNG bytes — the only externally promised formats). Internal serialization formats are *not* semver-governed; the schemas of `OutputBundle`, `ValidationReport`, and `BenchmarkReport` are.
- **Identity and ordering.** All ids (`region_id`, `arc_id`, `face_id`, palette index) are dense, 0-based, deterministic (assignment rules per ENGINE_SPEC), and stable for a given (input, config, seed). Every collection field is stored sorted by id; iteration order is therefore deterministic (I2).
- **Units.** Stated per field. The raster/graph objects use working px; vector/layout objects use pt; user-facing config uses mm (MATH_SPEC §2).
- **Numeric types.** `f32` = IEEE-754 binary32, `f64` = binary64, `i32`/`i64` = signed integers, `u8` = byte. Array shapes in NumPy notation.
- **Validation.** Each object's rules are checked at construction (fail = `StageError` naming the producing stage). Cross-object invariants (I1–I4) belong to the validation subsystem, not to constructors.

## 2. Provenance (embedded record)

**Purpose.** Trace every artifact to what produced it (reproducibility, ARCHITECTURE.md §7).

| Field | Type | Units | Description |
|---|---|---|---|
| `stage_name` | str | — | producing stage (registry name) |
| `stage_version` | str | — | semver of the stage implementation |
| `config_hash` | str (hex64) | — | hash of the fully resolved config |
| `source_hash` | str (hex64) | — | SHA-256 of the original input bytes |

**Relationships.** Embedded by value in every artifact. **Ownership.** The producing stage constructs it; nothing else writes it. **Lifecycle.** Created with the artifact, dies with it. **Serialization.** JSON object in every debug snapshot and in the run report. **Memory layout.** four short strings; negligible. **Validation.** all fields non-empty; hashes are 64 lowercase hex chars. **Immutability/Thread safety.** global rules.

---

## 3. RasterImage

**Purpose.** Canonical image representation for the raster domain (ARCHITECTURE.md §4.1; produced by Raster Load, replaced by Preprocessing).

| Field | Type | Units | Description |
|---|---|---|---|
| `pixels` | f32[H, W, 3] | sRGB [0,1] | y-down, row-major |
| `work_scale` | f64 | pt/px | 0.0 until Preprocessing sets the working resolution; > 0 afterwards |
| `resize_factor` | f64 | — | working px per source px (≤ 1.0; 1.0 = no resize) |
| `icc_applied` | bool | — | true if an embedded profile was converted |
| `exif_orientation` | i32 | — | original EXIF tag value 1–8 (1 = none) |
| `provenance` | Provenance | — | §2 |

**Relationships.** Consumed by Preprocessing, Color Analysis, Quantization, Edge Snap; referenced (read-only) by Large Region Split. **Ownership.** Pipeline context owns the binding; the array buffer is owned by the artifact. **Lifecycle.** `RasterImage` (source-resolution) is replaced by `RasterImage'` (working) after Preprocessing; the source-resolution instance is dropped from the context then (peak-memory control). **Serialization.** debug snapshot = 16-bit PNG + JSON sidecar. **Memory layout.** contiguous C-order f32; 12 bytes/px (2 MP working raster ≈ 29 MiB; the dominant raster-domain allocation). **Validation.** finite values in [0,1]; H,W ≥ 64; shape rank 3, last dim 3. **Immutability/Thread safety.** global rules.

## 4. PaletteColor

**Purpose.** One palette entry — the authoritative color record.

| Field | Type | Units | Description |
|---|---|---|---|
| `index` | i32 | — | dense palette index (coverage-ordered at creation, ENGINE_SPEC §7.7) |
| `lab` | f64[3] | CIELAB | authoritative value |
| `srgb` | f64[3] | sRGB [0,1] | derived from `lab` (MATH_SPEC §3), gamut-clamped |
| `coverage_px` | i64 | px | pixels assigned at quantization time |

**Relationships.** Composed by `Palette`; referenced by index from `LabelMap`, `Region`, `Legend`. **Ownership.** `Palette` owns its colors. **Lifecycle.** Created by Quantization; entries dropped/renumbered only by Tiny Region Merge's palette compaction (which produces a *new* Palette). **Serialization.** JSON `{index, lab, srgb_hex, coverage_px}`. **Memory layout.** plain record, ~64 B. **Validation.** `srgb` = clamped conversion of `lab` to 1e-9; `coverage_px ≥ 0`. **Immutability/Thread safety.** global rules.

## 5. Palette

**Purpose.** The ordered color set of the page.

| Field | Type | Units | Description |
|---|---|---|---|
| `colors` | PaletteColor[K] | — | sorted by `index`, dense 0…K−1 |
| `provenance` | Provenance | — | §2 |

Derived (computed once, cached in the object): `delta_e_table: f64[K, K]` — pairwise ΔE00 (MATH_SPEC §4.2).

**Relationships.** Consumed by Noise Removal, Region Graph, Merge, Palette Optimization, Legend, PNG Preview, validators. **Ownership.** context binding. **Lifecycle.** v1 from Quantization; v2 (compacted, with `renumber_map` recorded in provenance notes) from Tiny Region Merge; immutable thereafter. **Serialization.** JSON array of §4 records. **Memory layout.** K ≤ 64 → ≤ 4 KiB + 32 KiB table. **Validation.** dense indices; K ∈ [2, 64]; `min offdiag ΔE00 ≥ merge_delta_e` (constructor re-check of ENGINE_SPEC §7.6). **Immutability/Thread safety.** global rules.

## 6. LabelMap

**Purpose.** Per-pixel palette assignment — authoritative raster geometry until the vector crossing.

| Field | Type | Units | Description |
|---|---|---|---|
| `labels` | i32[H, W] | palette idx | values ∈ [0, K) |
| `provenance` | Provenance | — | §2 |

**Relationships.** Pairs with the `Palette` of the same generation (matched via `config_hash` + producing stage); consumed by Noise Removal, Connected Components, Edge Snap, Contour Extraction, fidelity validator. **Ownership.** context binding. **Lifecycle.** v1 Quantization → v2 Noise Removal → (v3 Edge Snap, optional); superseded for geometry purposes by the component map after §9 but retained until validation (I1 audit needs it). **Serialization.** debug snapshot = indexed PNG (palette-mapped). **Memory layout.** contiguous i32, 4 B/px (2 MP ≈ 7.6 MiB). **Validation.** all values < K; shape = working raster shape. **Immutability/Thread safety.** global rules.

## 7. Region

**Purpose.** One 4-connected maximal pixel set of a single label (a RAG node; ENGINE_SPEC §9).

| Field | Type | Units | Description |
|---|---|---|---|
| `region_id` | i32 | — | dense, raster-scan first-occurrence order |
| `label` | i32 | palette idx | the region's color |
| `area_px` | i64 | px² | pixel count |
| `bbox` | i32[4] | px | (row_min, col_min, row_max, col_max), inclusive |
| `seed_px` | i32[2] | px | first pixel in raster order (reconstruction anchor) |
| `perimeter_px` | i64 | px | crack-edge count incl. page border (MATH_SPEC §5.1) |

**Relationships.** Node of `RegionGraph`; corresponds 1:1 to a `Face` after §15 (via `region_id`). **Ownership.** `RegionGraph` owns its regions. **Lifecycle.** created by Connected Components; merge/split stages produce new `RegionGraph`s with re-derived records (ids of surviving regions are preserved; split products get fresh ids appended). **Serialization.** JSON record. **Memory layout.** ~64 B fixed record; stored in structure-of-arrays form inside RegionGraph for cache-friendly sweeps. **Validation.** `area_px ≥ 1`; bbox within raster; `seed_px` inside bbox. **Immutability/Thread safety.** global rules.

## 8. RegionGraph

**Purpose.** Region adjacency graph — the substrate of merge/split decisions (MATH_SPEC §5.1).

| Field | Type | Units | Description |
|---|---|---|---|
| `regions` | Region[R] | — | sorted by id, dense |
| `component_map` | i32[H, W] | region id | per-pixel region id (authoritative geometry in the graph domain) |
| `edges` | (i32, i32, f64, i64)[E] | —, —, ΔE00, px | (a, b, w_col, w_len), a < b, sorted lexicographically |
| `provenance` | Provenance | — | §2 |

**Relationships.** Pairs with the same-generation `Palette`; consumed by Tiny Region Merge, Large Region Split, Label Placement (region↔label mapping), fidelity validator. **Ownership.** owns `regions` and `component_map`. **Lifecycle.** v1 Region Graph stage → v2 Merge → (v3 Split); frozen thereafter. **Serialization.** debug snapshot = component map PNG + JSON graph dump. **Memory layout.** component map dominates (4 B/px); edges as parallel arrays (SoA). **Validation.** edge endpoints exist; `Σ w_len + Σ border = B` (double-entry identity, MATH_SPEC §5.1); component map values dense in [0, R). **Immutability/Thread safety.** global rules.

## 9. TopologyGraph

**Purpose.** Intermediate junction/arc decomposition of the crack boundary (ENGINE_SPEC §14) — exists between Contour Extraction and Arc Graph assembly; not visible to any other stage.

| Field | Type | Units | Description |
|---|---|---|---|
| `junctions` | i32[V, 2] | doubled crack px | corner coordinates in doubled-integer frame (MATH_SPEC §1.2) |
| `arcs` | Arc[A] | — | §10, polylines in doubled crack px |
| `provenance` | Provenance | — | §2 |

**Relationships.** Consumed solely by Arc Graph assembly. **Ownership.** owns junctions and arcs. **Lifecycle.** shortest-lived artifact; dropped from context once `ArcGraph` exists. **Serialization.** debug JSON dump (the §25.2 diagnostics bundle format). **Memory layout.** arc polylines dominate: i32 pairs, ~8 B/vertex, ≈ B vertices total. **Validation.** every arc's endpoints ∈ junctions (or arc closed); per-arc (left, right) constant; Σ arc lengths = B. **Immutability/Thread safety.** global rules.

## 10. Arc

**Purpose.** One maximal boundary piece separating exactly two regions — the shared-boundary primitive that makes watertightness structural (MATH_SPEC §6.1).

| Field | Type | Units | Description |
|---|---|---|---|
| `arc_id` | i32 | — | dense; order per ENGINE_SPEC §14.4 |
| `points` | i32[P, 2] → f64[P, 2] | doubled crack px → pt | polyline; integer in TopologyGraph, f64 pt in ArcGraph (post-Φ) |
| `left_region` / `right_region` | i32 | region id | −1 = page exterior |
| `closed` | bool | — | island boundary (anchor = points[0] = points[−1]) |

**Relationships.** Owned by TopologyGraph then ArcGraph; referenced by `Face.arc_walk` (id + direction flag); 1:1 with a `Curve` in CurveSet. **Ownership.** containing graph. **Lifecycle.** geometry refined in place across §16–17 *by replacement of the containing ArcGraph* (each stage emits a new ArcGraph; arc ids and (left,right) never change after creation). **Serialization.** JSON `{arc_id, left, right, closed, points}`. **Memory layout.** contiguous coordinate array per arc; arcs stored in one pooled buffer with (offset, length) indices to avoid per-arc allocation. **Validation.** P ≥ 2 (≥ 4 if closed); `left ≠ right`; consecutive points distinct. **Immutability/Thread safety.** global rules.

## 11. ArcGraph

**Purpose.** The planar map in physical units: arcs + faces + correspondence to regions (ENGINE_SPEC §15). After this artifact exists, no stage may consult the raster.

| Field | Type | Units | Description |
|---|---|---|---|
| `arcs` | Arc[A] | pt | §10, coordinates mapped by Φ |
| `faces` | Face[F] | — | see below |
| `work_scale` | f64 | pt/px | the applied `s` (provenance of the single scaling) |
| `provenance` | Provenance | — | §2 |

**Face** (embedded record):

| Field | Type | Units | Description |
|---|---|---|---|
| `face_id` | i32 | — | dense; equals `region_id` of the corresponding region |
| `label` | i32 | palette idx | printed color |
| `outer_walk` | (i32, bool)[] | — | ordered (arc_id, reversed) pairs, closed |
| `hole_walks` | (i32, bool)[][] | — | zero or more closed walks, sorted by min anchor |

**Relationships.** Faces ↔ Regions 1:1 (id equality); consumed by Simplification, Smoothing (as replaced generations), Bézier Fitting, topology validator. **Ownership.** owns arcs and faces. **Lifecycle.** v1 Arc Graph → v2 Simplification → v3 Smoothing → consumed by Bézier Fitting (which emits CurveSet; the ArcGraph stays bound until validation). **Serialization.** debug JSON dump (the canonical diagnostics format for topology bugs). **Memory layout.** pooled arc coordinates (f64, 16 B/vertex) + face walk index arrays. **Validation.** Euler identity (MATH_SPEC §5.2); every arc referenced by exactly 2 face walks counting sides; walks closed head-to-tail; `work_scale > 0`. **Immutability/Thread safety.** global rules.

## 12. BezierSegment

**Purpose.** One cubic Bézier piece (MATH_SPEC §9.1).

| Field | Type | Units | Description |
|---|---|---|---|
| `control` | f64[4, 2] | pt | b₀…b₃ |

**Relationships.** Composed by `Curve`. **Ownership.** containing Curve. **Lifecycle.** with CurveSet. **Serialization.** 8 floats. **Memory layout.** 64 B; segments of a curve contiguous. **Validation.** finite; `b₀ ≠ b₁ or b₂ ≠ b₃` (no doubly-degenerate segments); consecutive segments share endpoints exactly (bitwise). **Immutability/Thread safety.** global rules.

## 13. Curve

**Purpose.** The fitted Bézier chain for one arc.

| Field | Type | Units | Description |
|---|---|---|---|
| `arc_id` | i32 | — | source arc |
| `segments` | BezierSegment[S] | pt | ordered chain |
| `corner_indices` | i32[] | — | segment joints that are intentional C0 corners |
| `max_fit_error_pt` | f64 | pt | fitter's measured max deviation (QM-04 evidence) |

**Relationships.** 1:1 with Arc; composed by CurveSet. **Ownership.** CurveSet. **Lifecycle.** created by Bézier Fitting; terminal geometry. **Serialization.** JSON; SVG path `d`-string is a *rendering* of this object, not its storage form. **Memory layout.** contiguous segment array. **Validation.** chain endpoints equal the arc's junction coordinates bitwise (watertightness at junctions, ENGINE_SPEC §18.4); G1 at non-corner joints (mirrored unit tangents to 1e−9); `max_fit_error_pt ≤` config bound. **Immutability/Thread safety.** global rules.

## 14. CurveSet

**Purpose.** Final vector geometry: all curves + the face structure carried over unchanged (ARCHITECTURE.md §4.1).

| Field | Type | Units | Description |
|---|---|---|---|
| `curves` | Curve[A] | pt | sorted by arc_id, dense |
| `faces` | Face[F] | — | identical records to the source ArcGraph (by value) |
| `provenance` | Provenance | — | §2 |

**Relationships.** Consumed by Label Placement, all renderers, printability/topology validators. **Ownership.** owns curves; faces copied by value from ArcGraph v3. **Lifecycle.** terminal geometry artifact; lives to the end of the run. **Serialization.** debug JSON; goldens compare its SVG rendering. **Memory layout.** ~12 000 segments × 64 B ≈ 0.8 MiB typical. **Validation.** curve set dense over arc ids; face walks reference existing curves. **Immutability/Thread safety.** global rules.

## 15. Label

**Purpose.** Placement of one region's printed number (ENGINE_SPEC §19).

| Field | Type | Units | Description |
|---|---|---|---|
| `region_id` | i32 | — | target face |
| `printed_number` | i32 | — | after palette permutation (§16) |
| `anchor` | f64[2] | pt | text center (in-region: the pole; leader: outside point) |
| `font_size_pt` | f64 | pt | ≥ `font_min_pt` |
| `mode` | enum {IN_REGION, LEADER} | — | placement kind |
| `leader` | f64[2, 2] \| null | pt | segment (from text edge to pole), LEADER only |
| `clearance_pt` | f64 | pt | r* of the face (QM-10 evidence) |

**Relationships.** 1:1 with faces (coverage gate QM-21); composed by `LabelPlan` (the artifact = `labels: Label[F]` sorted by region_id + provenance — LabelPlan has no other fields and is not separately specified). **Ownership.** LabelPlan. **Lifecycle.** created by Label Placement; validator repairs (in-region → leader demotion) produce a new LabelPlan. **Serialization.** JSON record. **Memory layout.** ~96 B/label. **Validation.** `font_size_pt ∈ [font_min, font_max]`; LEADER ⇒ `leader ≠ null`; rotation does not exist in the model (QM-23 is enforced structurally). **Immutability/Thread safety.** global rules.

## 16. Legend

**Purpose.** The number↔color key layout (ENGINE_SPEC §21) plus the palette permutation (ENGINE_SPEC §20).

| Field | Type | Units | Description |
|---|---|---|---|
| `permutation` | i32[K] | — | `printed_number = permutation[palette_index] + 1` (printed numbers are 1-based) |
| `chips` | (i32, f64[2], f64)[K] | —, pt, pt | (palette_index, top-left, chip side) in printed-number order |
| `band_rect` | f64[4] | pt | legend band (x, y, w, h) within the page |
| `number_font_pt` | f64 | pt | chip label size |
| `provenance` | Provenance | — | §2 |

**Relationships.** Consumed by all renderers; permutation consumed by Label Placement (printed numbers) and the mystery-leakage gate (QM-19). **Ownership.** context binding. **Lifecycle.** created by Legend Generation; terminal. **Serialization.** JSON. **Memory layout.** ≤ 64 entries; trivial. **Validation.** `permutation` is a bijection on [0, K); chips within `band_rect`; band within page; chip side ≥ 4 mm in pt. **Immutability/Thread safety.** global rules.

## 17. Page

**Purpose.** Resolved physical page geometry — a *config-derived value object*, not a pipeline artifact (no provenance; it is part of the resolved config snapshot).

| Field | Type | Units | Description |
|---|---|---|---|
| `width_pt` / `height_pt` | f64 | pt | trim size |
| `margin_pt` | f64 | pt | uniform margin |
| `content_rect` | f64[4] | pt | artwork box (excludes legend band) |
| `legend_rect` | f64[4] | pt | legend band |
| `dpi` | i32 | px/in | raster-export resolution |

**Relationships.** Read by Arc Graph (Φ), Legend, renderers, PDF trim box. **Ownership.** resolved config. **Lifecycle.** constructed at config resolution, before any stage runs (the band reservation ordering rule, ENGINE_SPEC §21.1). **Serialization.** part of the resolved-config JSON. **Memory layout.** trivial. **Validation.** `content_rect` and `legend_rect` disjoint, both inside margins; all extents > 0 (ConfigError otherwise). **Immutability/Thread safety.** global rules.

## 18. ValidationReport

**Purpose.** Structured result of one validator (ENGINE_SPEC §25); the aggregate of all four gates the run.

| Field | Type | Units | Description |
|---|---|---|---|
| `validator_name` | str | — | e.g. `topology` |
| `findings` | Finding[] | — | see below; sorted (severity desc, location) |
| `passed` | bool | — | no FATAL remaining after declared repairs |
| `metrics` | map[str, f64] | per QM | measured values feeding QM ids (e.g. `ssim`, `min_inscribed_mm`) |

**Finding**: `severity` enum {INFO, WARNING, REPAIRED, FATAL} · `invariant` str ("I1"…"I4", "palette") · `message` str · `location` str (region/arc id or coordinates) · `repair_applied` bool.

**Relationships.** Consumed by the orchestrator (gate decision), embedded in `OutputBundle.report`, exported to the benchmark harness (shared measurement rule, BENCHMARK_SPEC §6). **Ownership.** orchestrator collects; bundle embeds copies. **Lifecycle.** created per validator run; a repair triggers full re-validation producing fresh reports (max 2 iterations). **Serialization.** JSON; schema semver-governed (public plugin interface — third-party validators emit the same shape). **Memory layout.** trivial. **Validation.** `passed = (no FATAL)` consistency; finding locations non-empty. **Immutability/Thread safety.** global rules.

## 19. OutputBundle

**Purpose.** The atomic final deliverable (ARCHITECTURE.md §11: all or nothing).

| Field | Type | Units | Description |
|---|---|---|---|
| `svg` | bytes | — | canonical output (I2 hash surface) |
| `pdf` | bytes \| null | — | if `pdf.enabled` |
| `previews` | map[str, bytes] | — | keys exactly {`lineart`, `solved`} |
| `report` | RunReport | — | see below |

**RunReport**: `resolved_config` (full JSON) · `engine_version` str · `input_hash` hex64 · `seed` i64 · `warnings` str[] · `stage_timings_s` map[stage, f64] · `stage_metrics` map[stage, map[str, f64]] · `validation` ValidationReport[4] · `renumber_map` i32[K] · dataset of every artifact's provenance.

**Relationships.** Returned by `convert()`; consumed by adapters and the benchmark harness. **Ownership.** caller after return. **Lifecycle.** assembled by the orchestrator only after all validators pass; written to disk via temp-dir + rename (atomicity). **Serialization.** the object *is* serialized form (bytes + JSON report); schema semver-governed (public interface #1). **Memory layout.** ≤ ~6 MiB typical (QM-32 bands). **Validation.** `svg` non-empty; previews keys exact; report embeds exactly 4 validation reports, all `passed`. **Immutability/Thread safety.** global rules.

## 20. BenchmarkReport

**Purpose.** One benchmark run's complete machine-readable result (BENCHMARK_SPEC §11 is the normative schema; this section fixes its object model).

| Field | Type | Description |
|---|---|---|
| `run_id`, `timestamp_utc`, `git_sha`, `engine_version` | str | identity |
| `machine` | MachineFingerprint | cpu, cores, mem, container digest, kernel, lockfile hash, canary time |
| `dataset_version`, `score_version`, `report_schema` | i32 | comparability keys (§10.4 of BENCHMARK_SPEC) |
| `metrics` | map[fixture, map[preset, map[QM-id, MetricResult]]] | `MetricResult = {value f64, band [lo, hi], class enum{gate, monitor}, pass bool}` |
| `stages` | map[fixture, map[preset, map[stage, {wall_s f64, rss_mib f64}]]] | perf detail |
| `golden` | map[str, GoldenResult] | outcome enum {identical, changed_compatible, incompatible} + SSIMs |
| `score` | {total f64, dimensions map[str, f64]} | §10.2 of BENCHMARK_SPEC |
| `verdict` | {accepted bool, failures FailureTuple[]} | `FailureTuple = (metric, fixture, preset, value, band)` |

**Relationships.** Produced by the benchmark harness from `OutputBundle`s + validators; appended to `benchmarks/history/`; consumed by leaderboard/chart generation and drift detection. **Ownership.** CI artifact store. **Lifecycle.** append-only history; never edited. **Serialization.** JSON, 6-significant-digit numbers, schema-validated in CI (`report_schema: 3`). **Memory layout.** n/a (file object). **Validation.** schema validation; `verdict.accepted ⇔ failures = []`; comparability keys present. **Immutability/Thread safety.** append-only file; readers tolerate concurrent appends by reading complete files only.

---

## 21. Object relationship diagram

```
RasterImage ──▶ LabelMap ◀──── Palette ◀─── PaletteColor
                  │                │
                  ▼                │
   Region ◀── RegionGraph ◀───────┘ (ΔE edges)
     │            │ component_map
     │            ▼
     │      TopologyGraph ── Arc ──▶ ArcGraph ── Face
     │                                  │          │ 1:1 (face_id = region_id)
     │                                  ▼          │
     │        BezierSegment ── Curve ─ CurveSet ◀──┘
     │                                  │
     └──────────────┐                   ▼
                    ▼             Label / LabelPlan     Legend ◀── Page
                    └───────────────────┴──────┬───────────┘
                                               ▼
                        ValidationReport ─▶ OutputBundle ─▶ BenchmarkReport
```

## 22. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-06 | Initial complete data model specification (18 objects + provenance). |

