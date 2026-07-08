# Module Design — Quantization (`stages/raster/quantize`)

**Status:** v1.1 — **implemented** at `src/mysterycbn/stages/raster/quantize.py` (default `labkmeans` per §5, plus `mediancut` as the first registered alternative; octree remains §16.1 future work). The phase-2 legacy `modules/quantize.py` (§17) is superseded and awaits deletion with the phase-3 migration.
**Governing specs:** [ENGINE_SPEC.md §7](../ENGINE_SPEC.md) (normative algorithm), [MATH_SPEC.md §3–4](../MATH_SPEC.md) (color math), [QUALITY_SPEC.md QM-15/16/27](../quality/QUALITY_SPEC.md) (gates), [DATA_MODEL_SPEC.md §4–6](../DATA_MODEL_SPEC.md) (artifact shapes). This document refines those specs to implementation-ready detail; on any conflict, this document loses and must be corrected.
**Note:** a phase-2 implementation exists at `src/mysterycbn/modules/quantize.py` (pre-v2 layout). It migrates to `stages/raster/quantize.py` under this design at phase 3; behavioral deltas from that code are listed in §17.

---

## 1. Purpose

Reduce the working raster to K perceptually separated palette colors and a per-pixel label map. This is the pipeline's single most quality-determining stage: every downstream region, boundary, and printed number derives from this assignment. It is also the first *irreversible* step — color information not captured here is unrecoverable.

## 2. Requirements

| Id | Requirement | Source |
|---|---|---|
| R1 | Deterministic: identical (input, config, seed) ⇒ byte-identical LabelMap and Palette | I2 |
| R2 | Exact-K control: caller requests `n_colors`; output K ≤ n_colors, reduced only by ΔE00 merging | ENGINE_SPEC §7 |
| R3 | Perceptual objective: minimize within-class LAB variance; all thresholds in ΔE00 | MATH_SPEC §4.2 |
| R4 | Palette separation: min pairwise ΔE00 ≥ `merge_delta_e` by construction | QM-16 |
| R5 | Fidelity: mean per-pixel ΔE00 ≤ 11.0 (gate), ≤ 9.0 (target) on `F-photo-2`, K=16 | QM-15 |
| R6 | Coverage-ordered numbering: label 0 = largest coverage, descending | ENGINE_SPEC §7.7 |
| R7 | Runtime ≤ 2.0 s at 1600 px, K=16, single core | ENGINE_SPEC §26 |
| R8 | Peak additional memory ≤ 3 raster-sized float buffers | QM-31 budget share |
| R9 | Third-party isolation: OpenCV/NumPy internals never appear in the interface | ARCHITECTURE.md §1.1 |
| R10 | Side-effect-free outside the context; single config section `quantize` | Stage protocol |

## 3. Inputs

| Artifact / value | Type | Constraints |
|---|---|---|
| `raster_working` | `RasterImage` (DATA_MODEL §3) | f32 sRGB [0,1], working resolution, `work_scale > 0` |
| config section `quantize` | see §14 | validated at config resolution |
| `stage_seed` | u64 | `SHA-256(seed ‖ "quantize")[:8]` (ENGINE_SPEC §1.3) |

Requires (stage protocol): `raster_working`. The stage must not read any other artifact or config section.

## 4. Outputs

| Artifact | Type | Contract |
|---|---|---|
| `label_map` | `LabelMap` (DATA_MODEL §6) | i32[H,W], values ∈ [0, K), dense |
| `palette` | `Palette` (DATA_MODEL §5) | K ∈ [2, 64]; LAB authoritative; ΔE00 table cached; min offdiag ΔE00 ≥ `merge_delta_e` |

Provides: `label_map`, `palette`. Both carry provenance `{stage: "quantize", version, config_hash, source_hash}`.

## 5. Algorithm

Normative summary (ENGINE_SPEC §7): seeded LAB k-means with k-means++ init, stride sampling, full assignment, ΔE00 center merging, coverage renumbering. Refinements fixed by this design:

1. **LAB conversion** — full-raster sRGB→LAB (MATH_SPEC §3), f64 intermediate, stored f32 (LAB dynamic range fits f32 with < 0.001 ΔE error; halves memory, R8).
2. **Sampling** — if N > `sample_px`: stride `t = ⌊N / sample_px⌋` over the row-major flattened raster, taking indices `0, t, 2t, …` until exactly `sample_px` samples. No RNG (stability under seed changes: the seed affects *init* only, not the data).
3. **Init (k-means++)** — first center: sample index `stage_seed mod sample_px`. Subsequent: standard D² sampling using a PCG64 stream seeded with `stage_seed + restart_index`. `n_init = 4` restarts.
4. **Lloyd iterations** — ΔE76 assignment (sanctioned inner loop), mean update in LAB, stop when `max_k ‖c_k − c_k′‖₂ < 0.05` or `max_iter`. Empty cluster: re-seed at the sample point with max distance to its assigned center (ties → lowest index); counts as movement (prevents premature stop).
5. **Restart selection** — keep the restart with lowest inertia `Σ min_k ‖x − c_k‖²`; ties → lowest restart index (R1).
6. **Full assignment** — all N pixels to nearest center, ΔE76, blocked evaluation (§8).
7. **Merge pass** — while min offdiag ΔE00 < `merge_delta_e`: merge the argmin pair (ties → lexicographic pair order) into the coverage-weighted LAB mean; relabel; recompute affected ΔE00 rows only.
8. **Finalize** — recompute exact per-class LAB means over *assigned pixels* (not sample); derive sRGB (gamut-clamped, MATH_SPEC §3.2); renumber by (coverage desc, then LAB lexicographic for equal coverage — R1); emit artifacts.

Step 8's mean recomputation matters: sample-fit centers are biased by the stride sample; exact means reduce QM-15 by ~0.3 ΔE00 at negligible cost.

## 6. Pseudocode

```
QUANTIZE(raster, cfg, stage_seed):
    lab      ← SRGB_TO_LAB(raster.pixels)                     # f32[H,W,3]
    X        ← STRIDE_SAMPLE(flatten(lab), cfg.sample_px)     # f32[S,3]

    best ← nil
    for r in 0 .. 3:                                          # n_init restarts
        C ← KMEANSPP_INIT(X, cfg.n_colors, seed = stage_seed + r)
        repeat ≤ cfg.max_iter:
            A ← argmin_k ‖X − C_k‖²                           # ΔE76 assignment
            C′ ← class_means(X, A);  FIX_EMPTY(C′, X, A)
            if max_k ‖C_k − C′_k‖ < 0.05: break
            C ← C′
        if inertia(X, A, C) < inertia(best) or best = nil: best ← C

    labels ← BLOCKED_ARGMIN(lab, best)                        # i32[H,W], full raster
    (best, labels) ← MERGE_CLOSE(best, labels, cfg.merge_delta_e)   # ΔE00 pair merging
    means  ← exact class means of lab under labels
    order  ← sort classes by (coverage desc, LAB lex)
    return LabelMap(remap(labels, order)), Palette(reorder(means, order))
```

`MERGE_CLOSE` invariant: after return, `min_{a≠b} ΔE00 ≥ merge_delta_e` (R4 holds by loop condition, not by trust).

## 7. Complexity

| Phase | Time | Notes |
|---|---|---|
| LAB conversion | O(N) | vectorized |
| sample fit | O(n_init · I · S · K) | S = 1e5, I ≤ 50 → ~10⁹ f32 mul worst case, ~0.6 s vectorized |
| full assignment | O(N · K) | dominant at large N; blocked |
| merge pass | O(K³) worst | K ≤ 64 → trivial |
| finalize | O(N + K log K) | one bincount pass |

Total O(N·K + n_init·I·S·K); fits R7 with margin ~2× at defaults.

## 8. Memory

Peak = input raster (12 B/px, owned by context) + LAB f32 copy (12 B/px) + label i32 (4 B/px) + distance block. Blocked argmin evaluates distances in row blocks of `B = 65 536` pixels (block buffer `B×K` f32 = 16 MiB at K=64), keeping the full `N×K` distance matrix (which would be 400+ MiB) off the table. Additional: sample copy 1.2 MiB, centers/tables < 1 MiB. **Peak ≈ 2.3 raster-equivalents + 16 MiB — within R8.**

## 9. Edge Cases

| Case | Behavior |
|---|---|
| distinct colors < n_colors | duplicate centers collapse in merge pass; K shrinks; never an error |
| `n_colors = 2` | valid (silhouette); merge pass may not reduce below 2 (guard: stop merging at K = 2) |
| grayscale input | centers on L axis; a*, b* ≈ 0; no special path |
| flat single-color input | all restarts identical; K collapses to 1 → **clamped to K = 2** by splitting the center ± 0.5 L (the degenerate-page path; downstream merge handles it, validator warns) |
| N ≤ sample_px | S = N, stride 1 |
| extreme palette request (64) on small raster | S ≥ 100·K enforced; else `sample_px` raised to min(N, 100·K) |
| NaN/Inf pixels | impossible per RasterImage validation (DATA_MODEL §3); assert only in debug |
| alpha-flattened white dominance | expected; coverage ordering gives white label 0 (legend convention) |

## 10. Quality Metrics

Owned gates and monitors (measured per QUALITY_SPEC):

| Metric | Class | Band |
|---|---|---|
| QM-15 mean ΔE00 fidelity | Gate | ≤ 11.0 (target ≤ 9.0), `F-photo-2` K=16 |
| QM-16 palette separation | Gate | min offdiag ΔE00 ≥ `merge_delta_e` |
| QM-27 determinism (via SVG hash) | Gate | this stage must contribute zero nondeterminism |
| stage wall time | Gate | ≤ 2.0 s (ENGINE_SPEC §26) |
| inertia per fixture | Monitor | regression tolerance +5 % vs baseline |
| K after merge, per fixture | Monitor | ±2 vs baseline |

## 11. Unit Tests

All under `tests/unit/test_quantize.py` + property tests in `tests/property/`:

1. **Exact recovery** — synthetic 4-color image (analytic ground truth, BENCHMARK_SPEC §3 tier 1): recovers 4 centers, each within 0.5 ΔE00; every pixel correctly labeled.
2. **Determinism** — two runs, identical bytes (labels + palette); different `seed` may differ; different `stage_name` hash isolation verified.
3. **Merge invariant** — construct centers at ΔE00 = 3 apart; post-merge min offdiag ≥ threshold; K decreased accordingly; weighted-mean math checked against hand values.
4. **Coverage ordering** — 3-color synthetic with known areas: labels ordered by area; tie case ordered by LAB lex.
5. **Empty-cluster path** — adversarial init forcing an empty cluster (K > distinct colors in sample); re-seed rule hit; no crash; determinism preserved.
6. **Flat-input clamp** — single-color raster yields K = 2 with the ±0.5 L split.
7. **Sampling** — N ≤ sample_px uses all pixels; stride arithmetic exact at boundaries (N = sample_px, N = sample_px+1).
8. **Property (Hypothesis)** — random small rasters: (a) every label < K, dense; (b) palette sRGB = clamped conversion of LAB; (c) determinism under repeated runs; (d) merge invariant holds for random thresholds.
9. **Contract suite** — the shared quantizer contract test (ENGINE_SPEC §10.3): any registered quantizer implementation must pass tests 2, 4, and the artifact-shape checks.

## 12. Benchmarks

- `benchmarks/perf/`: stage wall + RSS at 1600 px for K ∈ {8, 16, 30}, ladder fixtures; budget 2.0 s (K=16).
- `benchmarks/quality/color.py`: QM-15, QM-16, inertia, final K per fixture×preset.
- Baseline update policy per BENCHMARK_SPEC §7.1.

## 13. Configuration

Section `quantize` (the only section this stage may read — R10):

| Key | Type | Default | Range | Notes |
|---|---|---|---|---|
| `n_colors` | int | 16 | 2–64 | auto-tunable (analyze stage may propose if user left unset) |
| `merge_delta_e` | float | 7.0 | 0–30 | preset easy: 12.0 |
| `sample_px` | int | 100 000 | 10⁴–10⁶ | effective value ≥ 100·n_colors |
| `max_iter` | int | 50 | 10–200 | |
| `impl` | str | `"labkmeans"` | registered names | plugin selection (ENGINE_SPEC §8) |

## 14. Public Interface

Quantization is a **Stage plugin** (public interface #4). Its public surface is exactly:

- **Stage identity:** `name = "quantize"`, `version` (semver of this design's implementation), `requires = ["raster_working"]`, `provides = ["label_map", "palette"]`, `config_section = "quantize"`.
- **Artifacts:** `LabelMap`, `Palette` per DATA_MODEL §5–6 — the schema third-party quantizers must emit.
- **Contract semantics** (enforced by the shared contract test): determinism (R1), separation invariant (R4), coverage ordering (R6), density of labels.

Nothing else — not the k-means internals, not the sampling scheme — is public. A third-party octree quantizer replacing this module interacts only through the rows above.

## 15. Internal Interface

Private structure of the default implementation (not semver-governed, listed for reviewability):

| Function | Contract |
|---|---|
| `stride_sample(flat_lab, s) → X` | exact-count stride sampling, §5.2 |
| `kmeanspp_init(X, k, rng) → C` | D² seeding, PCG64 stream |
| `lloyd(X, C, tol, max_iter) → (C, A, inertia)` | with `fix_empty` rule §5.4 |
| `blocked_argmin(lab, C, block) → labels` | §8 memory scheme |
| `merge_close(C, labels, τ) → (C, labels)` | invariant per §6; stops at K=2 |
| `finalize(lab, labels) → (Palette, LabelMap)` | exact means, ordering, artifact construction |

Color conversions come from `foundation/color` (never reimplemented here — ARCHITECTURE.md §3); the ΔE00 table from `Palette`'s cached table.

## 16. Future Extensions

1. **Octree quantizer** (`impl = "octree"`) for a fast preset — must pass the §14 contract suite.
2. **Edge-prior weighting** (§14.2 of ARCHITECTURE.md): per-pixel weights in the class-mean updates, supplied as a raster prior artifact; interface impact = one new *optional* requires entry.
3. **Weighted sampling** by local gradient (denser samples near edges) — internal-only change.
4. **Numba blocked-argmin kernel** if profiling shows R7 pressure at 6000 px working resolution — with pure-NumPy reference retained (ARCHITECTURE.md §13.3).

## 17. Deltas vs the phase-2 implementation

The existing `modules/quantize.py` differs from this design in: (a) it uses OpenCV's `cv2.kmeans` with `cv2.setRNGSeed` — deterministic on one build, but not seed-stream-compatible with §5.3 and not guaranteed stable across OpenCV versions; must be replaced by the explicit PCG64 scheme; (b) it clusters **all** pixels (no sampling — the §5.2 stride sampler is new); (c) its merge pass is dominant-first greedy on **ΔE76**, not argmin-pair on ΔE00 (R4 currently holds only in ΔE76 terms); (d) no exact-mean finalize pass; (e) coverage tie-break unspecified; (f) it has a `chroma_weight` knob and a 1-based `PaletteColor.number` — both dropped in this design (weighting superseded by §16.2; numbering moves to `Legend.permutation` per DATA_MODEL §16); (g) a single-color collapse raises `StageError`, whereas this design clamps to K = 2 (§9). Migration at phase 3 must adopt this design and will change golden hashes — one `golden-update` PR, per BENCHMARK_SPEC §4.3.

## 18. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-06 | Initial design. |
