# Technical Quality Comparison Framework (Sprint 24)

**Status:** v1.0 — evaluation infrastructure only, no algorithm code. Companion to [DATASET_STANDARDS.md](DATASET_STANDARDS.md) (the fixture set this framework runs over), [GOLDEN_TEST_STANDARDS.md](GOLDEN_TEST_STANDARDS.md) (the sibling golden-comparison framework), and the Sprint 23 quality validator (`src/mysterycbn/validate/quality_metrics.py`).

## 1. Purpose and scope

This framework measures and compares the engine's **technical** output quality — region count, compactness, boundary smoothness, average edge length, region size distribution, label density, printability — across the three difficulty presets (`easy`/`medium`/`hard`), and generates advisory recommendations from the deltas it finds.

**This is not an artwork-quality or aesthetic-judgment tool, and it does not compare against or reproduce any copyrighted image.** ARCHITECTURE.md §10 states the repo's hard legal invariant: *"Fixtures are original, in-repo assets (no copyrighted imagery — also a legal invariant)."* Every fixture this framework runs comes from `benchmarks/datasets/`'s synthetic, procedurally-generated categories (animals, flowers, people, landscape, architecture, food, vehicles, cartoons) — see DATASET_STANDARDS.md §2. No photograph, scraped image, or externally-sourced asset is used anywhere in this package, and none should ever be added to it.

## 2. What "comparison" means here

The comparison axis is **cross-preset**: for each dataset fixture, the real pipeline is run three times — once per difficulty preset — and the resulting technical-quality metrics are compared pairwise (easy→medium, medium→hard). This reuses the exact preset definitions production `convert()` resolves against (`app/config_defaults.D_MIN_MM_BY_PRESET` / `N_COLORS_BY_PRESET`), not a comparison-specific approximation of them.

This is deliberately scoped smaller than "compare arbitrary configurations or engine versions" — there is no baseline/candidate snapshot storage in this framework (that would be a natural extension once there's a need to compare across engine versions, similar in spirit to `benchmarks/golden`'s blessed-golden model, but is out of scope here).

## 3. Metrics measured

| Metric | Source | New in Sprint 24? |
|---|---|---|
| Region count | `curve_set.faces` (QM-13) | Reused from Sprint 23 |
| Mean compactness | isoperimetric ratio per face (QM-14) | Reused from Sprint 23 |
| Boundary smoothness | curvature energy (QM-08) | Reused from Sprint 23 |
| Average edge length | mean flattened boundary edge length (mm) | **New** (`metrics.py::average_edge_length_mm`) |
| Region size distribution | min/max/mean/median/stdev/p10/p90 of face area (mm²) | **New** (`metrics.py::region_size_distribution`) |
| Label density | printed labels per cm² of printable page area | **New** (`metrics.py::label_density_per_cm2`) |
| Printability score | `1 - tiny_region_pct/100`, floor 0.5 (BENCHMARK_SPEC §10.2) | Reused from Sprint 23 |

Average edge length and region size distribution had no prior implementation anywhere in the repo (confirmed absent by full-repo search, not merely unexported) — both are independently re-derived from the same final `curve_set` geometry Sprint 23's validator reads, following the same "flatten and measure" convention rather than trusting any construction-time value.

### 3.1 Known characteristic: boundary smoothness reads exactly zero on these fixtures

The synthetic dataset generators (Sprint 20) rasterize shapes on a pixel grid before curve-fitting, so region boundaries are inherently blocky: every turn angle in the flattened geometry is either ~0° (straight run) or exactly 90° (a raster-grid corner). QUALITY_SPEC's QM-08 formula excludes corners >60° as "intentional, not noise" — with every non-zero angle in these fixtures landing exactly at 90°, the filtered curvature-energy sum is legitimately zero across every category and preset. This is a property of the synthetic dataset, not a bug in the metric: a raster-sourced photograph would produce genuinely varied sub-60° angles and a non-trivial QM-08 value. Don't read "0.0 everywhere" here as evidence the metric is broken; it means this dataset's boundaries are, in fact, perfectly angular at the resolution these fixtures render at.

## 4. Palette construction across presets

`benchmarks/framework/pipeline.py`'s default synthetic palette builder (`_palette_for`, a single-radius LAB hue wheel) only clears the QM-16 `merge_delta_e` FATAL floor up to about 10-16 colors (documented limitation, see `fixtures.py`) — insufficient for the `hard` preset's 24 colors. `benchmarks/comparison/runner.py::_palette_for_preset` varies L* across three bands as well as hue, which clears the floor through all three preset sizes (verified: minimum pairwise ΔE00 stays ≥ 8.89 at the worst case, n=16). This is a comparison-harness-scoped palette builder, not a change to the shared one other packages use.

`n_colors` is also floored at each fixture's own label-map maximum (`_as_bench_fixture`): a preset with fewer colors than a fixture's labels actually use would violate `LabelMap.validate_against` before the comparison could measure anything, so every preset is guaranteed runnable on every fixture.

## 5. Architecture

| Module | Responsibility |
|---|---|
| `metrics.py` | Net-new metrics: average edge length, region size distribution, label density |
| `runner.py` | Runs one dataset fixture under one/all presets via the real pipeline (`benchmarks.framework.pipeline.run_pipeline`, unmodified execution path) |
| `evaluate.py` | `QualitySnapshot` — every technical-quality measurement for one (fixture, preset) run |
| `recommend.py` | Rule-based `Recommendation` generation from cross-preset deltas |
| `report.py` | `ComparisonReport` — assembles snapshots + recommendations across a scope (examples / one category / full dataset) |
| `cli.py` | `python -m benchmarks.comparison.cli run --scope ...` |

No engine, stage, or rendering code is modified. Two small, backward-compatible additions were made to the shared `benchmarks/framework/pipeline.run_pipeline` (used by this package, `benchmarks/golden`, and `benchmarks/framework/quality.py` alike): an optional `d_min_mm` override (default `None` preserves prior behavior) and an optional `palette_factory` override (default `None` uses the existing `_palette_for`). Both are additive parameters with safe defaults — no existing caller changes behavior.

## 6. Recommendations

`recommend.py` runs a fixed set of independent, side-effect-free rules over each consecutive preset pair (easy→medium, medium→hard) for a fixture's snapshot sequence:

| Rule | Fires when | Severity |
|---|---|---|
| Region count growth | region count grows ≥1.3x | info |
| Boundary smoothness regression | curvature energy worsens ≥1.5x | caution |
| Compactness drop | mean compactness falls ≥15% | caution |
| Tiny-region increase | tiny-region % rises ≥5 points | caution |
| Label density growth | label density grows ≥1.5x | info |
| Edge length collapse | average edge length halves or more | info |
| Printability near floor | score falls and drops to ≤0.6 | caution |

A rule that finds nothing worth flagging contributes nothing — an empty recommendation list means clean, not unmeasured. Recommendations are advisory only: nothing in this framework can fail a build or block a run, matching Sprint 23's quality-metrics validator's Monitor-only posture.

## 7. Usage

```bash
# The 8-fixture example ladder (one per category), fast
python -m benchmarks.comparison.cli run --scope examples --out report.json

# One category, every tier/difficulty
python -m benchmarks.comparison.cli run --scope category --category animals

# The full 32-fixture dataset x 3 presets = 96 pipeline runs
python -m benchmarks.comparison.cli run --scope full
```

The written JSON (`ComparisonReport.to_dict()`) contains, per fixture: every preset's `QualitySnapshot` and every cross-preset `Recommendation`, plus a summary block (fixture count, recommendation count, caution count).

Like `benchmarks/golden/cli.py`, this CLI re-execs itself with `PYTHONHASHSEED=0` pinned — see GOLDEN_TEST_STANDARDS.md §8 for the underlying engine hash-seed sensitivity this works around at the harness level.

## 8. Scope

This framework is evaluation infrastructure: metric computation, cross-preset comparison, and advisory recommendation generation. It does not implement or modify any engine algorithm, stage, or validator, and every sample it evaluates is a synthetic, in-repo, non-copyrighted construction (ARCHITECTURE.md §10) — never a reproduction of, or comparison against, any external or copyrighted artwork.
