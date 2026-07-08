# Golden Test Framework (Sprint 21)

**Status:** v1.0 — testing infrastructure only, no algorithm code. Companion to [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) §4 (golden protocol this framework extends) and [DATASET_STANDARDS.md](DATASET_STANDARDS.md) (the fixture set this framework runs over).

## 1. Purpose

`benchmarks/golden/` runs every fixture in the Sprint 20 categorized dataset (`benchmarks/datasets/`) through the real engine and compares the output against a frozen golden on three independent axes: perceptual, SVG structural, and topology. It complements BENCHMARK_SPEC.md §4's existing golden protocol (implemented for the synthetic fixture ladder in `benchmarks/framework/visual.py`) rather than replacing it — this tier is scoped to the categorized dataset specifically, with its own storage location and its own topology axis.

## 2. Pipeline

```
Every dataset fixture
    ↓
convert() -- the real engine, via benchmarks/framework/pipeline.run_pipeline
    ↓
SVG (+ PDF preview, when the pdf extras are installed)
    ↓
compare against stored Golden Output
    ↓
GoldenReport (perceptual + structural + topology outcomes)
```

No stage is re-implemented: `benchmarks/golden/runner.py` adapts a `DatasetFixture` (Sprint 20) into the existing `benchmarks.framework.fixtures.Fixture` shape and calls the framework's own `run_pipeline` — the one, already-reviewed path through the engine (BENCHMARK_SPEC.md §6).

## 3. Modules

| Module | Responsibility |
|---|---|
| `runner.py` | Adapts dataset fixtures and runs them through `run_pipeline` |
| `tolerances.py` | `GoldenTolerances` — every pass/fail threshold, in one place |
| `topology_compare.py` | `TopologyFingerprint` + region/arc/face count comparison (new axis) |
| `compare.py` | `GoldenReport` — combines perceptual (SSIM), structural (SVG diff), and topology into one outcome |
| `storage.py` | On-disk golden read/write under `benchmarks/golden_store/<fixture_id>/` |
| `update.py` | The bless workflow — the only code path that writes goldens |
| `report.py` | `GoldenSuiteReport` — aggregates every fixture's `GoldenReport` for one run |
| `cli.py` | `python -m benchmarks.golden.cli run \| bless` |

Perceptual (SSIM) and structural (arc count / face-side multiset / per-arc segment count) comparisons reuse `benchmarks/framework/visual.py`'s existing primitives rather than duplicating them.

## 4. The three comparison axes

### 4.1 Perceptual

Luminance SSIM of the rasterized PDF preview vs. the golden's stored preview. Pass threshold: `GoldenTolerances.ssim_min` (default `0.97`, matching BENCHMARK_SPEC.md §4.2). Skipped when the `pdf` extras aren't installed or the golden has no preview.

### 4.2 SVG structural

When the SVG byte hash differs from golden: arc count, face-side multiset, and per-arc Bezier segment count (within `GoldenTolerances.segment_count_tolerance`, default 10%) must all match — identical to BENCHMARK_SPEC.md §4.2's structural diff.

### 4.3 Topology (new in Sprint 21)

Region count, arc count, and face count, compared between a `TopologyFingerprint` frozen at bless-time and a fresh run. This catches a case the perceptual/structural axes can miss: visually near-identical output with a structurally different region graph (e.g. a merge-threshold regression that fuses two regions but happens to look the same at preview resolution).

Region/arc/face counts are analytic ground truth for a fixed, deterministic fixture (BENCHMARK_SPEC.md §3) — the default tolerance is **zero**: `GoldenTolerances.topology_region_count_tolerance` / `topology_arc_count_tolerance`. A non-zero band is available for a fixture whose engine output is legitimately non-deterministic in count (none currently), but should be treated as an explicit, reviewed exception, not a default.

A `GoldenReport.passed` requires the SVG outcome to not be `INCOMPATIBLE` *and* the topology comparison to pass — the two axes are independent gates, not substitutes for each other.

## 5. Tolerance configuration

All thresholds live in `benchmarks/golden/tolerances.py`'s `GoldenTolerances` dataclass, with `DEFAULT_TOLERANCES` as the module-level default. Every comparison function takes `tolerances` as a keyword argument, and every `GoldenReport`/`GoldenSuiteReport` records the tolerances used — a report is self-describing without cross-referencing config.

## 6. Golden update ("bless") workflow

Goldens are written **only** by `benchmarks/golden/update.py`, and only on explicit invocation — never implicitly during a comparison run (BENCHMARK_SPEC.md §4.3).

```bash
# Bless the frozen one-per-category golden subset
python -m benchmarks.golden.cli bless --suite golden

# Bless every fixture in the full categorized dataset
python -m benchmarks.golden.cli bless --suite full

# Bless specific fixtures
python -m benchmarks.golden.cli bless --fixture D-animals-examples-01 --fixture D-food-examples-01
```

Blessing writes, per fixture, under `benchmarks/golden_store/<fixture_id>/`:

- `page.svg` — the golden SVG
- `preview.png` — the rasterized PDF preview (when available)
- `topology.json` — the `TopologyFingerprint`
- `GOLDEN_MANIFEST.json` — fixture id, category, engine version, dataset version, topology summary

A blessing commit should include the report diff for reviewer inspection, per BENCHMARK_SPEC.md §4.3's discipline: review is of the report, not of pixels on someone's laptop.

## 7. Running comparisons and producing reports

```bash
# Compare current engine output to the golden subset (1 fixture/category)
python -m benchmarks.golden.cli run --suite golden --out benchmarks/golden_reports/latest.json

# Compare against the full categorized dataset
python -m benchmarks.golden.cli run --suite full --out benchmarks/golden_reports/full.json
```

Exit code is `0` iff every fixture's `GoldenReport.passed`. The written JSON report (`GoldenSuiteReport.to_dict()`) contains, per fixture: outcome, SSIM value, topology comparison, tolerances used, and diff details — plus a `summary` block (total/passed/failed).

## 8. Known issue: engine determinism under hash-seed randomization

Building this framework surfaced that the engine's region/topology pipeline is, for at least some fixtures (observed on the `animals` category's overlapping-blob label maps), sensitive to Python's per-process hash-seed randomization: region/arc/face counts varied run-to-run for byte-identical input across separate process invocations with `PYTHONHASHSEED` unset. This is a genuine gap against the QM-27 determinism gate (BENCHMARK_SPEC.md §8.5) — almost certainly unordered-container iteration order (`set`/`dict`) leaking into a merge or ordering decision somewhere in the graph/vector stages.

This is an **engine bug**, and Sprint 21 is scoped to testing infrastructure only ("no engine modifications") — it is not fixed here. As a harness-level workaround so golden comparisons are reproducible in the meantime:

- `benchmarks/golden/cli.py` re-executes itself with `PYTHONHASHSEED=0` if not already set.
- The `golden-dataset` CI job (`.github/workflows/ci.yml`) sets `PYTHONHASHSEED: "0"` at the job level.
- Tests in `benchmarks/golden/tests/` bless and compare within the same process, so they don't depend on hash-seed pinning for correctness (a same-process run is internally consistent regardless of the seed).

Follow-up: file an engine-side investigation into hash-seed sensitivity in the region-merge/topology path; this workaround should be removed once the root cause is fixed.

## 9. CI integration

The `golden-dataset` job (`.github/workflows/ci.yml`) runs `python -m benchmarks.golden.cli run --suite golden` on every push/PR, with `PYTHONHASHSEED` pinned, and uploads the JSON report as a build artifact. It does not gate merges more strictly than the existing `test`/`bench-smoke` jobs today — promoting it to a required, blocking check is a follow-up decision once the golden subset has been reviewed and blessed against a stable engine baseline.

## 10. Scope

This framework is testing infrastructure: golden comparison, tolerance configuration, the bless workflow, reporting, and CI wiring. It does not implement or modify any engine algorithm, stage, or validator.
