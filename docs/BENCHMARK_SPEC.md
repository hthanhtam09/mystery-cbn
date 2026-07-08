# Mystery Color-by-Number Engine — Benchmark Specification

**Status:** v1.0 — authoritative specification of the automatic evaluation system. Companion to [QUALITY_SPEC.md](quality/QUALITY_SPEC.md) (metric definitions QM-01…QM-33), [ENGINE_SPEC.md](ENGINE_SPEC.md), [ARCHITECTURE.md](ARCHITECTURE.md) §9.
**Purpose:** every engine change is evaluated by this system with zero human judgement in the loop. Humans review *reports*; they never produce *measurements*.

---

## 1. System overview

```
                    ┌────────────────────────────────────────────┐
 git push / PR ──▶  │ CI benchmark job (pinned container)        │
                    │  1. fixture integrity check (hashes)       │
                    │  2. run engine on ladder (seed 0)          │
                    │  3. compute QM-01…QM-33                    │
                    │  4. golden comparison (SVG/PNG)            │
                    │  5. regression analysis vs baselines/      │
                    │  6. emit report JSON + charts + score      │
                    └───────────────┬────────────────────────────┘
                                    ▼
              pass/fail gate      benchmarks/reports/<run-id>.json
              (PR status)         history DB · leaderboard · charts
```

Three suites, runnable independently:

| Suite | Path | Trigger | Wall budget |
|---|---|---|---|
| **smoke** | `benchmarks/smoke/` | every push | ≤ 3 min (2 fixtures, Gate metrics only) |
| **full** | `benchmarks/perf/` + `benchmarks/quality/` | every PR to main; nightly | ≤ 45 min (full ladder × presets) |
| **golden** | `tests/golden/` | every PR to main | ≤ 10 min (perceptual + structural diff) |

---

## 2. Benchmark datasets

### 2.1 Fixture inventory

All fixtures are **original, in-repo assets** (`assets/fixtures/`) — no copyrighted imagery (legal invariant, ARCHITECTURE.md §10). Each fixture is pinned by SHA-256 in `assets/fixtures/MANIFEST.json`; the run aborts if any hash mismatches (a silently edited fixture invalidates all history).

| Fixture id | Category | Size | Content requirements (verifiable) |
|---|---|---|---|
| `F-photo-05/2/12/24` | photograph | 0.5/2/12/24 MP | colorfulness M ∈ [40, 90]; edge density ρ ∈ [0.08, 0.25] |
| `F-illu-2` | flat-shaded illustration | 2 MP | ≤ 40 distinct colors pre-noise; ρ ≤ 0.10 |
| `F-flat-2` | hard-edged flat art | 2 MP | ≤ 12 distinct colors; no gradients (max per-channel local σ ≤ 2/255) |
| `F-noise-2` | high-noise photo | 2 MP | ρ ≥ 0.30 |
| `F-degen-1` | degenerate flat color | 1 MP | 1 distinct color |
| `F-gray-2` | grayscale photo | 2 MP | max chroma C* ≤ 2 |
| `F-thin-2` | synthetic thin-structure chart | 2 MP | contains 1–4 px lines, known ground truth (§3) |
| `F-alpha-2` | RGBA illustration | 2 MP | ≥ 20 % transparent pixels |

Category membership is *asserted by the harness* (the content-requirement columns are computed checks, not descriptions) — a replacement fixture that drifts out of its category fails CI.

### 2.2 Dataset versioning

The fixture set carries `dataset_version` (integer). Adding a fixture increments it; **modifying or removing** one requires an ADR plus regeneration of all goldens and baselines in the same commit. Reports record `dataset_version`; historical comparison (§10) only compares runs with equal versions per fixture.

---

## 3. Ground truth

Three tiers, by decreasing strength:

1. **Analytic ground truth** — synthetic fixtures with mathematically known answers, generated deterministically by in-repo generators (checkerboards, disks, gradients, `F-thin-2`). Known quantities: exact region counts, exact boundary geometry, exact palette. Used by unit/property tests and by benchmark assertions like "disk fixture inscribed diameter = D ± QM-03 band".
2. **Construction-invariant ground truth** — quantities every valid output must satisfy regardless of content: QM-01/02 (topology), QM-11 (tiny regions), QM-21 (coverage), QM-26/27/28 (validity/determinism). These need no reference data.
3. **Golden reference** (§4) — frozen previous-blessed outputs for content-dependent quantities where no analytic truth exists (region shapes on photographs). Weakest tier: goldens encode "accepted", not "correct"; they detect *change*, and the QM metrics decide whether change is regression.

No human-labeled ground truth exists anywhere in the system.

---

## 4. Golden images

### 4.1 Golden set

Per fixture × preset: the blessed `page.svg`, `lineart.png`, `solved.png`, stored under `tests/golden/<fixture>/<preset>/` with a `GOLDEN_MANIFEST.json` (hashes, engine version that produced them, resolved-config hash, dataset_version).

### 4.2 Comparison methods (all automatic)

| Artifact | Method | Pass band |
|---|---|---|
| SVG | byte hash | equal → pass, skip further checks |
| SVG (hash differs) | structural diff: arc count, face count, per-arc `data-left/right` multiset, segment count per arc ±10 %, coordinate RMS ≤ 0.3 mm | all conditions → "changed-compatible" |
| PNG previews | luminance SSIM vs golden | ≥ 0.97 → "changed-compatible" |

Outcomes: **identical** (pass), **changed-compatible** (pass only if the PR is labeled `golden-update` and includes regenerated goldens — the diff images are attached to the report for reviewer inspection; the *decision* remains the QM gates), **incompatible** (fail).

### 4.3 Golden update protocol

Goldens regenerate only via the dedicated CI job (never on developer machines — §9), and only in a commit that also passes all QM gates. The report archives before/after previews side-by-side; review is of the report, not of pixels on someone's laptop.

---

## 5. Performance metrics

Measured per §9's controlled environment; definitions per QUALITY_SPEC:

| Metric | Source | Aggregation |
|---|---|---|
| `T_e2e` per fixture×preset | QM-30 | median of 3 runs (discard first if cold-cache flag set) |
| per-stage wall time (all 22 stages) | tracer | median of 3 |
| `M_peak` per fixture | QM-31 | max over runs |
| per-stage peak RSS delta | tracer | max |
| output sizes | QM-32 | exact |
| tracing overhead ratio | QM-33 | paired medians |

Stage-level budgets are the ENGINE_SPEC §26 table; each stage is regression-gated independently (a stage may not hide inside another's slack).

## 6. Quality metrics

The full QM-01…QM-33 battery (QUALITY_SPEC §2–§7) is computed on every full-suite run, per fixture × preset. The benchmark harness and the production validator share the same measurement implementations (single source; the benchmark imports the validators, never re-implements them).

---

## 7. Regression detection

### 7.1 Baselines

`benchmarks/baselines/<machine-class>.json` holds, per fixture×preset×metric: baseline value, tolerance band, and the run-id that established it. Baselines change **only** by explicit reviewed commit including the report diff (never automatically).

### 7.2 Decision rules

For each metric m with baseline b and observed x:

```
Gate metric:      fail iff x outside its absolute band (QUALITY_SPEC)
Monitor metric:   fail iff x violates b ± tol(m)      (tolerances per QUALITY_SPEC)
Noise guard:      perf metrics use median-of-3 and a 2-run confirmation:
                  a perf regression must reproduce in an automatic re-run
                  before the job reports failure (guards against scheduler noise)
```

### 7.3 Drift detection (nightly)

Beyond per-run checks, the nightly job fits a linear trend over the last 30 same-dataset runs per (metric, fixture); if the trend's projected 30-day change exceeds the metric's tolerance, it opens a `benchmark-drift` issue automatically. This catches slow-boil regressions that stay inside per-run tolerance.

---

## 8. Acceptance criteria

A run is **accepted** iff all of:

1. Fixture manifest hashes verify; dataset_version matches baselines.
2. Every Gate metric within band on every fixture×preset (QUALITY_SPEC §8 table).
3. Every Monitor metric within tolerance vs baseline, after the §7.2 noise guard.
4. Golden comparison yields no "incompatible" outcome; "changed-compatible" requires the `golden-update` label + regenerated goldens in the same PR.
5. Determinism: QM-27 hash identity across the run pair and across the CI platform matrix.
6. The report file itself validates against the report schema (§11).

Any failure marks the PR status check red with the failing (metric, fixture, value, band) tuples in the status summary — the reviewer never digs through logs for the verdict.

---

## 9. CI integration and machine reproducibility

### 9.1 Environment pinning

- Benchmarks run **only** inside the versioned CI container image (`benchmarks/Dockerfile`, digest-pinned): fixed OS, glibc, Python patch version, BLAS (single-threaded, `OMP_NUM_THREADS=1`), pinned wheels via lockfile, pinned reference rasterizer binary (QM-29).
- CPU governor `performance`, turbo disabled where the runner allows; run pinned to physical cores (`taskset`), hyperthread siblings idle.
- **Machine fingerprint** recorded in every report: CPU model, core count, memory, container digest, kernel, `dataset_version`, lockfile hash.
- **Machine classes:** baselines are keyed by machine class (e.g. `ci-x86-v1`). A new runner generation gets a new class with freshly established baselines; cross-class comparison of absolute times is forbidden by the tooling (quality metrics compare across classes freely — they are hardware-independent).
- **Calibration canary:** each run first executes a fixed synthetic workload (matrix multiply + crack-trace of a stored 1 MP map); if canary time deviates > 10 % from the class's recorded canary, the run aborts as "environment unstable" rather than reporting misleading numbers.

### 9.2 Job matrix

| Job | When | Suites | Platforms |
|---|---|---|---|
| `bench-smoke` | every push | smoke | linux-x86 |
| `bench-full` | PR→main, nightly | full + golden | linux-x86 |
| `bench-determinism` | PR→main | QM-27 cross-platform | linux-x86, linux-arm64, macos-arm64 |

Only linux-x86 produces perf numbers; the other platforms verify output hashes only.

---

## 10. Historical comparison, score, leaderboard

### 10.1 History store

Every accepted run's report is appended to `benchmarks/history/` (one JSON per run, git-LFS or artifact store) and indexed by `(engine_version, git_sha, machine_class, dataset_version, timestamp)`. History is append-only.

### 10.2 Engine Score

A single scalar per run for trend lines and the leaderboard — **never a gate** (gates are §8; the score is a communication device):

```
Score = 100 · Π_d ( S_d )^{w_d}          (weighted geometric mean, ∈ (0, 100])

Dimension d        w_d   S_d definition (each ∈ (0,1])
fidelity           0.30  min(1, mean over fixtures of (QM-17 / 0.992))
geometry           0.25  min(1, 0.05 / max(QM-08, 0.05)) — smoothness, capped
printability       0.15  1 − QM-12/100 (leader ratio complement, floor 0.5)
color              0.15  min(1, 9.0 / max(QM-15, 9.0))
speed              0.10  min(1, 12 / max(T_e2e(F-photo-2), 12))
efficiency         0.05  min(1, 600 / max(M_peak(F-photo-2), 600))
```

Properties: monotone in every dimension, equals 100 when all targets are met (targets, not merely acceptable bounds), geometric mean prevents one dimension buying off another. Weights and formulas are versioned (`score_version`); scores are comparable only within a score_version.

### 10.3 Leaderboard

`benchmarks/reports/leaderboard.md`, regenerated nightly: one row per engine version (best accepted run per version, same machine class), columns = Score, the six dimension subscores, T_e2e, M_peak, delta vs previous version. Sorted by version (chronological), not by score — the leaderboard shows *history*, and the top of the file states the current release's row.

### 10.4 Historical comparison rules

- Compare only equal `(machine_class, dataset_version, score_version)` tuples; the tooling refuses otherwise.
- Any metric's full history is queryable per fixture; the nightly drift job (§7.3) consumes this store.

---

## 11. Report format

One JSON document per run, schema-versioned (`report_schema: 3`), validated in CI:

```
{
  "run_id": "...", "timestamp_utc": "...", "git_sha": "...",
  "engine_version": "...", "config_hash_per_preset": {...},
  "machine": { fingerprint fields per §9.1 },
  "dataset_version": 1, "score_version": 1,
  "suites": ["full","golden"],
  "metrics": { "<fixture>": { "<preset>": { "QM-01": {"value":0,"band":[0,0],"class":"gate","pass":true}, ... } } },
  "stages":  { "<fixture>": { "<preset>": { "<stage>": {"wall_s":..., "rss_mib":...} } } },
  "golden":  { "<fixture>/<preset>": {"svg":"identical|changed-compatible|incompatible", "ssim_lineart":..., "ssim_solved":...} },
  "score":   { "total": 97.3, "dimensions": {...} },
  "verdict": { "accepted": true, "failures": [] }
}
```

Numbers are emitted with fixed precision (metrics 6 significant digits) so report diffs are meaningful. The `failures` array carries `(metric, fixture, preset, value, band)` tuples verbatim — the same tuples shown in the PR status.

## 12. Charts

Generated per nightly run into `benchmarks/reports/charts/` (SVG, deterministic rendering — the chart files themselves are diffable):

| Chart | Content |
|---|---|
| `score-history` | Score + dimension subscores vs engine version (line) |
| `perf-stages` | stacked per-stage wall time vs version, 2 MP fixture |
| `perf-ladder` | T_e2e vs megapixels, log-log, current vs previous release |
| `memory-history` | M_peak per fixture vs version |
| `quality-grid` | small-multiple: each Gate metric vs version with its band shaded |
| `golden-diff-<f>` | side-by-side golden vs candidate previews + SSIM heatmap, only when changed |

Every chart plots the acceptance band or baseline±tolerance as a shaded region so a regression is visible without reading numbers.

## 13. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-06 | Initial complete benchmark specification. |
