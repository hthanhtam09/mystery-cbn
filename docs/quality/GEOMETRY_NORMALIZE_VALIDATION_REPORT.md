# Geometry Normalize — Validation Report (Sprint 36C)

**Scope.** Full validation pass over the `geometry_normalize` stage (Passes 1–3: Duplicate
Point Cleanup, Spike Removal, Minimum Gap Enforcement / Gap Repair) and the corresponding
removal of the obsolete face-aware pinch-repair logic from `stages/vector/curves.py`
(Sprint 36B.1–36B.4). No production code was modified in this sprint — this is a report on
the state left by 36A–36B.

**Files in scope**: `src/mysterycbn/stages/vector/geometry_normalize.py`,
`src/mysterycbn/stages/vector/curves.py`, `src/mysterycbn/app/config_defaults.py`,
`src/mysterycbn/app/registry_bootstrap.py`, plus their test suites (`tests/unit/`,
`tests/property/`, `tests/golden/`, `benchmarks/perf/`).

---

## 1. Test results

| Tier | Command | Result |
|---|---|---|
| Unit | `pytest tests/unit -q` | **321 passed** |
| Property (Hypothesis) | `pytest tests/property -q` | **28 passed** |
| Golden (unit-level digests) | `pytest tests/golden -q` | **19 passed** |
| Contracts + integration | `pytest tests/contracts tests/integration -q` | **19 passed** |
| Quality benchmarks | `pytest benchmarks/quality -q` | **7 passed** |
| Performance benchmarks (this stage) | `pytest benchmarks/perf/test_geometry_normalize_perf.py -q` | **2 passed** |
| **Full suite** | `pytest -q` | **387 passed, 0 failed** |

No skips, no xfails, no flaky reruns observed across three consecutive full-suite runs.

## 2. Golden dataset (Sprint 21 harness)

Ran the blessed multi-category dataset comparison (`python -m benchmarks.golden.cli run
--suite golden`) against the current engine with `geometry_normalize` live in the default
pipeline:

```
golden run 935b9b2ad6b1: ACCEPTED
{'total': 8, 'passed': 8, 'failed': 0}
```

All 8 blessed fixtures (`animals`, `flowers`, `people`, `architecture`, `cartoons`, `food`,
`landscape`, `vehicles`) produced **byte-identical SVG output** against their blessed
baseline. This means the three `geometry_normalize` passes are either true no-ops on this
dataset's geometry or converge to output identical to the pre-Sprint-36 pipeline — no
observed drift on the standing golden corpus.

## 3. Determinism (I2)

Verified directly (not just via the golden harness) on two independent fixtures:

- A synthetic two-tone image: two `convert()` calls on byte-identical input produced
  byte-identical SVG output (`bundle1.svg == bundle2.svg`).
- A structured fixture with a thin feature (a 1px-wide stripe crossing a filled block,
  chosen specifically to make `geometry_normalize` do real work, not pass through): same
  result — byte-identical SVG across repeated runs.

Additionally, the `geometry_normalize` stage's own test suite includes dedicated
determinism/idempotence property tests (Hypothesis, 100 examples each) for all three
passes: same-input-same-output, order-independence of arc-list iteration, and idempotence
(a second normalization pass on already-normalized output makes no further change). All
pass.

## 4. Topology (I3), Fidelity (I1), Printability (I4)

Ran the four canonical validators live via `convert()` on multiple fixtures and presets:

| Fixture | Preset | fidelity | topology | printability | palette |
|---|---|---|---|---|---|
| Two-tone blob | medium | ✅ | ✅ | ✅ | ✅ |
| Thin-stripe structured fixture | easy | ✅ | ✅ | ✅ | ✅ |
| Thin-stripe structured fixture | medium | ✅ | ✅ | ✅ | ✅ |
| Thin-stripe structured fixture | hard | ✅ | ✅ | ✅ | ✅ |

All four validators pass on every combination tested, including the case designed to
exercise `geometry_normalize`'s actual repair logic (confirmed via
`stage_timings_s["geometry_normalize"]` being nonzero and comparable in cost to sibling
stages — ~0.23–0.30s on a 96×96 fixture).

**One pre-existing failure mode identified and confirmed unrelated to this work**: a
96×96 uniform-per-pixel-random-noise image fails the `printability` gate with an
unrepairable FATAL. This was reproduced **identically with `geometry_normalize.enabled`
set to both `True` and `False`** via `overrides={"geometry_normalize": {"enabled": False}}`
— the failure is caused by the noise image producing thousands of single-pixel regions
that cannot clear the printability floor regardless of geometry normalization, not by any
Sprint 36 change. Documented here as a known engine limitation on adversarial/non-photographic
input, not a regression.

## 5. Performance

`benchmarks/perf/test_geometry_normalize_perf.py` (pytest-benchmark), matching the
complexity split named in `GAP_REPAIR_DESIGN.md` §7:

| Benchmark | Arcs | Mean | Notes |
|---|---|---|---|
| Sparse (low boundary density) | 2000 | ~165–170 ms | Matches the engine's O(A) broad-phase assumption |
| Dense parallel (worst-case clustering) | 300 | ~1.30–1.32 s | Confirms the named O(A²) worst case is real and measurable at far fewer arcs |

No fixed pass/fail budget is asserted yet (per the design doc's own note that a budget
should be set from real-fixture arc-count measurement, not asserted a priori) — both
benchmarks currently serve as tracked timing baselines for future regression detection.
The ~8× per-arc cost multiplier between sparse and dense confirms the density-sensitivity
risk was correctly anticipated at design time, not discovered as a surprise here.

Stage cost in a full `convert()` run (96×96 fixture, few hundred arcs): `geometry_normalize`
contributes ~0.23–0.30s, in the same order of magnitude as sibling vector-stage costs on
comparable inputs.

## 6. Quality metrics

- **Golden dataset**: 8/8 byte-identical (§2).
- **Determinism**: 0 observed non-determinism across all repeated-run checks (§3).
- **Validator pass rate on non-adversarial fixtures**: 4/4 validators × 4/4 (fixture,
  preset) combinations tested = 100%.
- **mypy (strict)**: `src/` as a whole is clean except for `geometry_normalize.py`, which
  has **3 errors** (see §7) — every other module in the 86-file source tree passes strict
  mypy with zero errors, isolating the type-check debt entirely to this sprint's own file.
- **ruff**: `geometry_normalize.py`/`curves.py` production code is clean; the test files
  added in Sprint 36B (`test_geometry_normalize.py`,
  `test_geometry_normalize_properties.py`) have 18 lint findings, entirely
  style/unused-variable class (line length, unused unpacked tuple elements from
  `_minimum_gap_enforcement`'s `(out, repaired)` return, one unsorted import block) — none
  affect correctness or test validity (confirmed by the 100% pass rate above).
- **import-linter**: 0 new architecture-layer violations. The 2 pre-existing violations
  (`validate.output_validity → render`, `validate.printability → stages`) are unchanged
  from before Sprint 36 and are the same ones ADR-001 already documents as out-of-scope,
  predating this work.

## 7. Regressions

**None found in behavior.** Full suite (387 tests), golden dataset (8/8), and live
validator checks all pass; the noise-image printability failure is confirmed pre-existing.

**Type-check debt introduced, not yet fixed** (production-code quality regression, not a
behavioral one):

| # | Location | mypy error | Assessment |
|---|---|---|---|
| 1 | `geometry_normalize.py:203` | `Returning Any from function declared to return "ndarray[...]"` | In `_clean_duplicate_points`'s return path; numpy's fancy-indexing return type isn't narrowed. Cosmetic — the runtime type is correct, mypy just can't prove it without an explicit cast. |
| 2 | `geometry_normalize.py:757` | `Argument 1 to "_candidate_pairs" has incompatible type "list[Arc]"; expected "tuple[Arc, ...]"` | `_minimum_gap_enforcement` builds a mutable `list[Arc]` (`arc_list`) for in-place repair but passes it where `_candidate_pairs` declares a `tuple[Arc, ...]` parameter. Functionally harmless (both are iterables of `Arc` and the function only reads), but the signature should accept `Sequence[Arc]` or the call site should pass a tuple. |
| 3 | `geometry_normalize.py:900` | `Argument 2 to "put" of "PipelineContext" has incompatible type "dict[str, int]"; expected "Artifact"` | `GeometryNormalizeStage.run` binds the `geometry_normalize_metrics` dict via `ctx.put(...)`, but `PipelineContext.put`'s declared parameter type is the `Artifact` protocol, which the plain metrics dict does not structurally satisfy. Runtime behavior is unaffected (the in-memory context doesn't enforce the protocol), but this is the one place this stage's design (a bare-dict metrics artifact, chosen in Sprint 36A.4 to avoid inventing a new artifact type) is in tension with the kernel's typed contract. |

None of these three errors are new architecture violations or correctness bugs — they are
narrow, mechanical type-annotation gaps, isolated to one file, with zero observed runtime
impact across 387 passing tests. Recorded here as debt to close in a follow-up, not fixed
in this validation-only sprint per the "no production code" instruction.

## 8. Remaining risks

1. **Gap Repair's success envelope is narrower than a first read of the design doc
   suggests.** During implementation (Sprint 36B.3), hand-verification showed the
   single-witness + fixed-taper mechanism reliably repairs a "clean pinch" (witness at an
   *existing* vertex on both arcs, neighbors far beyond threshold) but frequently — and, in
   one now-fixed case, non-monotonically due to a floating-point bug — fails to clear a
   gap when vertex insertion is required on one side, or when the gap is sustained across
   many vertices (dense parallel runs). This is by design a *skip*, not a corruption (every
   skip is independently verified before commit, and `StageError` fires if a committed
   repair somehow still fails), so it is safe — but it means a meaningful fraction of
   real-world thin-feature pinches may go unrepaired by Pass 3 and instead surface
   downstream as an I3/I4 validator concern rather than being fixed at the source. No
   golden-dataset fixture currently exercises this path with a *repaired* outcome other
   than the one hand-constructed golden test (`test_gap_repair_golden.py`, a synthetic pair
   appended to real fixture data) — the real 8-fixture golden corpus shows 0 detected gaps
   requiring repair, so this risk is currently untested against organic image data.
2. **No fixed performance budget set for Gap Repair.** §5's benchmarks are tracked but not
   gated; the dense-parallel worst case (~1.3s at only 300 arcs) could become a real
   bottleneck on production-scale images with many thin, closely-packed features (the
   exact content category — ropes, grass, hair — this whole sprint arc was motivated by).
   A budget should be set once real fixture arc-counts in that content class are measured.
3. **Type-check debt (§7)** should be closed before the next sprint touches this file
   further, to avoid compounding on top of it.
4. **Pass 1 and Pass 2's own dedicated per-pass design documents remain unwritten** — both
   are implemented and tested, but `docs/modules/geometry_normalize.md`'s own §16 names
   this as an explicit tracked gap (mirroring Gap Repair's `GAP_REPAIR_DESIGN.md`), not yet
   closed.
5. **The `PIPELINE_STAGES`/`ArcGraph` artifact-chain docs (ENGINE_SPEC.md, ARCHITECTURE.md)
   were updated for the stage's insertion (Sprint 36A.5) but ENGINE_SPEC's own `§`-numbered
   module sections were deliberately *not* renumbered** to avoid a large, unrelated diff;
   the stage appears in the pipeline diagram unnumbered. This is intentional and documented
   at the time, but is worth flagging as a standing inconsistency for whoever eventually
   does a full spec renumbering pass.
