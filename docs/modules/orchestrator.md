# Module Design — Orchestration Layer (`app/`)

**Status:** v1.0 — implemented (Sprint 19). Governing spec: [ARCHITECTURE.md §1.1, §4.2, §5](../ARCHITECTURE.md).

## Purpose

Wire every previously-isolated stage into one working engine, exposed through a single public function:

```python
from mysterycbn.app import convert
bundle = convert("examples/flower.jpg", preset="medium")
```

Before Sprint 19, `app/orchestrator.py`'s `Orchestrator` was an abstract class with no concrete implementation, `adapters/cli`/`adapters/api` were empty, and no code path registered any stage into `InMemoryStageRegistry` — confirmed by the Sprint 18 architecture audit (`grep -rn "InMemoryStageRegistry" src/` matched only the class's own definition; `grep -rn "def convert" src/` matched only the abstract method). The kernel's `SequentialExecutor`/`DefaultPlanResolver` (`kernel/pipeline.py`) were fully implemented but never instantiated by any caller.

This module changes none of that kernel infrastructure — it uses it.

## Files

| File | Responsibility |
|---|---|
| `app/config_defaults.py` | Layer 1 (`BUILTIN_DEFAULTS`) + layer 2 (`DIFFICULTY_PRESET`) config, and the canonical 16-slot `PIPELINE_STAGES` order. |
| `app/registry_bootstrap.py` | Constructs one instance of every concrete `Stage` and registers it into a fresh `InMemoryStageRegistry` per run. |
| `app/orchestrator.py` | The `Orchestrator` ABC (pre-existing; only its `OutputBundle` type import was corrected — see "Deviation" below). |
| `app/orchestrator_impl.py` | `ConcreteOrchestrator`, the first and only concrete subclass. |
| `app/api.py` | The `convert()` free function — the engine's sole public entry point. |

## Algorithm

1. **Read source** — `str`/`Path`/`bytes` → raw bytes → `SourceBytes` (raises `InputError` on an unreadable path).
2. **Resolve config** — `LayeredResolver.resolve()` merges `BUILTIN_DEFAULTS` → `DIFFICULTY_PRESET` (easy/medium/hard) → `PROGRAMMATIC` (caller overrides) into a `FrozenConfig` (ARCHITECTURE.md §7's five-layer resolution; `AUTO_TUNE` is not yet wired in Sprint 19 — see "Deviations").
3. **Build + resolve the plan** — `build_registry()` constructs all 16 stages (`load` → `png`); `DefaultPlanResolver.resolve()` statically validates the requires/provides chain against `["source_bytes"]`.
4. **Execute** — `SequentialExecutor.execute()` runs every stage in order, emitting `ProgressUpdate` events, checking `cancel_token.is_cancelled()` between stages, and tracing per-stage wall time via `InMemoryTracer`.
5. **Validate** (outside the Stage protocol — a validator raises, it does not provide an artifact) — `run_validation()` runs the four canonical validators (fidelity, topology, printability, palette); an unrepaired FATAL raises `QualityError`, and no `OutputBundle` is ever constructed (atomicity, ARCHITECTURE.md §11). `run_output_validity()` then checks the rendered SVG/PDF bytes (QM-26..28); a failure raises `EngineError`.
6. **Assemble `OutputBundle`** — pulls `svg`, `pdf`, `png_previews` from the context, builds a `RunReport` (resolved config, engine version, input hash, per-stage timings, the 4 validation reports), and constructs `model.reports.OutputBundle` — whose own `__post_init__` re-checks the atomicity invariants (non-empty SVG, exactly `{"lineart","solved"}` preview keys, exactly 4 validators, all passed).

## Pipeline stage order (16 slots)

```
load → preprocess → analyze → quantize → denoise → regions → merge_tiny →
topology → arcgraph → simplify → bezier → labels → legend → svg → pdf → png
```

Mapping notes against the Sprint 19 brief's stage names:
- **Contour Extraction** = `topology` + `arcgraph` (ENGINE_SPEC's own §14/§15 split — junction/arc decomposition, then face assembly + the single Φ page-scale application). No third "contours" stage was introduced.
- **Simplify** = the new `simplify` stage (see `stages/vector/simplify.md`), wired for the first time.
- **Curve Smoothing** = the existing `bezier` stage (`CurveFitStage`): G1-continuous Bézier fitting already performs the smoothing responsibility (ENGINE_SPEC §18); no separate smoothing algorithm exists in this codebase, and Sprint 19's brief is orchestration only, not new algorithm design.
- **Validation** is not a pipeline slot — it runs between step 4 and step 6, directly in `ConcreteOrchestrator.convert()` (see Algorithm above), because a validator's contract (raise vs. provide) does not fit the `Stage` protocol.

## Deviations from the pre-existing model (documented, not silent)

1. **`Orchestrator.convert()`'s return type.** `app/orchestrator.py` originally imported `OutputBundle` from `model/artifacts.py` (a structural `Protocol` with field `previews_png`, no atomicity checks). `model/reports.py` separately defines a concrete `OutputBundle` dataclass (field `previews`, full atomicity `__post_init__`) — the only one that is ever actually constructible with the invariants ARCHITECTURE.md §11 requires. The abstract method's import was corrected to point at the concrete class; this is a one-line type-annotation fix on an unimplemented abstract method, not a redesign of either model class.
2. **`AUTO_TUNE` config layer is not yet wired.** `AnalyzeStage` still runs and produces an `auto_tune` artifact (`AutoTuneProposal`), but `ConcreteOrchestrator` does not feed it back into a second config resolution pass. This is a known, narrow gap (see Compliance Report) — implementing it is pure plumbing, no new algorithm, and is left as a follow-up so Sprint 19 stays scoped to "orchestration only."

## Rejected alternatives

- **Running everything through `SequentialExecutor` alone**, including validation and bundle assembly as fake "stages": rejected because `run_validation` raises `QualityError` instead of providing an artifact (does not fit the Stage protocol's `provides` contract), and final bundle assembly needs to combine several artifacts into one atomic dataclass with its own invariant checks — forcing that through `ctx.put()` would just relocate the same logic behind a leakier abstraction.
- **Orchestrator calling every stage's `.run()` directly, bypassing the kernel executor**: rejected — it would discard the cache/tracer/progress/cancellation infrastructure that `SequentialExecutor` already implements and that no prior caller had ever exercised; Sprint 19's brief is to *use* existing modules, and the executor was existing, tested-in-isolation-but-never-invoked infrastructure.

## Quality requirements

- **Determinism (I2)** — same input + seed ⇒ byte-identical SVG. Verified: `test_convert_is_deterministic_given_the_same_seed`.
- **Atomicity** — a failed `convert()` never returns a partial bundle; every failure path raises before `OutputBundle(...)` is reached. Verified: `test_output_bundle_is_atomic_and_never_partially_constructed`, `test_convert_raises_input_error_for_a_missing_file`, `test_convert_raises_engine_error_for_garbage_bytes`.
- **Progress monotonicity** — fractions strictly non-decreasing from 0.0 (first `STAGE_STARTED`) to 1.0 (last `STAGE_FINISHED`). Verified: `test_convert_emits_progress_from_zero_to_one_across_every_stage`.
- **Cancellation** — cooperative, checked between every stage; a token flipped mid-run stops before any subsequent stage executes. Verified: `test_convert_stops_mid_pipeline_on_cancellation`.
- **Preset coverage** — `easy`/`medium`/`hard` all resolve and validate. Verified: `test_convert_runs_under_every_difficulty_preset`.

Full coverage: `tests/integration/test_convert_end_to_end.py` (15 tests, first end-to-end tests in the repository).

## Configuration

| Key | Type | Default | Range | Notes |
|---|---|---|---|---|
| `quality.d_min_mm` | float | 3.5 (medium) | 2.5–5.0 by preset | Printability floor, threaded to `DenoiseStage`, `MergeTinyStage`, and `ValidationSettings`. |
| `quantize.n_colors` | int | 16 (medium) | 8–24 by preset | |
| `page.{width_mm,height_mm,margin_mm}` | float | US Letter (215.9 × 279.4, 12.7 margin) | — | Single source of truth threaded to `ArcGraphStage`, `LegendStage`, `SvgExportStage`, `PdfExportStage`, `PngPreviewStage` — avoids the class of bug where two stages disagree on page geometry. |

## Future improvements

- Wire the `AUTO_TUNE` config layer (analyze stage's proposals currently computed but discarded).
- `adapters/cli`/`adapters/api` remain empty; `convert()` is the library-level entry point Sprint 19 scoped in — CLI/HTTP wrapping is future work, not attempted here.
- `edge_snap`/`split_large` (documented optional plugins, ARCHITECTURE.md §15) remain unimplemented — out of scope, unchanged from prior sprints.
