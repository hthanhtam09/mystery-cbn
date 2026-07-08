# ADR-001: Sprint 19 Orchestration Layer — Architecture Compliance Report

**Status:** Accepted. **Date:** Sprint 19. **Supersedes:** none (first ADR in this repository).

## Context

The Sprint 18 architecture audit established, with file:line evidence, that the engine had no working end-to-end path from an input image to an `OutputBundle`. Every algorithmic stage (raster, graph, vector, layout, validation, SVG/PDF rendering) was implemented and unit-tested in isolation, but:

- `app/orchestrator.py`'s `Orchestrator` was an `ABC` with a single `@abstractmethod` and zero concrete subclasses anywhere in the repository.
- `adapters/cli/__init__.py` and `adapters/api/__init__.py` were docstring-only, zero lines of executable code.
- No code path registered any concrete `Stage` class into `InMemoryStageRegistry` outside of unit tests using fake stages.
- `kernel/pipeline.py`'s `SequentialExecutor`/`DefaultPlanResolver` — both fully implemented — were never instantiated by any caller or test.
- `render/png.py` did not exist; `stages/vector/simplify.py`, `stages/layout/legend.py` did not exist, despite `SvgExportStage`/`PdfExportStage` already declaring `legend` as a required input artifact.

Sprint 19's mandate: "Transform the existing modules into a fully functioning engine... Implement only the orchestration layer... DO NOT redesign existing modules."

## Decision

Implement the orchestration layer as five new modules under `app/`, plus three minimal new stages to fill contract gaps that blocked the pipeline (`simplify`, `legend`, `render/png`), wiring everything through the pre-existing `SequentialExecutor`/`DefaultPlanResolver` kernel infrastructure rather than writing a parallel execution path.

## What was implemented

| File | New/Changed | Purpose |
|---|---|---|
| `app/config_defaults.py` | New | Built-in config defaults + easy/medium/hard presets; the canonical 16-stage pipeline order. |
| `app/registry_bootstrap.py` | New | Registers all 16 concrete stages into a fresh `InMemoryStageRegistry` per run. |
| `app/orchestrator_impl.py` | New | `ConcreteOrchestrator` — the first concrete `Orchestrator` subclass. Runs the kernel executor, then validation, then bundle assembly. |
| `app/api.py` | New | The `convert()` free function — the engine's sole public entry point. |
| `app/orchestrator.py` | Changed (1 import) | `Orchestrator.convert()`'s return type corrected from `model.artifacts.OutputBundle` (Protocol, field `previews_png`, no invariant checks) to `model.reports.OutputBundle` (concrete dataclass, field `previews`, full atomicity `__post_init__`) — the only one of the two ever actually constructible per DATA_MODEL_SPEC §19's invariants. |
| `stages/vector/simplify.py` | New | Thin `Stage` wrapper around the pre-existing, previously-unwired `DefaultGeometryKernel.simplify_polyline`. |
| `stages/layout/legend.py` | New | Identity-permutation chip-band layout; the `Legend` artifact `SvgExportStage`/`PdfExportStage` already required but nothing produced. |
| `render/png.py` | New | Line-art + solved PNG previews (ENGINE_SPEC §24), independent of `render/pdf.py`'s PDF-rasterization convenience function. |
| `tests/integration/test_convert_end_to_end.py` | New | 15 tests: golden path, validation embedding, reproducibility record, presets, determinism, progress, cancellation, error propagation, atomicity. |

**No existing stage, validator, or renderer's algorithm was modified.** The only change to a previously-implemented file is the single import correction in `app/orchestrator.py` described above.

## Verification

```
$ python -m pytest -q
299 passed in 8.70s          # 284 pre-existing + 15 new integration tests, zero regressions

$ python -m mypy
Success: no issues found in 84 source files    # strict mode, zero errors

$ python -m ruff check src tests benchmarks
All checks passed!

$ python -m ruff format --check src tests benchmarks
163 files already formatted

$ python3 -c "from mysterycbn.app import convert; b = convert('...', preset='medium'); ..."
svg: 2130 bytes   pdf: 41591 bytes   previews: {'lineart': 10280, 'solved': 9977}
all validators passed: True
```

`convert()` was exercised live against real PNG bytes (a 64×64 two-tone image and a 96×96 gradient), producing a valid `OutputBundle` with all four canonical validators (fidelity, topology, printability, palette) passing, in every one of the three difficulty presets.

## Compliance status

### Fully compliant with the Sprint 19 brief

- ✅ `convert()` is the engine's one public entry point (`app/api.py`), matching the exact call signature in the brief: `convert("examples/flower.jpg", preset="medium")`.
- ✅ Returns an `OutputBundle` containing SVG, PDF, PNG line-art preview, PNG solved preview, `RunReport`, and per-stage timing metrics.
- ✅ All 16 declared pipeline stages run in the declared order (Load → ... → PNG), verified by `stage_timings_s` containing every stage name after a real run.
- ✅ Engine (kernel `SequentialExecutor`), Orchestrator (`ConcreteOrchestrator`), Execution Plan (`DefaultPlanResolver.resolve()`), Progress Events (`ProgressListener`/`ProgressUpdate`, verified monotonic 0→1), Cancellation (`CancelToken`, verified mid-pipeline stop), and Error Propagation (`EngineError` hierarchy, verified for input/config/quality failures) are all implemented and exercised by real, non-mocked tests.
- ✅ No existing module's algorithm was redesigned; every stage constructor/contract used exactly as it already existed.

### Deviations (each justified above, none silent)

1. **`Orchestrator.convert()`'s return-type import corrected.** A pre-existing inconsistency between two same-named `OutputBundle` types (`model.artifacts` Protocol vs. `model.reports` concrete class) meant the abstract method's own declared return type was never satisfiable by any real implementation. This is a one-line type-annotation fix on code that had zero concrete subclasses before Sprint 19 — not a redesign of either model class, both of which are unchanged.
2. **"Curve Smoothing" has no standalone stage.** The Sprint 19 pipeline diagram lists it as a distinct step; ENGINE_SPEC's own module numbering, however, only ever defined Bézier fitting (§18) as the smoothing mechanism, and no separate smoothing algorithm exists anywhere in the codebase (confirmed absent by the Sprint 18 audit). The existing `bezier` stage (`CurveFitStage`) is reused for this pipeline slot — introducing a new smoothing algorithm was out of scope for an orchestration-only sprint.
3. **`palette_order` (mystery-number shuffle) is not implemented.** `stages/layout/legend.py` uses the identity permutation only. ENGINE_SPEC §20's Spearman-rank-constrained shuffle is a new algorithm, not existing code to wire in, and was therefore out of scope. `Legend`'s permutation is isolated behind one expression in `build_legend()` so this is a contained follow-up.
4. **`AUTO_TUNE` config layer is not fed back.** `AnalyzeStage` still produces its `auto_tune` proposal artifact, but `ConcreteOrchestrator` does not run a second config-resolution pass to apply it. Pure plumbing, deferred as a named follow-up rather than attempted under time pressure within this sprint's scope.
5. **`adapters/cli`/`adapters/api` remain empty.** Sprint 19's goal was explicitly the library-level `convert()` function ("This becomes the only entry point for the engine"); CLI/HTTP wrapping was not requested and was not attempted.
6. **Two pre-existing `lint-imports` layer-graph violations remain** (`validate.output_validity → render`, `validate.printability → stages`), predating Sprint 19 and outside this sprint's "do not redesign existing modules" mandate. A third violation this sprint's own new code would have introduced (`render.png → validate.common`, from reusing a flattening helper) was caught during verification and fixed by duplicating the ~15-line flattening primitive locally in `render/png.py` rather than crossing the layer boundary — `lint-imports` after this fix reports the same 2 pre-existing violations and zero new ones.

## Consequences

- The engine now has exactly one code path from bytes-in to `OutputBundle`-out, and it is the same path `benchmarks/framework/pipeline.py` could be pointed at in a future increment to benchmark the true raster-to-render pipeline (that harness currently starts mid-pipeline from a synthetic `LabelMap`, per the Sprint 18 audit — unchanged by this sprint, since touching it was out of scope).
- Future CLI/HTTP adapters have a stable, tested library function to wrap (`app.api.convert`) rather than needing to re-derive pipeline wiring themselves.
- The `palette_order` and `AUTO_TUNE` follow-ups are now clearly scoped, isolated, and documented (see `docs/modules/legend.md` and this ADR) rather than being an undocumented silent gap.
