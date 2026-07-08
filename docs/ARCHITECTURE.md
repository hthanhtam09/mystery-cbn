# Mystery Color-by-Number Engine — Architecture Specification

**Status:** v2.0 — authoritative. Interface changes require updating this document first.
**Horizon:** designed for a 10-year maintenance life. Every decision below is annotated with what it protects against.

---

## 0. Product definition and invariants

A deterministic **image → vector converter**: raster in, printable mystery color-by-number page out (numbered closed regions, legend, preview). It converts; it never redraws, hallucinates, or recomposes.

Hard invariants — every release must satisfy all four; they are enforced by the validation subsystem (§11), not by convention:

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| I1 | **Fidelity** — every output region maps to a connected pixel set of the quantized input; composition preserved | provenance check: region ↔ label-map correspondence audit |
| I2 | **Determinism** — same input + config ⇒ byte-identical SVG | seeded RNG everywhere; SVG hash test in CI |
| I3 | **Topological validity** — regions form a watertight planar partition (no gaps, overlaps, self-intersections) | arc-graph construction guarantees it; geometry validator re-proves it |
| I4 | **Printability** — every region physically colorable (min inscribed diameter in mm), every number readable | printability validator with auto-repair or hard fail |

## 1. Overall architecture

### 1.1 Layered hexagonal core

```
┌─────────────────────────────────────────────────────────────┐
│  ADAPTERS (thin, replaceable)                                │
│  CLI · FastAPI · batch worker · future GUI                   │
├─────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER                                           │
│  Orchestrator: job lifecycle, presets, progress, cancellation│
├─────────────────────────────────────────────────────────────┤
│  DOMAIN CORE (the engine — pure library, no I/O framework)   │
│  Pipeline kernel · Stage registry · Data model · Validators  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────┐ │
│  │ raster     │ │ graph     │ │ vector    │ │ layout       │ │
│  │ stages     │ │ stages    │ │ stages    │ │ stages       │ │
│  └───────────┘ └───────────┘ └───────────┘ └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  FOUNDATION                                                  │
│  geometry kernel · color science · units · config · logging  │
│  errors · tracing · plugin loader                             │
└─────────────────────────────────────────────────────────────┘
```

Rules that keep this maintainable for a decade:

- **Dependency direction is strictly downward.** Foundation knows nothing about stages; the domain core knows nothing about FastAPI. A web-framework migration (FastAPI will not be the fashion in 2036) touches only the adapter layer.
- **The engine is a library first.** `convert(source, config) → OutputBundle` is the only entry point adapters use. Anything an adapter can do, a test can do.
- **Third-party isolation.** OpenCV, Shapely, scikit-image are *implementation details of individual modules*, never part of any interface. Interfaces speak NumPy arrays, plain dataclasses, and the engine's own geometry types. Rationale: over 10 years, libraries get abandoned (see PIL → Pillow, pyproj rewrites); swapping OpenCV's k-means for a Rust extension must be a one-module change.
- **Data model is the contract.** Stages never call each other; they communicate only through typed artifacts on the pipeline context. This is what makes every stage replaceable.

### 1.2 Three processing domains

The pipeline crosses two irreversible representation boundaries:

1. **Raster domain** — dense arrays, working-resolution pixels.
2. **Graph domain** — discrete regions + adjacency (the label raster is still authoritative geometry).
3. **Vector domain** — arc-graph topology, Bézier curves, physical units (pt/mm). *After entry into this domain, no stage may consult the raster.*

Each boundary crossing is a named, validated conversion stage — the two riskiest points in the system get the heaviest testing.

## 2. Folder structure

```
mystery-cbn/
├── docs/
│   ├── ARCHITECTURE.md            ← this file
│   ├── adr/                       ← Architecture Decision Records, numbered, append-only
│   ├── modules/                   ← one design doc per module (required before code)
│   └── quality/                   ← print-quality standards, golden criteria
├── src/mysterycbn/
│   ├── foundation/
│   │   ├── geometry/              ← crack tracing, arc graph, bezier, polylabel, predicates
│   │   ├── color/                 ← sRGB↔LAB, ΔE (76/2000), colorfulness metrics
│   │   ├── units.py               ← mm/pt/px conversions; the ONLY place units convert
│   │   ├── config/                ← schema, presets, layering, migration
│   │   ├── errors.py  logging.py  tracing.py
│   │   └── plugins.py             ← discovery + registry (§8)
│   ├── model/                     ← PipelineContext, artifacts, Palette, RegionGraph, ArcGraph, CurveSet
│   ├── kernel/                    ← Pipeline, Stage protocol, registry, scheduler, cancellation
│   ├── stages/
│   │   ├── raster/                ← load, preprocess, analyze, quantize, denoise, edge_snap
│   │   ├── graph/                 ← regions, merge_tiny, split_large
│   │   ├── vector/                ← contours, simplify, smooth
│   │   └── layout/                ← labels, palette_order, legend
│   ├── validate/                  ← fidelity, topology, printability, palette validators
│   ├── render/                    ← svg, pdf, png backends behind one Renderer interface
│   ├── app/                       ← Orchestrator, JobSpec, OutputBundle, progress API
│   └── adapters/
│       ├── cli/    api/           ← FastAPI app + schemas (versioned, /v1)
├── tests/
│   ├── unit/  property/  golden/  integration/  contracts/
├── benchmarks/                    ← per-stage + end-to-end suites, baselines/
├── assets/                        ← test fixtures, fonts, ICC profiles
└── plugins/                       ← first-party optional plugins (proof the plugin API works)
```

Conventions with teeth: import-linter (or equivalent) CI rule enforces the layer graph; a module may import only from its own layer and below. `docs/adr/` records every reversal-worthy decision — the 2031 maintainer must be able to learn *why*, not just *what*.

## 3. Module dependency graph

```
adapters/cli ──┐
adapters/api ──┼──▶ app/orchestrator ──▶ kernel ──▶ model ──▶ foundation
plugins ───────┘         │                 ▲
                         │                 │ (registry lookup only)
                         ▼                 │
                     render ──────▶ stages/* ──▶ model, foundation
                         │
                         ▼
                     validate ──▶ model, foundation
```

- `foundation` depends on nothing internal (only NumPy-level third parties).
- `model` depends only on `foundation`.
- `kernel` depends on `model` — it schedules stages but knows no concrete stage.
- Concrete `stages/*` register themselves; the kernel discovers them via the registry. **No stage imports another stage.** Cross-stage sharing goes through `foundation` (e.g., both quantize and merge use `foundation/color.delta_e`).
- `validate` and `render` are siblings of stages, orchestrated by `app`.
- Cycles are structurally impossible; CI fails on any upward or sideways import.

## 4. Data flow

### 4.1 Artifact chain

```
SourceImage (file/bytes)
  → RasterImage        H×W×3 float32 sRGB [0,1] + provenance (source hash, ICC, EXIF applied)
  → RasterImage'       working resolution + work_scale; edge-preserved
  → ImageStats         colorfulness, edge density, luminance histogram
  → LabelMap + Palette H×W int32 palette indices; Palette in LAB (authoritative) + sRGB
  → RegionGraph        nodes: RegionRecord(id, label, area_px, bbox, seed_px)
                       edges: adjacency + shared boundary length + ΔE
  → ArcGraph           nodes: junction points (≥3 labels meet, or border corners)
                       arcs: polylines with (left_label, right_label); faces: arc walks
  → CurveSet           per-arc Bézier chains, physical units; faces unchanged
  → LabelPlan          per region: anchor, font size, optional leader line
  → LegendPlan         palette order, chip layout, renumbering map
  → OutputBundle       svg bytes, pdf bytes, preview pngs, run report (timings, warnings, metrics)
```

Rules: artifacts are **immutable once produced** (a stage replaces, never mutates in place — enables caching, diffing, and debug snapshots). Every artifact carries a `provenance` field naming the stage and config hash that produced it. Units policy: raster domain in working px; the ArcGraph conversion applies `work_scale` exactly once; everything after is in points. Only `foundation/units.py` may convert.

### 4.2 Control flow

Orchestrator builds a `Plan` (ordered stage list resolved from config + registry), validates the requires/provides chain statically, then executes. Between stages it checks a cancellation token and emits progress events (`stage_started/finished`, fraction complete) — this is what the API's job-status endpoint and a future GUI progress bar both consume.

## 5. Public interfaces (stable — semver-governed)

These are the only surfaces external consumers may touch. Everything else can change in a minor release.

1. **Library API** — `convert(source, config, *, on_progress, cancel_token) → OutputBundle`; `EngineConfig` and its presets; `OutputBundle`; the exception hierarchy. This is the primary contract.
2. **CLI** — `mysterycbn convert INPUT -o DIR --preset medium --set quantize.n_colors=20`; exit codes are contractual (0 ok, 2 input error, 3 config error, 4 quality-gate failure, 5 internal).
3. **HTTP API** — versioned under `/v1`: submit job, poll status/progress, fetch artifacts. Async job model from day one (conversions take seconds to minutes; a synchronous endpoint would be a trap we'd support forever).
4. **Plugin API** (§8) — the Stage/Renderer/Validator protocols and registration entry points.
5. **Config schema** — the on-disk config document format, versioned with `schema_version` and migrated forward automatically (§7).

Deprecation policy: public symbols are removed only after one minor release of runtime deprecation warnings; the HTTP API keeps `/v1` alive until `/v2` has feature parity plus a migration guide.

## 6. Internal interfaces

Internal contracts are protocol-typed and narrow:

- **Stage protocol** — identity (`name`, `version`), declared `requires`/`provides` (artifact names), `config_section` it reads, and `run(ctx)`. Stages must be: deterministic given (artifacts, config, seed); side-effect-free outside the context; single-purpose.
- **Renderer protocol** — consumes `(CurveSet, LabelPlan, LegendPlan, PageConfig)`, produces bytes + media type. SVG is the canonical renderer; PDF and PNG may derive from it or render natively, but all three must agree geometrically (contract-tested).
- **Validator protocol** — consumes the context, returns a structured `ValidationReport` (findings with severity, location, auto-repair applied?). Validators never mutate artifacts except through declared, logged repairs.
- **Geometry kernel API** — pure functions over its own types (Point, Polyline, Arc, Face). No stage does raw computational geometry; it calls the kernel. This concentrates the hardest, most bug-prone math where property tests hammer it.
- **Color science API** — likewise: all color conversion and difference math in one place, one implementation.

Internal interface changes require: update the module design doc, update this file if layering changes, add a contract test.

## 7. Configuration system

Five-layer resolution, later wins, fully recorded:

```
built-in defaults → difficulty preset → user config file → programmatic overrides → auto-tuning proposals*
```

\* Auto-tuning (from the analyze stage, later from AI advisors §14) is special: it may only fill values the user did **not** explicitly set. Explicit human intent always wins.

Properties:

- **Typed and frozen.** One schema tree (pydantic today — but the schema is the contract, the library an implementation detail). Validation at construction; cross-field rules (margins vs page size, font size vs min region) enforced in the schema, not scattered through stages.
- **Physically meaningful units.** Quality knobs are in mm, pt, ΔE — never raw pixels — so configs survive resolution changes.
- **Versioned + migratable.** Config documents carry `schema_version`; loading an old version applies ordered migrations. A 2028 config must load in 2033.
- **Reproducibility record.** Every OutputBundle embeds the *fully resolved* config + engine version + input hash. Any output ever produced can be regenerated exactly.
- **Per-stage sections.** A stage sees only its own section — no stage reads another stage's knobs (prevents hidden coupling).

## 8. Plugin architecture

Everything variable is a plugin; the kernel ships empty and the built-in stages are simply plugins that happen to live in-tree. Extension points:

| Extension point | Contract | Example third-party use |
|---|---|---|
| Stage implementation | Stage protocol + artifact types | alternative quantizer (octree, neural palette) |
| Renderer | Renderer protocol | EPS export, laser-cutter output |
| Validator | Validator protocol | publisher-specific print rules |
| Difficulty preset | config fragment | brand-specific "toddler" preset |
| Analyzer/advisor | proposes config overrides | AI content-aware tuning (§14) |

Mechanics: discovery via Python entry points (`mysterycbn.plugins` group) plus explicit registration for embedded use. Each plugin declares `api_version`; the loader refuses incompatible plugins with a clear message rather than failing mid-pipeline. Selection is by configuration (`stages.quantize.impl = "octree"`), so swapping implementations requires zero engine changes.

Stability promise: the plugin API is public interface #4 — protocols and artifact schemas are semver-governed. What is *not* promised: artifact internals beyond the documented schema, stage ordering internals, foundation internals.

First-party proof: `split_large` and `edge_snap` (optional quality stages) ship as plugins from day one, guaranteeing the extension point actually works because we depend on it ourselves.

## 9. Benchmark framework

Quality regressions are worse than speed regressions for this product, so the framework tracks both:

- **Performance suites** (`benchmarks/perf/`): per-stage and end-to-end wall time + peak RSS on a fixed fixture ladder (0.5 / 2 / 12 / 24 MP; photo, illustration, flat-art, high-noise). Results stored as JSON with machine fingerprint; CI compares against committed baselines with a tolerance band (fail > 20% regression on any stage).
- **Quality suites** (`benchmarks/quality/`): scalar metrics per fixture — region count, mean region compactness, boundary smoothness (curvature energy), palette ΔE spread, label fit rate, watertightness residual (must be 0), SSIM of solved-preview vs quantized source (fidelity proxy for I1). Tracked over time; a "faster" algorithm that drops quality metrics is rejected by the same CI gate.
- **Golden ledger**: each benchmark run records resolved config + engine version, so any historical number can be reproduced.
- Baselines are updated only by explicit, reviewed commits — never automatically.

## 10. Test strategy

Test pyramid, all layers required for every module (a module PR without all four is incomplete by definition):

1. **Unit tests** — synthetic inputs with analytically known ground truth (checkerboards, disks, gradients). Fast, exhaustive on edge cases.
2. **Property-based tests** (Hypothesis) — the geometry kernel and topology stages live or die here. Invariant properties: face areas sum to page area; every arc borders exactly two faces; simplification never flips arc sidedness; determinism (run twice, hash artifacts); LAB round-trip error bounds.
3. **Contract tests** — every implementation of a protocol (all quantizers, all renderers, all plugins in CI) runs one shared suite proving it honors the protocol semantics, not just the type signature. Renderer agreement (SVG vs PDF geometry) is a contract test.
4. **Golden tests** — full pipeline on the fixture ladder; rendered previews compared by perceptual diff (SSIM threshold) against versioned goldens; SVG structural diff for topology. Golden updates require a human-visible before/after in review.
5. **Integration tests** — CLI exit codes, API job lifecycle, cancellation mid-pipeline, config migration across versions.

Fixtures are original, in-repo assets (no copyrighted imagery — also a legal invariant). Mutation testing on the geometry kernel annually or after major refactors.

## 11. Error handling strategy

Single hierarchy rooted at `EngineError`:

```
EngineError
├── InputError          unreadable/unsupported/degenerate input        → CLI 2 / HTTP 400
├── ConfigError         invalid or inconsistent configuration          → CLI 3 / HTTP 422
├── StageError          stage failed; carries stage name + artifact state → CLI 5 / HTTP 500
├── QualityError        validation gate failed and repair impossible   → CLI 4 / HTTP 409
└── CancelledError      cooperative cancellation                       → CLI 130 / HTTP 499
```

Principles:

- **Fail fast on contract, degrade gracefully on quality.** Contract violations (missing artifact, bad config) abort immediately. Quality problems attempt *declared, logged auto-repairs* (merge an unprintable sliver, add a leader line); only unrepairable findings abort.
- **Errors carry context, not stack noise**: stage name, artifact provenance, config hash, and — in debug mode — a snapshot bundle (label map PNG, arc-graph dump) written to a diagnostics directory. A support case must be reproducible from the bundle alone.
- **No partial outputs.** OutputBundle is atomic: either all artifacts validated and written, or none (temp dir + rename).
- Warnings (repairs applied, auto-tune overrides) accumulate in the run report; adapters surface them, never swallow them.

## 12. Logging strategy

- **Structured logging** (JSON-capable), one logger per module namespace (`mysterycbn.stages.quantize`), with a run-scoped correlation ID stamped on every record — mandatory once the API serves concurrent jobs.
- Levels: `DEBUG` per-stage internals; `INFO` stage boundaries + timings + key metrics (region count, palette size); `WARNING` auto-repairs and auto-tune overrides; `ERROR` failures with context.
- **Tracing is separate from logging**: `foundation/tracing` collects per-stage timings, artifact sizes, and optional debug artifacts into the run report. Logging is for operators; tracing is for engineers and the benchmark harness. Neither is ever load-bearing (I2: output identical with logging disabled).
- The library never configures global logging handlers (library etiquette); adapters do.

## 13. Performance strategy

Ordered by leverage, applied only after golden tests lock quality:

1. **Algorithmic** — working-resolution decoupling (trace at ≤1600 px, render at page resolution) is the master lever; everything downstream of quantize is resolution-bounded.
2. **Vectorize first** — NumPy/OpenCV bulk operations; Python loops only in graph/geometry code where N = regions/arcs (thousands), not pixels (millions).
3. **Numba/Cython escape hatches** — reserved for proven hotspots (crack tracing, Bézier fitting), always with a pure-Python reference implementation kept for testing and as documentation. Contract: `assert fast(x) == reference(x)` in CI.
4. **Caching by content** — artifacts keyed by (input hash, upstream config hash); re-running with a changed label-placement knob must not re-run quantization. Falls out of artifact immutability (§4).
5. **Parallelism** — stage-internal only (per-arc Bézier fitting is embarrassingly parallel); the pipeline itself stays sequential and deterministic. Batch throughput scales by process, in the worker adapter, not inside the engine.
6. **Budgets, not vibes** — per-stage time budgets on the 2 MP fixture recorded in `benchmarks/baselines/`; regressions fail CI (§9). Target: 2 MP photo → full bundle in ≤ 15 s single-core; quality is allowed to spend more when configured.

## 14. Future AI integration points

AI augments; the deterministic core remains the product. Every AI touchpoint is an *advisor or preprocessor behind an existing plugin interface* — removable without trace, never able to violate I1–I4 because its output re-enters the validated pipeline:

1. **Content-aware parameter advisor** (analyzer plugin) — a vision model proposes config overrides (palette size, min region size, smoothing strength) from image content. Output: a config fragment in the auto-tune layer (§7) — user settings still win.
2. **Semantic edge prior** (raster-stage plugin) — a segmentation model (SAM-class) contributes an edge-importance map that guides quantization boundaries and edge-snapping toward object boundaries. Output: a raster prior, consumed like any gradient map — the model never draws lines directly.
3. **Saliency-weighted detail budget** (graph-stage plugin) — allocate smaller `min_region_mm` inside salient areas (faces, subjects), coarser in backgrounds. Output: a per-region weight map for merge/split decisions.
4. **Palette aesthetics advisor** (layout plugin) — reorder/renumber and tune the legend for mystery-effect (avoid the picture being guessable from the numbers) using a learned scorer. Output: a permutation — geometry untouched.
5. **Quality critic** (validator plugin) — a learned model scores outputs against a corpus of professional pages; used in benchmarking (§9) and as a soft gate. Output: findings in a ValidationReport.
6. **Difficulty estimator** — predicts completion time / age suitability for publishing metadata.

Architectural guarantees that make this safe: AI components are optional dependencies (separate extra), always behind a feature flag, always with a deterministic fallback, and their outputs are *data* (priors, weights, config fragments) — never geometry written directly to the output. Determinism policy: runs with AI advisors record the advisor's outputs in the reproducibility record, so even AI-assisted runs replay exactly.

---

## 15. Module dossier

Complexity scale: ▲ low · ▲▲ moderate · ▲▲▲ high (research-grade). Replaceability: how cheaply a from-scratch reimplementation slots in, given the interfaces above.

### Foundation

| Module | Purpose | Input → Output | Depends on | Cx | Replaceability |
|---|---|---|---|---|---|
| `foundation/geometry` | All computational geometry: crack-boundary extraction, arc-graph build, polyline ops, Bézier fitting, polylabel, robust predicates | label rasters / polylines → ArcGraph, curves, anchors | NumPy, (Shapely internally) | ▲▲▲ | **Low by design** — hardest code in the system; that's *why* it's isolated behind a pure API with property tests, so a future Rust port is a drop-in |
| `foundation/color` | Color science: sRGB↔LAB, ΔE76/ΔE2000, colorfulness | arrays → arrays/scalars | NumPy | ▲ | High — textbook math, one file |
| `foundation/units` | mm/pt/px conversion; single source of truth | scalars → scalars | — | ▲ | High |
| `foundation/config` | Schema, presets, layering, migration, resolved-config hashing | dicts/files → frozen config | pydantic (detail) | ▲▲ | Medium — schema is the contract, library swappable |
| `foundation/plugins` | Discovery, version gating, registry | entry points → registered factories | importlib.metadata | ▲ | High |
| `errors/logging/tracing` | Cross-cutting concerns (§11–12) | — | stdlib | ▲ | High |

### Model & kernel

| Module | Purpose | Input → Output | Depends on | Cx | Replaceability |
|---|---|---|---|---|---|
| `model` | Artifact types: PipelineContext, Palette, LabelMap, RegionGraph, ArcGraph, CurveSet, plans, reports | — (definitions) | foundation | ▲ | **Deliberately low** — this *is* the contract; changing it is a versioned event |
| `kernel` | Stage protocol, plan resolution, requires/provides validation, execution, timing, cancellation, progress events | Plan + context → executed context | model | ▲▲ | Medium — small and stable; rarely needs replacing |

### Raster stages

| Module | Purpose | Input → Output | Depends on | Cx | Replaceability |
|---|---|---|---|---|---|
| `load` | Decode, EXIF orient, ICC→sRGB, alpha→white, normalize | file/bytes → RasterImage | Pillow/pyvips | ▲ | High |
| `preprocess` | Working-res resize + edge-preserving smoothing (bilateral/guided), optional CLAHE | RasterImage → RasterImage' | OpenCV | ▲▲ | High — protocol-typed; guided-filter or ML denoiser slots in |
| `analyze` | Global stats → auto-tune proposals | RasterImage → ImageStats + config fragment | foundation/color | ▲ | High — AI advisor (§14.1) is its plugin successor |
| `quantize` | Perceptual palette extraction + label map (seeded LAB k-means; ΔE dedupe; coverage numbering) | RasterImage' → LabelMap + Palette | OpenCV, foundation/color | ▲▲ | High — octree/median-cut/neural alternatives behind same contract |
| `denoise` | Label-raster cleanup: modal filter, area opening | LabelMap → LabelMap | scikit-image | ▲ | High |
| `edge_snap` *(plugin, optional)* | Pull quantization boundaries onto image gradients | RasterImage' + LabelMap → LabelMap | OpenCV | ▲▲ | High |

### Graph stages

| Module | Purpose | Input → Output | Depends on | Cx | Replaceability |
|---|---|---|---|---|---|
| `regions` | Connected components + region adjacency graph with boundary lengths and ΔE edges | LabelMap → RegionGraph | scikit-image, NetworkX | ▲▲ | Medium |
| `merge_tiny` | Enforce printability floor: fold sub-minimum regions into best neighbor (priority queue by area; ΔE + shared-boundary cost) | RegionGraph → RegionGraph | foundation/color | ▲▲ | High — cost function is the knob; learned cost (§14.3) slots in |
| `split_large` *(plugin, optional)* | Split oversized flat regions along content seams for better puzzle rhythm | RegionGraph → RegionGraph | geometry | ▲▲▲ | High |

### Vector stages

| Module | Purpose | Input → Output | Depends on | Cx | Replaceability |
|---|---|---|---|---|---|
| `contours` | The domain crossing: crack-trace the label raster once into a shared-boundary ArcGraph; apply work_scale; guarantee I3 by construction | LabelMap + RegionGraph → ArcGraph | foundation/geometry | ▲▲▲ | Low-medium — the heart; replace only with equal topology guarantees |
| `simplify` | Topology-preserving polyline simplification per shared arc (Visvalingam–Whyatt with sidedness guard) | ArcGraph → ArcGraph | foundation/geometry | ▲▲ | Medium |
| `smooth` | Fit G1 cubic Bézier chains per arc; corner preservation by angle threshold; junction continuity | ArcGraph → CurveSet | foundation/geometry | ▲▲▲ | Medium — Schneider fitting today; alternatives must pass the same fit-error contract |

### Layout stages, validation, rendering

| Module | Purpose | Input → Output | Depends on | Cx | Replaceability |
|---|---|---|---|---|---|
| `labels` | Number placement: pole of inaccessibility, font sizing vs inscribed circle, leader lines for slivers, overlap resolution | CurveSet + RegionGraph → LabelPlan | geometry/polylabel | ▲▲ | High |
| `palette_order` | Legend ordering + optional renumbering for mystery effect and spatial balance | Palette + RegionGraph → LegendPlan | — | ▲ | High — AI advisor (§14.4) successor |
| `validate/*` | Prove I1–I4: fidelity audit, watertightness, printability, palette distinguishability; declared auto-repairs | context → ValidationReport | geometry, color | ▲▲ | Medium — findings schema is the contract |
| `render/svg` | Canonical output: curves, labels, legend, page furniture | plans → SVG bytes | svgwrite (detail) | ▲▲ | High |
| `render/pdf` | Print-ready PDF (trim size, 300 DPI, embedded fonts) | plans (or SVG) → PDF bytes | ReportLab/PyMuPDF | ▲▲ | High |
| `render/png` | Line-art preview + solved (colored) preview | plans → PNG bytes | pyvips/Pillow | ▲ | High |

### Application & adapters

| Module | Purpose | Input → Output | Depends on | Cx | Replaceability |
|---|---|---|---|---|---|
| `app/orchestrator` | Job lifecycle: plan → execute → validate → render → atomic OutputBundle; progress + cancellation | JobSpec → OutputBundle | kernel, validate, render | ▲▲ | Medium |
| `adapters/cli` | Terminal UX, exit codes, batch globbing | argv → files | app | ▲ | High |
| `adapters/api` | FastAPI async job API under /v1 | HTTP → app calls | app, FastAPI | ▲ | **High by design** — the framework fashion of 2036 replaces this folder only |

---

## 16. Implementation order (unchanged from v1, mapped to new layout)

1. ✅ foundation/config, errors; model; kernel
2. ✅ stages/raster: load, preprocess, quantize
3. ✅ stages/raster: denoise → stages/graph: regions, merge_tiny
4. ✅ foundation/geometry: crack tracing + arc graph (**the heart**)
5. ✅ stages/vector: simplify (wired Sprint 19) · smooth (see note below)
6. ✅ stages/layout: labels · legend (identity permutation, Sprint 19) — palette_order shuffle not yet implemented · validate/*
7. ✅ render: svg → pdf → png (lineart + solved, Sprint 19) · ✅ app/orchestrator (`ConcreteOrchestrator` + `convert()`, Sprint 19) · adapters/cli not yet implemented
8. adapters/api · golden suite · benchmarks · plugin loader
9. Optional plugins (split_large, edge_snap) · perf passes · AI advisors

*(Sprint 19 note on step 5's "smooth": no standalone smoothing stage exists;*
*Bézier fitting (`stages/vector/curves.py`, registered as pipeline slot*
*`bezier`) already performs G1-continuous smoothing as part of fitting*
*(ENGINE_SPEC §18) — Sprint 19's orchestration-only scope did not introduce*
*a new smoothing algorithm. `app/orchestrator.py`'s `Orchestrator.convert()`*
*return type was corrected from `model.artifacts.OutputBundle` (a looser*
*Protocol, field `previews_png`) to `model.reports.OutputBundle` (the*
*concrete, atomicity-checked dataclass, field `previews`) — see*
*`docs/modules/orchestrator.md` for the full compliance report.)*

*(Existing code from phase 1–2 predates the v2 folder layout; it is migrated into `foundation/ model/ kernel/ stages/` at the start of phase 3 — a mechanical move, interfaces unchanged.)*
