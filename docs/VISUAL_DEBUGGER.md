# Visual Pipeline Debugger (Sprint 22)

**Status:** v1.0 — pure developer tool, not test/benchmark infrastructure and not part of the installed package. Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (the 14-stage pipeline this tool visualizes) and [ENGINE_SPEC.md](ENGINE_SPEC.md).

## 1. Purpose

`tools/visual_debugger/` runs an image through the real engine (the same wiring `mysterycbn.app.api.convert()` uses internally) and produces one self-contained HTML report showing every stage's output — viewable in a browser, every artifact downloadable, no server, no build step, no UI framework.

## 2. Usage

```bash
python -m tools.visual_debugger.cli path/to/image.jpg -o report.html
python -m tools.visual_debugger.cli path/to/image.jpg --preset hard --seed 1
```

Open `report.html` in any browser. Nothing else is needed — no local server, no external assets (every image and download is a base64 `data:` URI embedded in the file).

## 3. Stages shown

| Report label | Artifact(s) |
|---|---|
| Original | `source_bytes` (the input file, decoded for preview) |
| Working Resolution | `raster_working` (post-preprocess raster) |
| Image Stats | `image_stats` (colorfulness, edge density, L\*a\*b\* histogram summary) |
| Quantized | `label_map` + `palette` (right after quantize, before denoise/merge) |
| Label Map (post-merge) | `label_map` (final, after denoise + tiny-region merge) |
| Region Graph | `region_graph` (component map, colorized by region id) |
| Arc Graph | `arc_graph` (boundary arcs, traced as an SVG overlay) |
| Curves | `curve_set` (fitted Bézier chains, traced as an SVG overlay) |
| Labels | `label_plan` (every region's printed-number placement) |
| Legend | `legend` (chip layout + palette permutation) |
| SVG | `svg` (the final rendered page, viewable inline) |
| Preview | `png_previews` (line-art + solved rasterizations) |

"Quantized" and "Label Map (post-merge)" are the same `label_map` artifact viewed at two points in the pipeline — the engine doesn't introduce a second distinct raster artifact between quantize and merge, so the debugger shows the one artifact's value at both points it's rebound.

A stage renders as **greyed out** in the report if its artifact wasn't bound (e.g. a stage failed before producing it, or the run didn't reach that far) — the report never silently drops a missing stage.

## 4. Architecture

| Module | Responsibility |
|---|---|
| `runner.py` | Drives the same 14-stage pipeline `ConcreteOrchestrator.convert()` uses, but keeps the populated `InMemoryContext` around afterward instead of collapsing it into an `OutputBundle` |
| `render_artifact.py` | One `view_*` function per artifact type: colorizes label maps/region graphs, traces arc graphs/curve sets as SVG, formats stats/label-plan/legend as text, passes through already-rendered SVG/PDF-preview bytes |
| `stages.py` | Maps artifact names to the report's stage labels and dispatches each bound artifact to its renderer |
| `html_report.py` | Assembles one HTML string: inline CSS (light/dark aware), a table of contents, one `<section>` per stage, base64 `data:` URIs for every image and download |
| `cli.py` | `python -m tools.visual_debugger.cli <image> -o <report.html>` |

No engine file is modified. `runner.py` duplicates `ConcreteOrchestrator`'s plan-building glue (config layering, registry, `DefaultPlanResolver`, `SequentialExecutor`) rather than adding a debug hook to the orchestrator — the public `convert()` API deliberately returns only the validated `OutputBundle` (ARCHITECTURE.md §11 atomicity), and this tool needs the intermediate context, which is a legitimate reason to build a separate, dev-only driver rather than changing the production entry point.

## 5. Design choices

- **No UI framework.** The report is one `.py` module building a plain HTML string with an f-string template and a `<style>` block — no React/Vue/build step/CDN dependency. This is deliberate: a debugging tool that requires `npm install` to view its own output defeats the purpose.
- **Self-contained.** Every image and downloadable artifact is base64-encoded inline as a `data:` URI. The report has zero external references (the only `http://` string in the output is the SVG namespace URI, `xmlns="http://www.w3.org/2000/svg"`, not a network fetch) and can be emailed, committed, or opened from disk with no server.
- **Downloads, not just previews.** Every stage's artifact is available as a real file via a `download` link — raw PNGs for raster stages, `.svg` for arc-graph/curve-set traces, `.txt` for label-plan/legend/stats, and the final `page.svg` — so a developer can pull any intermediate artifact into another tool (an SVG viewer, a diff, an image editor) instead of only eyeballing the embedded preview.
- **Reuses the real pipeline.** No stage is re-implemented or mocked; the debugger runs the identical `Stage` instances, config resolution, and executor the production `convert()` call uses, so what's shown is what actually happened, not an approximation.

## 6. Scope

This is a developer tool: it does not implement or modify any engine algorithm, stage, or validator, and it is not wired into CI as a gate (see the `visual-debugger` CI job, which only runs its own unit tests — it does not run against every commit's output as a quality check, unlike `benchmarks/golden`).
