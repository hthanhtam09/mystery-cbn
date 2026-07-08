# Module Design — SVG Export (`render/svg`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §22](../ENGINE_SPEC.md) (legend geometry from §21's `Legend` artifact).

## Purpose

The canonical output renderer: byte-deterministic, structurally clean SVG line art that all other outputs must geometrically agree with. Input `CurveSet` + `LabelPlan` + `Legend` + `Palette` + page config; output UTF-8 SVG bytes (`SvgDocument` artifact).

## Design

A **direct string serializer** — no svgwrite/lxml. Writer libraries do not guarantee attribute ordering across versions; invariant I2's byte-identical promise cannot be delegated to a third party's internals.

Document structure (fixed layer order, validated):

1. `<svg>` — `viewBox` in pt, explicit `width`/`height` in **mm** (print-safe physical sizing); the only comment is the engine version (constant per build, part of the reproducibility record).
2. `<g id="regions">` — one `<path>` per **arc**, not per face: each shared boundary drawn once (half the ink, no double-stroke darkening). `M`/`C` commands from the Bézier chains, `Z` on closed arcs; `stroke:#000` at `svg.stroke_pt`, round caps/joins. `data-left`/`data-right` carry the printed numbers of the bordering regions (0 = page exterior) for downstream tooling.
3. `<g id="labels">` — `<text>` per label, middle/central anchored, bundled font referenced by family name (the PDF is the self-contained deliverable; documented product decision).
4. `<g id="leaders">` — 0.25 pt `<line>` per leader label.
5. `<g id="legend">` — rounded chip rects (1.5 pt corners, 0.3 pt outline) filled with the palette sRGB exactly, numbers right of chips — geometry taken verbatim from the `Legend` artifact (§21 owns layout; this stage only serializes).
6. `<g id="frame">` — the content-box frame (margins made visible as page furniture).

**Determinism rules (I2's test surface):** all coordinates formatted to exactly `svg.decimals` places with negative zero normalized to `0.000`; elements emitted in id order; attribute order fixed by the serializer; LF newlines; no timestamps. A golden SHA-256 of the full output bytes gates any change.

**Print-safe contract:** explicit mm sizing, pure black strokes only (tested: the stroke attribute set is exactly `{#000}`), no scripts, no external references, no embedded rasters.

## Validation

`validate_svg(bytes, curve_set?)` re-proves the structural contract and is run by the stage on every render: well-formed XML, SVG 1.1 root with `viewBox` + mm dimensions, exact layer order, and (with a `CurveSet`) each arc present exactly once in id order. Violations raise `StageError`. Corruption cases (truncation, renamed layer, missing arc, stripped units) are unit-tested.

## Quality requirements

- Byte-identical across runs — golden hash on a full-pipeline fixture + double-render unit test.
- Valid SVG 1.1, arc-once — validator, run per render and in tests.
- Budget: ≤ 0.3 s for 12 000 segments, output ≤ 2 MB (ENGINE_SPEC §26) — measured ≈ 64 ms / ~1.1 MB.

## Configuration

| Key | Default | Range |
|---|---|---|
| `svg.stroke_pt` | 0.3 | 0.05–2 |
| `svg.decimals` | 3 | 2–5 (I2 hash fixtures pinned to 3) |

## Artifacts

Requires `curve_set`, `label_plan`, `legend`, `palette`; provides `svg` (`SvgDocument`, stage `svg` v1.0.0). Page geometry via the `page` config (same tuple as the Arc Graph stage).

## Future

Optional `data-` metadata toggle for a lighter file.
