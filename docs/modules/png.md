# Module Design — PNG Preview (`render/png`)

**Status:** v1.0 — implemented (Sprint 19). Governing spec: [ARCHITECTURE.md §15](../ARCHITECTURE.md) "render/png" row; [ENGINE_SPEC.md §24](../ENGINE_SPEC.md).

## Purpose

Produce the two PNG previews `OutputBundle` requires (`previews == {"lineart", "solved"}`, DATA_MODEL_SPEC §19). Before Sprint 19, `render/` contained only `svg.py`/`pdf.py`; the closest existing code, `render_preview_png` in `pdf.py`, rasterizes an already-built PDF and produces a single line-art-only image — never a "solved" flood-filled variant (confirmed by the Sprint 18 audit). This module is new code, not a modification of `svg.py`/`pdf.py`.

## Algorithm

Both outputs flatten each face's Bézier rings (chord-density sampling, tolerance 0.1 mm) using a local `_flatten_face_rings`/`_flatten_bezier` pair — deliberately duplicated from the numerically-identical primitives in `validate/common.py` and `stages/layout/labels.py`, because ARCHITECTURE.md §3's layer graph places `render`, `validate`, and `stages` as siblings with no cross-imports permitted (only `model`/`foundation` are shared below them). Importing across would have broken the import-linter's "v2 layer graph" contract, which this session's `lint-imports` run confirmed catches exactly this class of violation.

- **Solved** (`render_solved_png`) — even-odd polygon fill per face (outer ring positive, hole rings painted white) in ascending `face_id` order, hard edges, no anti-aliasing, no labels — ENGINE_SPEC §24 step 3's explicit reason for hard edges: this is the I1 SSIM-probe input, so it must match the quantized label raster's per-pixel color classes exactly.
- **Line art** (`render_lineart_png`) — white canvas, black 1 px stroked face boundaries, printed numbers via `label_plan` — what the customer actually prints.

Both use Pillow (`PIL.Image`/`ImageDraw`) — the dependency ARCHITECTURE.md's dossier names for this module row ("pyvips/Pillow").

## Rejected alternatives

- **Rasterizing the finished SVG via a native rasterizer (resvg/CairoSVG)**: rejected per ENGINE_SPEC §24's own stated reasoning — adds a native dependency and a second geometry interpretation, disqualified as the *default* choice.
- **Reusing `render_preview_png` (PDF rasterization) for both outputs**: rejected — it can only ever produce one flattened rendering of whatever the PDF already draws (line art), structurally incapable of producing an independent, unlabeled, hard-edged "solved" flood-fill, which is the actual I1 probe input ENGINE_SPEC §25.1 requires.

## Quality requirements

Both outputs verified to decode as valid same-size PNGs across every integration test (`test_convert_end_to_end_from_bytes_produces_a_valid_bundle`). Manually verified during implementation: the solved preview of a known 4-color fixture produces exactly 5 unique RGB values (4 palette colors + white margin), not visually all-white or corrupted.

## Configuration

| Key | Type | Default | Range |
|---|---|---|---|
| `png.dpi` | int | 150 | 72–300 |

## Future improvements

- SSIM comparison of the solved preview against the quantized working raster is not yet wired into `validate/fidelity.py` (that validator currently does its own independent rasterization audit, not a comparison against this module's output) — a future increment could plug this module's `render_solved_png` output directly into the QM-17 gate.
