# Module Design — PDF Export (`render/pdf`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §23](../ENGINE_SPEC.md) (page frame §1.3, legend geometry §21).

## Purpose

The self-contained print deliverable: exact trim size, embedded subset font, vector geometry identical to the SVG. Input `CurveSet` + `LabelPlan` + `Legend` + `Palette` + page config; output PDF bytes plus a 300 DPI PNG preview (`PdfDocument` artifact).

## Design

A **native vector re-render** of the same plans the SVG renderer consumes — never a conversion of the SVG. Both outputs sit downstream of the *same* geometry, which is what the renderer-agreement contract test proves. ReportLab is the drawing backend (pure-Python-installable, first-class TTF subsetting); PyMuPDF rasterizes the preview and powers structural validation. Both live behind the `pdf` optional extra.

Drawing order matches §22's layer order exactly:

1. **Regions** — one `beginPath`/`curveTo` chain per **arc** in id order (each shared boundary stroked once), closed when the chain closes; 0.3 pt black, round caps/joins.
2. **Labels** — the bundled font at each anchor, horizontally centered, vertically central via the embedded font's ascent/descent (the `dominant-baseline=central` equivalent).
3. **Leaders** — 0.25 pt lines.
4. **Legend** — rounded chip rects (1.5 pt corners) filled with the palette sRGB, black outline, numbers right of chips; geometry taken verbatim from the `Legend` artifact (§21 owns layout).
5. **Frame** — the content-box rectangle.

**The single y-flip (§1.3 / MATH_SPEC §1).** The engine's page frame is y-down; PDF user space is y-up. One `translate(0, H); scale(1, −1)` at the canvas transform maps every plan coordinate unchanged; text anchors counter-flip locally so glyphs read upright. No other y-up frame exists anywhere.

**Fonts.** The bundled OFL-licensed DejaVu Sans (`assets/fonts/DejaVuSans.ttf`, pinned by SHA-256 — `bundled_font_path` raises `StageError` on a missing or hash-mismatched asset) is embedded as a subset. No system font is ever referenced: the canvas's `initialFontName` overrides ReportLab's Helvetica default, which would otherwise leak into the page resources unembedded.

**Determinism.** ReportLab invariant mode pins the creation date (fixed epoch `D:20000101000000+00'00'`, no wall clock) and document ID, so same inputs give same bytes within a ReportLab version. The PDF *file* bytes are **not** hash-gated (object numbering is not canonical across library versions); the golden surface is the page **content stream** — every drawing operator, coordinate, color and text placement — plus the geometric-agreement contract. Metadata: title, `Creator` = engine version, `Subject` = resolved-config hash.

**300 DPI preview.** `render_preview_png` rasterizes page 1 of the *finished* PDF (PyMuPDF, `dpi/72` matrix), so the preview shows exactly what the print file contains — Letter comes out 2550 × 3300 px.

## Validation

`validate_pdf(bytes, page_mm=…)` re-proves the structural contract and is run by the stage on every render: parseable single-page PDF, exact trim (media) box, every referenced font embedded. Violations raise `StageError`.

## Quality requirements

- Renderer agreement: ≥ 1 000 deterministic points sampled along the arcs agree with SVG space within 0.05 pt — contract test (`tests/contracts/test_renderer_agreement.py`), three seeds.
- Fonts embedded, subset only, no system fonts — verified by parsing the output.
- Trim box exact; y-flip proven on an asymmetric fixture via extracted text positions.
- Budget: ≤ 0.5 s for 12 000 segments (ENGINE_SPEC §26).

## Configuration

| Key | Default | Range |
|---|---|---|
| `pdf.stroke_pt` | 0.3 | 0.05–2 |
| `pdf.preview_dpi` | 300 | 72–1200 |

`pdf.enabled` (ENGINE_SPEC §23) is orchestration-level: the orchestrator includes or omits the stage; the stage itself always renders. `pdf.embed_solved_page` is deferred until the solved-preview module exists.

## Tests

- Unit (`tests/unit/test_pdf.py`): font asset pin, trim box, y-flip label placement, embedding presence + no-system-font, byte determinism + fixed date, metadata, 300 DPI preview dimensions, validator corruption cases, config validation, stage end-to-end.
- Golden (`tests/golden/test_pdf_golden.py`): SHA-256 of the page content stream on the deterministic seed-0 fixture (same fixture as the SVG golden); double-render byte identity.

## Future improvements

PDF/X-1a compliance profile for print bureaus; `embed_solved_page` (solved preview as page 2).
