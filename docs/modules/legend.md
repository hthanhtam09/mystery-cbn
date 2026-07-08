# Module Design — Legend (`stages/layout/legend`)

**Status:** v1.0 — implemented (Sprint 19, identity-permutation only). Governing spec: [ARCHITECTURE.md §15](../ARCHITECTURE.md) "Layout stages"; [ENGINE_SPEC.md §20-21](../ENGINE_SPEC.md).

## Purpose

Produce the `Legend` artifact (`palette → legend`) that `SvgExportStage`/`PdfExportStage` already declared as a required input (`requires = (..., "legend", ...)`) but that no stage ever produced — confirmed absent by the Sprint 18 audit (`find src -iname "*legend*" -o -iname "*palette_order*"` returned nothing). Without this stage, `svg`/`pdf` could never run in a full pipeline.

## Algorithm

1. **Permutation** — identity: `printed_number = palette_index + 1`, in the coverage-descending order the quantize stage already produces (DATA_MODEL_SPEC §4). No numbering-obfuscation ("mystery") shuffle is applied.
2. **Chip layout** — one square chip per palette entry, `chip_mm` side + `gap_mm` gutter, wrapped to fit the page's printable width (`page.width_mm - 2·margin_mm`), row count computed by ceiling division.

## Rejected alternatives

Implementing ENGINE_SPEC §20's Spearman-constrained mystery-shuffle permutation now: rejected as out of scope for Sprint 19 ("implement only the orchestration layer" — the shuffle is a new algorithm, not existing code to wire in). `build_legend`'s permutation is a pure, isolated concern (`_default_permutation`-equivalent logic lives entirely in one expression), so a future `palette_order` advisor can replace it without touching this stage's `Legend`-construction contract or any caller.

## Quality requirements

`Legend.__post_init__` (pre-existing model validation, unchanged) already enforces: permutation is a bijection, one chip per palette entry, every chip lies inside `band_rect`. Verified across every fixture in `tests/integration/test_convert_end_to_end.py` (2–24 color palettes via the three difficulty presets) with no `Legend` construction failure.

## Configuration

| Key | Type | Default | Range |
|---|---|---|---|
| `legend.chip_mm` | float | 6.0 | 1–30 |
| `legend.gap_mm` | float | 2.0 | 0–10 |
| `legend.number_font_pt` | float | 8.0 | > 0 |

## Future improvements

Wire ENGINE_SPEC §20's palette-order advisor (Spearman-rank-constrained shuffle to prevent tone-ramp guessing, QM-19) once implemented.
