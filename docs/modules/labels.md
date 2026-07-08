# Module Design — Label Placement (`stages/layout/labels`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §19](../ENGINE_SPEC.md); math [MATH_SPEC.md §13–§14](../MATH_SPEC.md); data model [DATA_MODEL_SPEC.md §15](../DATA_MODEL_SPEC.md).

## Purpose

Place every region's printed number where a human can read it and unambiguously associate it with the region (invariant I4's readability half). Input `CurveSet` + `RegionGraph`; output `LabelPlan` (labels sorted by region id) plus FATAL `Finding`s for faces that cannot be labeled — the §25 validator decides abort; this stage never aborts on its own.

## Algorithm

1. **Pole of inaccessibility / largest empty circle** (MATH_SPEC §13.3) — quadtree branch-and-bound maximization of the face's signed distance function over Bézier boundaries flattened at 0.1 mm; holes respected by even-odd containment over all rings; precision `polylabel_precision_pt` (default 0.5 pt > flattening tolerance — the config cross-rule). Ties broken by (U, cell-center lex order) — deterministic. Yields `(c*, r*)`: the largest empty circle's center (anchor) and radius (clearance; `2r*` is the printability diameter, QM-10 evidence stored on each label). Child-cell distance evaluations are batched 4-at-a-time (this is what brought the benchmark from 924 → 566 ms).
2. **Font scaling** (§14.1) — a string of `n` digits at size S occupies `(n·ω_f·S) × (κ_f·S)` with pinned DejaVu Sans metrics (ω_f = 0.636, κ_f = 0.729); inscribing that box in the clearance circle gives the exact optimum `S_fit = 2r*/√((n·ω_f)² + κ_f²)` (the spec's 1.35·r* closed form for two digits — unit-tested against a brute-force bbox check), clipped to `[font_min, font_max]` (6/14 pt, quality config). Real per-number digit counts — no worst-casing.
3. **Leader lines** (§14.2) — when `S_fit < font_min`: 16 candidates on the ring `r* + ρ` (ρ = `leader_ring_mm`, default 4 mm) at fixed π/8 steps; feasible iff the bbox at `font_min` clears all page geometry by more than its own half-diagonal and the segment candidate→pole crosses < 3 arcs (normative constant). Choose minimal (crossings, angle index). No feasible candidate → `Finding(FATAL, I4)`.
4. **Collision avoidance** (§14.3) — greedy by descending clearance; a conflicting in-region label slides uphill along ∇d (central differences on the face distance) by up to r*/2 in fixed fractions, accepting only positions where its bbox still fits inside the face; still conflicting → demoted to leader; no leader either → FATAL finding. Deterministic (fixed step fractions, no RNG).

## Rejected alternatives

Centroid anchors (outside concave/annular faces); exact medial axis (degree-≥6 bisector curves, unstable — only the maximum of d is needed, per MATH_SPEC §13.2); ILP global placement (heavy solver, nondeterministic paths).

## Quality requirements

- 100 % of faces labeled or carried as FATAL findings (`len(labels) + len(findings) = F` — benchmarked).
- 0 label-bbox overlaps; in-region bboxes never cross region boundaries (fit condition at the pole; re-checked after displacement) — unit-tested.
- Determinism — unit-tested (double run, identical plans).
- Budget: ≤ 1.0 s for F = 800 (ENGINE_SPEC §26) — measured ≈ 0.57 s.

Unit coverage per spec: C-shape pole inside the C, annulus pole in the ring band with `r* ≈ (R−r)/2`, font formula vs brute force, leader fallback on a 1×30 mm sliver, overlap/displacement determinism.

## Configuration

| Key | Default | Range |
|---|---|---|
| `labels.polylabel_precision_pt` | 0.5 | 0.05–2 |
| `labels.leader_ring_mm` | 4.0 | 1–20 |
| font bounds | `quality.font_min_pt`/`font_max_pt` (6.0/14.0) | — |

## Artifacts

Requires `curve_set` + `region_graph` (generation pairing checked); provides `label_plan` (`LabelPlan`, added to `model/layout.py`) and `label_findings` (a `LabelFindings` wrapper carrying the FATAL findings with provenance). Printed numbers are `label + 1` (1-based) until the §20 palette permutation rewrites them.

## Future

Seeded simulated-annealing global placement as a registered alternative for dense pages.
