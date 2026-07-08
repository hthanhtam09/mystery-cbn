# Module Design — Curve Fitting (`stages/vector/curves`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §18](../ENGINE_SPEC.md); data model [DATA_MODEL_SPEC.md §13–§14](../DATA_MODEL_SPEC.md).

## Purpose

Convert each arc polyline into a compact chain of cubic Bézier segments — the final geometry renderers consume. Output `CurveSet` (per-arc chains + faces carried over unchanged from the ArcGraph).

## Fitter comparison and default choice

Four fitters were implemented behind one interface (`bezier.impl`) and measured on smooth-with-noise fixtures (1 920 vertices, jitter σ = 0.15 pt, tolerance 0.5 pt; `benchmarks/quality/test_curves_quality.py`):

| impl | max residual (pt) | segments | bending energy Σθ² | verdict |
|---|---|---|---|---|
| **`schneider`** (default) | **0.50 ≤ tol** | **132 (6.9 %)** | 11.3 | error-bounded, economical, smooth |
| `bezier` (uniform least-squares) | 0.58 > tol | 242 | 8.8 | smooth but **no error guarantee** |
| `chaikin` (quadratic B-spline limit) | 0.26* | 1 905 | 19.2 | doesn't interpolate vertices; 1 segment/vertex |
| `catmull` (Catmull–Rom) | 0.00 | 1 914 | 44.0 | interpolates the noise — worst smoothness, most segments |

\* Chaikin's residual is data-dependent (up to half the local feature size) — it has no bound, which disqualifies it despite good smoothness on gentle input.

**Default: Schneider** (Graphics Gems "An Algorithm for Automatically Fitting Digitized Curves") — the only fitter simultaneously error-bounded by construction, within the §18 segment budget (≤ 0.15 × vertices), and smoother than the interpolating alternative. Matches the spec's own selection.

## Algorithm (Schneider, per corner-free run)

1. End tangents = normalized average of the first/last ≤ 3 chords.
2. Least-squares cubic over chord-length parameters with fixed tangent directions (Wu/Barsky ⅓-chord fallback on degenerate α).
3. If max residual > tolerance: ≤ 4 Newton–Raphson reparameterization refits — skipped when the error exceeds 16× tolerance (Newton is hopeless there) and abandoned when improvement stalls below 5 % (perf heuristics; the unit tests pin the quality behavior).
4. Still exceeding → split at the max-error point with a centripetal tangent estimate and recurse (depth ≤ 32; the floor degrades to per-edge exact line segments — never incorrect).
5. Adjacent segments share the split point and mirrored tangent → G1 inside runs; 2-point runs → exact ⅓/⅔-chord line segments.

## Corners and topology

- Interior vertices with turn angle > `corner_angle_deg` (default 65°, core config) split arcs into runs; corner positions survive **bitwise** and joints there are intentional C0, recorded in `corner_indices`. A closed arc's anchor is a corner by definition (chain cut there).
- Chain endpoints interpolate the arc's junction coordinates **bitwise** — watertightness at junctions is positional identity, not tolerance (unit-tested for all four fitters).
- Faces are carried over from the ArcGraph by reference, unchanged — the planar structure is untouched.

Note: the fitter expects §17-smoothed input; raw crack staircases flag every vertex as a corner and degrade to per-edge lines (correct, uneconomical). Simplification/Smoothing (§16–§17) are not yet implemented.

## Configuration

| Key | Default | Range |
|---|---|---|
| `bezier.fit_error_mm` | 0.15 (core `CurveConfig`) | 0.02–2.0 |
| `bezier.corner_angle_deg` | 65.0 | 15–120 |
| `bezier.impl` | `schneider` | `schneider`, `bezier`, `chaikin`, `catmull` |

## Quality requirements

- Max residual ≤ fit error; junction exactness to the double-precision value; determinism — unit-tested.
- Segment count ≤ 0.15 × input vertices; smoothness (bending energy) below the interpolating comparator — gated in `benchmarks/quality`.
- Budget: ≤ 1.0 s for 80 000 vertices → ≤ 12 000 segments (ENGINE_SPEC §26) — measured ≈ 0.85 s.

## Artifacts

Requires `arc_graph`; provides `curve_set` (`CurveSet`, terminal geometry; stage `bezier` v1.0.0).

## Future

Numba hotspot port (sanctioned); G2 fitting as a registered alternative.
