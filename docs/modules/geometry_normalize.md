# Module Design — Geometry Normalize (`stages/vector/geometry_normalize`)

**Status:** v1.0 — wired into `PIPELINE_STAGES` (Sprint 36A.5), between `simplify` and `bezier`.
Pass 1 (Duplicate Point Cleanup) and Pass 2 (Spike Removal) are implemented (Sprint 36B.1,
36B.2); Pass 3 (Minimum Gap Enforcement) remains an identity placeholder (Sprint 36A.4),
pending its own implementation sprint on top of the already-frozen design (GAP_REPAIR_DESIGN.md).
Governing context:
[ADR-002](../adr/002-sprint36-print-aware-geometry-simplification.md) (as revised by the
accepted architecture review superseding face-aware pinch repair with this stage),
[GAP_REPAIR_DESIGN.md](GAP_REPAIR_DESIGN.md) (Pass 3 detail), [ENGINE_SPEC.md §16–17](../ENGINE_SPEC.md)
(Simplify, Smooth — the two existing idioms this stage extends), [MATH_SPEC.md §6–8](../MATH_SPEC.md)
(Sidedness, Polygon operations, Simplification mathematics).

## 1. Purpose

Enforce a small set of geometric preconditions on `ArcGraph` — no near-duplicate points, no
degenerate spikes, no arc pair closer than a configured minimum separation — before `bezier`
fits curves. This replaces the previous face-aware "pinch repair" logic that lived inside
`bezier` (`stages/vector/curves.py`), which discovered self-intersections by rasterizing
`ArcGraph.faces` walks *after* curve fitting and retried fitting at tighter tolerances. That
design coupled curve fitting to topology, broke the "arc-local, face-independent" contract
`fidelity` (I1) and `printability` (I4) validators depend on for their independent re-proof to
be meaningful, and used a hidden retry loop that could silently degrade output quality. Geometry
Normalize moves this correction upstream, onto `ArcGraph` geometry alone, so `bezier` can return
to being a pure per-arc fitter.

## 2. Pipeline position

```
... → arcgraph → simplify → geometry_normalize → bezier → labels → ...
```

Runs **after** `simplify`, because gap/spike/duplicate defects must be measured on the exact
polylines `bezier` will receive — measuring before `simplify` risks flagging distances that
`simplify`'s own vertex reduction would later change. Runs **before** `bezier`, so `bezier`
never has to reopen the question, discover cross-arc conflicts post-fit, or read `Face`
structure at all.

## 3. Inputs

`ArcGraph` (from `simplify`): `arcs: tuple[Arc, ...]`, `faces: tuple[Face, ...]`, `work_scale:
float`, `provenance: Provenance`. Also reads `simplify.tolerance_mm` from the resolved config
(the shared ceiling every internal pass's threshold is bounded by — see §10) at stage
construction time, not per-run.

## 4. Outputs

A new `ArcGraph` with the same `arcs` count, same dense `arc_id` ordering, identical `faces`
(unchanged object, passed through — this stage never constructs a new `Face`), identical
`work_scale`, and a re-stamped `Provenance` (`stage_name="geometry_normalize"`). Per-arc,
`points` may differ; `arc_id`, `left_region`, `right_region`, `closed` are always identical to
the input arc of the same `arc_id`. Arcs untouched by every pass are passed through as the
identical object (not merely equal — the same instance), enabling cheap downstream diffing.

## 5. Invariants

Preserved, never re-derived from scratch (each pass is individually responsible for not
breaking these; no repair-after-the-fact check exists inside this stage beyond each pass's own
local self-check, per GAP_REPAIR_DESIGN.md §5–6):

- **Dense arc_id / face_id ordering** — unchanged, since this stage never adds, removes, or
  reorders arcs or faces.
- **`_check_arc_references` (≤ 2 walk-sides per arc)** — unaffected, since `Face` objects are
  never constructed or mutated by this stage.
- **Junction endpoints immovable** — no pass ever displaces a vertex that is a shared endpoint
  between two or more arcs (GAP_REPAIR_DESIGN.md §5.2). This is what keeps the shared-edge
  property at every junction intact.
- **Sidedness (MATH_SPEC §6.2)** — every geometry-modifying edit is gated by the same guard
  formula `simplify`'s own VW pass uses (MATH_SPEC §8.2): a candidate change is admissible only
  if the swept region it traces contains no foreign vertex or segment and does not invert the
  local orientation of adjacent segments.
- **Watertightness (I3)** — a direct consequence of the sidedness guard plus junction
  immovability; not independently re-checked inside this stage (that remains `topology`'s and
  the validators' job), but never violated by construction under the stated guard.

## 6. Non-goals

- **No face awareness.** The stage's core per-pass transform operates on `tuple[Arc, ...]` only
  and cannot accept `Face` — the type signature itself makes face-dependence structurally
  impossible, not merely discouraged by convention.
- **No curve fitting.** Zero knowledge of `Curve`, `CurveSet`, `BezierSegment`. This stage never
  reads or produces anything in `bezier`'s output domain.
- **No acute-angle / corner handling.** Explicitly `bezier`'s responsibility
  (`corner_angle_deg`) — duplicating it here would recreate the same implicit coupling this
  redesign exists to remove, one stage earlier.
- **No area or region-floor decisions.** `merge_tiny`'s job, already done upstream of `regions`
  → `topology` → `arcgraph`; this stage does not reopen area-floor questions.
- **No raster/`component_map` access.** By this pipeline position the raster→vector crossing
  already happened at `topology`; reaching back for pixel data would be a layering violation of
  the kind `lint-imports` already flags elsewhere in this codebase.
- **No retry loops, no tolerance-halving fallback, no straight-line degradation.** The
  mechanism that made the old pinch-repair unsafe. Every pass either repairs fully within its
  bound or skips; nothing is retried at reduced strength.
- **No third-party plugin extensibility.** Passes are a fixed, hardcoded internal sequence, not
  a discoverable/registrable extension point (see architecture review §3: this stage does not
  have a genuine "choose one of several" axis the way `bezier`'s `impl` fitters do).

## 7. Execution order

Fixed, not configurable, three internal passes in sequence:

```
1. duplicate_point_cleanup
2. spike_removal
3. minimum_gap_enforcement   (Gap Repair — see GAP_REPAIR_DESIGN.md)
```

**Why this order.** Duplicate cleanup must run first because near-duplicate points destabilize
the distance and angle computations both later passes depend on — it is a numerical-stability
precondition, not an independent correction. Spike removal must run before gap enforcement
because a spike is a local, single-arc shape defect that can manufacture a spurious apparent
gap against a neighboring arc if measured before the spike itself is corrected; fixing the
arc's own shape first avoids Pass 3 ever reacting to an artifact of an unfixed spike in Pass 2's
scope. Each pass receives the previous pass's full output as its input (not the original
`ArcGraph`) — corrections compose sequentially, never in parallel.

## 8. Each normalization pass

### 8.1 Duplicate point cleanup

Collapses consecutive near-duplicate points on a single arc's polyline (points closer than a
configured epsilon, `duplicate_eps_mm`) into one point. Scope: single-arc, no cross-arc
reasoning. Purpose: numerical stability for Pass 2 and Pass 3's distance/angle computations,
not a correction in its own right. Full design (precise epsilon definition, tie-breaking,
property tests) is deferred to its own design document, per the same "one document per pass"
discipline this document and GAP_REPAIR_DESIGN.md both follow — not specified further here to
avoid drifting from what has actually been frozen.

### 8.2 Spike removal

Removes a degenerate single-vertex protrusion on one arc that juts out and immediately reverses
direction (a self-proximity defect, as opposed to Pass 3's inter-arc proximity defect). Scope:
single-arc. Explicitly the owner of self-approaching "fin" geometry that GAP_REPAIR_DESIGN.md
§1.1 excludes from gap measurement ("a self-approaching arc... is a spike, explicitly owned by
Pass 2, not this pass"). Full design deferred to its own design document, same discipline as
§8.1.

### 8.3 Minimum gap enforcement (Gap Repair)

Full design frozen in [GAP_REPAIR_DESIGN.md](GAP_REPAIR_DESIGN.md). Summary only, for this
document's completeness:

- **What:** detects arc pairs (excluding pairs sharing a junction) whose minimum
  segment-to-segment clearance falls below `min_gap_mm` (converted to pt), via broad-phase
  spatial-grid pruning (same cell-sizing convention as the VW guard, MATH_SPEC §8.2) then exact
  narrow-phase segment distance.
- **How repaired:** symmetric constrained displacement — both arcs' witness vertices moved
  apart by half the required separation increase each, bounded by `min(Δ/2,
  simplify.tolerance_mm)`, gated by the same sidedness guard `simplify`'s VW pass already uses.
- **Failure policy:** skip (never retry, never partially apply) if the guard fails or the
  required displacement exceeds the bound.
- See GAP_REPAIR_DESIGN.md §1–10 for the complete mathematical definition, detection algorithm,
  repair operation, bound derivation, topology proof, determinism proof, complexity, failure
  conditions, property tests, and benchmark requirements. This module document does not restate
  or paraphrase those sections further, to avoid two documents drifting out of sync.

## 9. Determinism guarantees

Every pass is a composition of pure, order-fixed functions: no PRNG, no wall-clock, no
iteration over unordered collections in any output-affecting path. Candidate/pair generation in
Pass 3 is canonicalized to `(min(arc_id), max(arc_id))` before processing; ties in distance or
witness selection resolve via fixed lexicographic tuples (GAP_REPAIR_DESIGN.md §1.3, §2.3).
Passes run in a fixed sequence (§7), each fully consuming the prior pass's output before the
next begins — no interleaving, no feedback loop back to an earlier pass. Consequence: identical
`(ArcGraph, config)` always produces byte-identical output `ArcGraph`, satisfying the engine's
I2 determinism contract without this stage needing its own seeded-PRNG stage-hash (it has
nothing stochastic to seed).

## 10. Error budget policy

Each pass owns its own named config threshold (`duplicate_eps_mm`, `spike_length_mm`,
`min_gap_mm`), self-validated at stage construction — not a single shared numeric budget across
passes, since the three thresholds measure incommensurable geometric quantities (a duplicate
epsilon and a gap width have no principled reason to share one literal value). All three
thresholds are bounded by one shared **ceiling**, not summed: none may exceed
`simplify.tolerance_mm`. This is enforced as a `ConfigError` at stage construction (reading
`simplify.tolerance_mm` from the resolved config), before any `ArcGraph` is processed — a
build-time guarantee, not a runtime hope. Rationale: passes edit largely disjoint vertices (a
duplicate-collapse site and a gap-repair site rarely coincide), so bounding each independently
against the same ceiling is the correct conservative bound; summing the three budgets would be
over-conservative and would under-permit legitimate per-pass corrections.

## 11. Config section

New, dedicated section, not folded into `simplify`, `merge`, or `quality`:

```
geometry_normalize:
  enabled: bool              # default true; false = identity pass-through, re-stamped provenance only
  duplicate_eps_mm: float    # Pass 1 threshold
  spike_length_mm: float     # Pass 2 threshold
  min_gap_mm: float          # Pass 3 threshold (Gap Repair) — must be ≤ simplify.tolerance_mm
```

Each threshold self-validated in the stage's own `__init__`, following the exact pattern
`MergeTinyStage`/`CurveFitStage` already use (range check + `ConfigError` on violation, before
any geometry is touched). `min_gap_mm`'s validation additionally requires the resolved
`simplify.tolerance_mm` value, per §10.

## 12. Failure policy

Following the engine's established idiom (`simplify`'s VW guard: "Failure modes. None; skipping
all removals is always legal," ENGINE_SPEC §16):

- **Repair** — the default path for any pass whose correction fits within its bound and passes
  its guard.
- **Skip** — legal, expected, not an error. A candidate correction whose required change exceeds
  its bound, or whose sidedness guard fails, is left unrepaired. Skipping never leaves the
  output worse than running with the stage disabled.
- **`ConfigError`** — at stage construction only, if any threshold violates the §10 ceiling
  rule, or fails its own declared range.
- **`StageError`** — only on a genuine precondition violation of the input `ArcGraph` itself
  (should be structurally unreachable given `ArcGraph.__post_init__`'s own guarantees; this
  stage does not re-derive or bypass that check).
- **Never:** a retry loop, a tolerance-halving fallback, or a degrade-to-straight-line fallback.
  Excluded by design, matching GAP_REPAIR_DESIGN.md §8's explicit prohibition, inherited by this
  stage as a whole.

## 13. Provenance

Standard `Provenance` stamping, identical in shape to every other stage:

```
Provenance(
    stage_name="geometry_normalize",
    stage_version="1.0.0",
    config_hash=<hash of geometry_normalize section + read simplify.tolerance_mm ceiling>,
    source_hash=arc_graph.provenance.source_hash,
)
```

`source_hash` carried through unchanged from the input `ArcGraph`, matching the pass-through
convention every stage in this pipeline already uses.

## 14. Metrics

Per-run structured counts surfaced alongside the existing `stage_timings_s` mechanism (no new
artifact type invented): `{"duplicates_removed": n, "spikes_removed": n, "gaps_repaired": n,
"gaps_skipped": n}`. Purpose: auditability of how much correction a given input required, and
the benchmark corpus sanity check that correction counts are near-zero on fixtures with no thin
or noisy geometry (a non-zero count there indicates a threshold or symmetry defect worth
investigating before shipping, per GAP_REPAIR_DESIGN.md §10).

## 15. Complexity

- **Pass 1 (duplicate cleanup) / Pass 2 (spike removal):** O(P) per arc, P = arc point count —
  single linear scan each, no cross-arc structure needed.
- **Pass 3 (gap enforcement):** broad phase O(A) expected (A = arc count) under the engine's
  existing bounded boundary-density assumption (MATH_SPEC §8.2); narrow phase O(P_a · P_b) per
  surviving candidate pair; worst case approaches O(A²) only if that density assumption is
  violated (e.g. pathological dense parallel hatching) — see GAP_REPAIR_DESIGN.md §7 for the
  full breakdown, including the explicit worst-case benchmark row.
- **Overall stage:** O(A + P_total) memory (grid plus per-arc point storage), matching the same
  order as `simplify`'s own VW pass memory bound.

## 16. Future extensions

- **Swappable pass implementations**, if a future increment genuinely needs a "choose one of
  several" axis for any individual pass (e.g. two competing spike-detection heuristics) — the
  precedent to reuse at that time is `bezier`'s `_FITTERS`-style plain dict-of-named-callables,
  not a new plugin/registry layer (architecture review §3). Not built now, since no pass
  currently has more than one candidate algorithm.
- **Per-pass design documents** for Pass 1 (duplicate cleanup) and Pass 2 (spike removal),
  matching the depth and structure of GAP_REPAIR_DESIGN.md, before either pass is implemented —
  named here as an explicit, tracked gap rather than left implicit.

## 17. Rejected alternatives

- **Face-aware post-fit pinch repair inside `bezier`** (the prior design) — rejected: coupled
  curve fitting to topology, broke the arc-local/face-independent contract `fidelity`/
  `printability` depend on, and used a hidden tolerance-halving retry loop that could silently
  degrade output quality. This is the regression this entire stage exists to fix; see the
  accepted architecture review for the full analysis.
- **A single shared numeric error budget across all three passes** — rejected in favor of
  per-pass thresholds bounded by one shared ceiling (§10): the three passes measure
  incommensurable geometric quantities, and a literal shared value would make at least one
  pass's threshold semantically meaningless.
- **Dynamic plugin registration for passes** (entry-point discovery via
  `foundation/plugins.py`, or even a lightweight named-dict registry like `bezier`'s
  `_FITTERS`) — rejected for v1.0: there is no genuine "choose one of several" axis among the
  three passes (they are a fixed sequence of preconditions, not interchangeable alternatives),
  and topology-adjacent correctness code is treated as core-engine-owned in this codebase's
  existing philosophy, not third-party-extensible (architecture review §3).
- **Configurable pass execution order** — rejected: the fixed order (§7) is load-bearing for
  correctness (each pass is a numerical-stability or scope precondition for the next), not a
  stylistic preference; making it configurable would allow a configuration that silently
  reintroduces the exact problems the ordering exists to prevent.
- **Vertex redistribution / adaptive densification as a standalone repair mechanism** —
  rejected as a *policy*, retained only as *plumbing* inside Pass 3 (inserting a vertex at an
  exact witness location so displacement has something to act on) — see GAP_REPAIR_DESIGN.md
  §3.1, §3.4 for the full comparison table and rationale.
