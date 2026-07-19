# Gap Repair — Algorithm Design (Geometry Normalize, Pass 3)

**Status:** Design frozen, pending implementation approval. Governing context: [ADR-002-successor
architecture review](../adr/002-sprint36-print-aware-geometry-simplification.md) (superseded in
direction by the `geometry_normalize` redesign), [ENGINE_SPEC.md §6](../ENGINE_SPEC.md) (Topology),
[MATH_SPEC.md §6–8](../MATH_SPEC.md) (Sidedness, Polygon operations, Simplification mathematics).

This document specifies **only** the third internal pass of the `geometry_normalize` stage
(`simplify → geometry_normalize → bezier`): **minimum-gap enforcement between arcs**. Passes 1
(duplicate-point cleanup) and 2 (spike removal) are out of scope here — each gets its own design
document before implementation, per the frozen `geometry_normalize` architecture.

This is a WHAT document, not a HOW document. No function signatures, no code structure, no data
types beyond what is already frozen in `model/vector.py` (`Arc`, `ArcGraph`). An engineer should be
able to implement this from the document alone, in whatever language or code shape fits the
codebase, without further design questions.

---

## 1. What exactly is a "narrow gap"?

### 1.1 Participating arcs

A gap is a property of an **unordered pair of arcs** `{a, b}`, `a ≠ b`, drawn from
`ArcGraph.arcs`, subject to one exclusion:

> **Exclusion rule.** If `a` and `b` share an endpoint (i.e. they meet at a common junction — either
> arc's first or last point equals the other's first or last point, compared as exact doubled-crack
> lineage, not by floating-point proximity), the pair is **not eligible** for gap measurement. Arcs
> that legitimately meet at a junction are expected to be close there; that is not a gap, it is
> topology, and any "repair" of it would be a topology violation, not a normalization.

This mirrors the existing VW guard's own scope discipline (MATH_SPEC §8.2: "no vertex or segment
crossing of any *other* geometry... this arc's non-adjacent segments") — gap repair, like the VW
guard, only ever reasons about geometry that is not already structurally related by the topology
graph.

A single arc is never checked against itself for self-gaps in this pass — a self-approaching arc
(a "fin" folding back near itself) is a **spike**, explicitly owned by Pass 2, not this pass. This
boundary is a hard scope line: Pass 3 only ever measures *inter*-arc distance on the pair set
defined above, never intra-arc distance.

### 1.2 Measurable distance

For a pair `{a, b}` not excluded above, define the **pairwise clearance** as the minimum Euclidean
distance between the two arcs' polylines, each treated as a chain of line segments (post-simplify,
post-Pass-1, post-Pass-2 points — i.e. the geometry as it stands when Pass 3 runs):

```
d(a, b) = min over all segments s_a in a, s_b in b of dist(s_a, s_b)
```

where `dist(s_a, s_b)` is the standard closed-form minimum distance between two line segments in
the plane (segment-to-segment, not vertex-to-vertex — using only vertex distance would miss a
segment passing close to another segment's interior, the single most common narrow-gap shape:
two nearly-parallel walls).

Units: pt (post-Φ scale, the space Pass 3 operates in — see §4 of the `geometry_normalize`
architecture review for why this stage sits after `simplify`, in pt space, not px).

### 1.3 Local minimum

`d(a, b)` as defined is already the **global** minimum clearance between the two arcs — there is no
separate "local minimum" search needed, because the segment-to-segment distance formula is exact
and the minimum over a finite segment set is well-defined and unique up to ties. What matters
operationally is *where* the minimum occurs — the **witness pair** `(s_a*, s_b*)` — because repair
(§3) acts locally at that witness location, not on the whole arc. When multiple segment pairs tie
exactly at the minimum distance (a measure-zero event in practice, but must be handled
deterministically — see §6), the witness is the pair with the lexicographically smallest
`(segment_index_in_a, segment_index_in_b)`, ties broken by `(min(a.arc_id, b.arc_id),
max(a.arc_id, b.arc_id))` — this is a total order, so a unique witness always exists.

### 1.4 Threshold

A pair `{a, b}` is a **narrow gap** iff:

```
d(a, b) < min_gap_pt
```

where `min_gap_pt = min_gap_mm · 72/25.4` (the standard mm→pt conversion every other mm-based
threshold in this codebase uses — `foundation/units`), and `min_gap_mm` is bounded at config-resolve
time by the shared ceiling rule already frozen in the `geometry_normalize` architecture: `min_gap_mm
≤ simplify.tolerance_mm`. No independent range is invented here.

This is a **strict** inequality by design: `d(a,b) == min_gap_pt` exactly is not a violation (this
matches the engine's established convention of treating exact-boundary cases as satisfying rather
than violating a floor — c.f. `merge_tiny`'s `area[r] >= a_min` treated as legal in `merge.py`).

---

## 2. How is a gap detected?

### 2.1 Broad phase

Purpose: avoid the naive O(arcs²) full segment-to-segment scan, which is unnecessary since the
overwhelming majority of arc pairs are far apart.

1. Compute each arc's axis-aligned bounding box (min/max over its points), **expanded by
   `min_gap_pt`** in each direction (a box inflated by the threshold — any pair whose *un-inflated*
   boxes are farther apart than `min_gap_pt` cannot possibly violate the threshold, by the triangle
   inequality on axis-aligned separation).
2. Insert all expanded boxes into a uniform spatial grid with cell size `2 · min_gap_pt` — this is
   the exact same cell-sizing convention MATH_SPEC §8.2 already uses for the VW sidedness guard
   ("uniform spatial hash with cell size `2τ_pt`"). Reusing this convention is deliberate: it is
   already proven in this codebase to bound expected candidate counts to O(1) per query under the
   engine's boundary-density assumptions (arcs disjoint except at junctions).
3. Candidate pairs are all `{a, b}` whose expanded boxes overlap in at least one shared grid cell,
   with each pair considered exactly once (dedup by the `(min(a.arc_id,b.arc_id),
   max(a.arc_id,b.arc_id))` key).

This produces a candidate set that is a **superset** of all true narrow gaps — the broad phase can
only produce false positives (checked and dismissed in narrow phase), never false negatives, because
box inflation by the exact threshold is a sound (conservative) pruning bound.

### 2.2 Narrow phase

For each candidate pair `{a, b}` surviving the broad phase:

1. Compute the exact segment-to-segment minimum distance (§1.2) between every segment of `a` and
   every segment of `b` — for typical arc lengths post-simplify (already vertex-reduced by VW) this
   is a small dense double loop, not itself gridded; a second-level grid is only justified if
   profiling on the golden dataset shows arcs long enough to warrant it (see §7, this is a stated
   complexity concern, not assumed a priori).
2. Track the running minimum and its witness segment-index pair (§1.3).
3. If the final minimum is `< min_gap_pt`, the pair is a confirmed narrow gap; record `(a.arc_id,
   b.arc_id, witness segment indices, d(a,b))`.

### 2.3 Why the result is deterministic

- Broad-phase grid cell assignment is a pure function of arc geometry (bounding boxes) and the
  fixed cell-size formula — no iteration-order dependence, no hashing of unordered structures whose
  bucket order affects output (the grid is used only to prune the pair *set*, never to decide an
  output value from bucket traversal order).
- Candidate pairs are canonicalized to `(min(arc_id), max(arc_id))` before any processing, so the
  same pair is never processed twice and is always processed in the same orientation regardless of
  discovery order — this directly satisfies the architecture review's named risk ("false
  independence... determinism must not depend on arc processing order").
- The exact segment-to-segment distance formula is a deterministic closed-form computation (no
  iterative solver, no random sampling).
- Tie-breaking for the witness (§1.3) and for which pair is processed first among independent
  detections (§3.3) both reduce to total orders on `(arc_id, segment_index)` tuples — total orders
  admit no ambiguity.
- The full set of detected gaps is therefore identical for any permutation of `ArcGraph.arcs`'
  iteration order, satisfying the determinism requirement independent of how the caller enumerates
  arcs.

---

## 3. How is a gap repaired?

### 3.1 Alternatives considered

| Approach | Description | Verdict |
|---|---|---|
| **Constrained displacement** (chosen) | Move the witness region's vertices on *both* arcs directly apart, along the local separating direction, by the minimum amount needed to clear the threshold, subject to a hard displacement cap (§4). | Selected — see §3.2. |
| Local projection | Project each arc's witness vertices onto a notional "corridor centerline" offset by `min_gap_pt/2`, snapping both sides to a canonical parallel shape. | Rejected: rewrites *shape*, not just spacing — it can flatten a naturally curved thin feature into two artificial parallel lines, which changes composition beyond what a normalization pass should do, and is harder to bound in terms of maximum shape change (a projection has no natural small-perturbation limit the way a displacement does). |
| Offset (uniform arc offsetting, like a stroke-outline operation) | Offset one or both whole arcs outward by a fixed amount wherever they are within threshold. | Rejected: a whole-arc offset is a global operation — offsetting a long arc to fix one narrow local site perturbs every other point on that arc too, violating the "smallest legal change" principle (§4) and risking new narrow gaps against a *third*, previously-unrelated arc elsewhere along the same offset arc. |
| Vertex redistribution (resample point spacing without moving the arc's shape) | Add/relocate vertices along the existing polyline path to give finer resolution near the gap. | Rejected as the repair mechanism itself (though retained as a *mechanism*, not a policy — see below): redistribution alone does not change `d(a,b)` at all. Two arcs that are geometrically 0.2mm apart remain 0.2mm apart no matter how many vertices represent them. This was the original "insert more points" idea the task explicitly disqualifies — it treats a geometric-proximity problem as a resolution problem, which it is not. |
| Adaptive densification | Same critique as vertex redistribution: densification changes how faithfully a curve is *sampled*, not how far apart two curves *are*. Useful as an enabling substep (see §3.4) but not a repair mechanism on its own. |

### 3.2 Chosen operation: constrained displacement, symmetric, minimum-magnitude

**Definition.** For a confirmed narrow gap `(a, b, witness, d)`:

1. Let `p ∈ a`, `q ∈ b` be the two closest points realizing `d(a,b)` (the closest points on the
   witness segment pair — for a proper segment-segment closest pair this is generally interior to
   both segments, not necessarily existing vertices; see §3.4 for how this is handled without
   inventing new vertices arbitrarily).
2. Let `u = (q − p) / ‖q − p‖` be the unit **separating direction** (from `a`'s witness point toward
   `b`'s).
3. The **required total separation increase** is `Δ = min_gap_pt − d(a, b)` (always positive for a
   confirmed gap).
4. Displace the vertices of `a` nearest the witness point by `−u · Δ/2` and the vertices of `b`
   nearest the witness point by `+u · Δ/2` — **symmetric**, half the correction on each side. This
   is the direct implementation of the "symmetric, order-independent" requirement the architecture
   review demanded: correction does not depend on which arc was "discovered first," because both
   arcs always receive exactly half the fix, computed once from the pair, not from either arc's own
   perspective.
5. The displacement is applied only to the vertex (or, per §3.4, vertices) closest to the witness
   point on each arc, tapering to zero over a small fixed neighborhood (the immediately adjacent 1–2
   vertices on each side receive a linearly-decaying fraction of the same displacement) so the arc's
   shape does not develop a sharp kink at the repair site — this taper is a shape-quality detail, not
   a topology-relevant one, and has no bearing on any proof in §5.

**Why this is the correct choice:**

- It is the only alternative among those considered whose effect is precisely and only "increase
  the clearance to the threshold," with no side effect on the arcs' broader shape, curvature, or
  point density — the minimum possible intervention for the stated goal, which is the same
  "smallest legal change" principle that governs every other geometry-modifying stage in this
  engine (VW's guard rejects rather than over-corrects; `smooth`'s displacement bounding does the
  same).
- Symmetry is what makes it trivially order-independent (§6), unlike moving only one arc (which
  would require an arbitrary "which one moves" rule) or moving by different amounts per side (which
  would require an arbitrary weighting rule).
- It operates in the same category of operation (small bounded vertex displacement) that
  `ENGINE_SPEC §17`'s smoothing stage already performs and bounds — this pass is not introducing a
  new *kind* of geometric intervention to the pipeline, only a new *trigger condition* for an
  already-established idiom (bounded displacement + junction pinning).

### 3.3 Processing order across multiple simultaneous gaps

Confirmed gaps are processed in ascending order of `(min(a.arc_id, b.arc_id), max(a.arc_id,
b.arc_id))` — a fixed total order, independent of detection order (§2.3). After each repair, the
two affected arcs' geometry is updated before evaluating the next gap in order — later gaps
involving an already-repaired arc are measured against its **post-repair** geometry, not a stale
copy. This is deliberately **not** a retry loop (the task's explicit prohibition): it is a
single deterministic pass over a fixed, pre-computed, ordered list of confirmed gaps, each handled
exactly once. No gap is re-examined after being processed, and no new gap-detection scan is
triggered by a repair (unlike the old pinch-repair's re-scan-after-fit loop, which is exactly the
"hidden geometry mutation" pattern being eliminated).

### 3.4 Enabling mechanism: vertex insertion is a substep, not the fix

If the witness closest-point on either arc's segment is not an existing vertex (i.e. it falls
strictly inside a segment between two vertices), a single new vertex is inserted at that exact
closest-point location before displacement is applied, so the displacement has an actual vertex to
move rather than needing to invent a fractional-segment edit. This is the sense in which
densification/point-insertion is used: purely as **plumbing** to give the real fix (displacement) a
vertex to act on, exactly as the task instructs ("do not simply say insert more points... discuss
constrained displacement... explain the geometric operation" — insertion is in service of
displacement, not a substitute for it).

---

## 4. Maximum allowed modification

### 4.1 Largest legal displacement

The displacement applied to any single vertex is bounded by:

```
max_displacement_pt = min(Δ/2, simplify.tolerance_mm · 72/25.4)
```

i.e., **never more than half the required separation increase, and never more than the
`simplify` tolerance already in force.** If enforcing the full `Δ/2` on both sides would require a
displacement larger than the `simplify.tolerance_mm` ceiling, the repair does not proceed partway —
per §8, it is refused outright (fail-fast, not "do what you can").

### 4.2 Why this bound, and why it guarantees fidelity and topology stay intact

- **Fidelity (I1).** I1 is a face-vs-`component_map` pixel-agreement measure. Any per-vertex
  displacement strictly bounded by the same `tolerance_mm` that `simplify` is already permitted to
  perturb geometry by cannot, by construction, introduce a *new* category of fidelity risk —
  it is bounded by an error budget the fidelity gate (I1 ≥ 99%) already has to tolerate from
  `simplify` alone. This is precisely the "shared ceiling, not shared budget" design frozen in the
  `geometry_normalize` architecture review (§2 of that document): each pass owns its own threshold,
  but none may exceed the `simplify.tolerance_mm` ceiling, which keeps the *combined* worst case
  boundable in the same terms the engine already reasons about for `simplify` alone.
- **Topology (I3).** The bound does not by itself guarantee topology preservation — that guarantee
  comes from the *guard* in §5, not from the magnitude bound alone. The magnitude bound's role is
  narrower and specific: it guarantees that *if* the guard passes, the resulting shape change is
  small enough to be classified as a normalization (a bounded local perturbation) rather than a
  redraw — consistent with MATH_SPEC §6.2's own framing ("a boundary may legally move up to the
  guard tolerance, but it may never cross other geometry or invert its faces" — two separate
  conditions, magnitude and sidedness, both required, neither sufficient alone).

---

## 5. Topology proof

**Claim.** Applying the constrained displacement of §3 to the witness vertices of a confirmed gap
`(a, b)` cannot create overlaps, create gaps, invert orientation, or disconnect regions — **provided
the sidedness guard below passes**; if the guard fails, the repair is refused (§8), not attempted
with a smaller step (no retry).

### 5.1 The guard

Before committing a displacement to a vertex `v → v'` on arc `a` (symmetrically for `b`), test:

> **Sidedness admissibility.** The candidate move is admissible iff the closed swept region between
> `v` and `v'` (the thin quadrilateral/triangle traced by the vertex moving from its old to new
> position, together with its two adjacent segments before and after) contains no vertex or segment
> of *any other* arc (including `a`'s or `b`'s own non-adjacent segments), and does not cross the
> arc's own immediately adjacent segments in a way that would invert their orientation.

This is **exactly** MATH_SPEC §8.2's sidedness/topology guard formula, applied to a displacement
candidate instead of a vertex-removal candidate: "the removal is admissible iff the closed
[region] contains no vertex or segment crossing of any *other* geometry." Gap Repair does not
invent a new topology-safety predicate — it reuses the engine's one sanctioned predicate family
(`orient`, MATH_SPEC §7.2, "the entire strategy" for concentrating robustness in one place) against
the same uniform spatial hash already built for broad-phase detection (§2.1), queried at the
displaced location.

### 5.2 Why each failure mode is excluded

- **No overlaps.** An overlap would require the displaced segment of `a` (or `b`) to cross into the
  interior of a third region's boundary or another arc's segment. The guard explicitly tests
  exactly this (foreign vertex/segment inside the swept region) before committing the move; a move
  that would create an overlap fails the guard and is refused.
- **No gaps.** A "gap" in the topological sense (I3's watertightness) would require an arc endpoint
  to move away from its junction, breaking the shared-edge property MATH_SPEC §6.1 relies on
  ("Because arcs are *shared*... gaps/overlaps are representationally impossible pre-simplification;
  the checks exist to catch violations *introduced* by geometry-modifying stages"). Junction
  endpoints are categorically excluded from displacement: §1.1's exclusion rule means any vertex
  that is a shared junction endpoint between two or more arcs is never a witness-adjacent vertex
  eligible for movement (the excluded-pair rule already removes junction-adjacent pairs from
  consideration entirely, and separately, junction vertices themselves are pinned — the same
  "junction pinning" idiom already established by `simplify` and `smooth`, ENGINE_SPEC §16/§17).
  With junctions immovable, the shared-edge property at every junction is untouched, and no new gap
  can open.
- **No orientation inversion.** Orientation of a ring is a shoelace-sign property (MATH_SPEC §7.1).
  The guard's second clause (does not cross the arc's own adjacent segments in a way that would
  invert their local orientation) is precisely a bounded `orient`-predicate check on the two
  segments neighboring the displaced vertex — the same predicate MATH_SPEC §7.2 calls "the only
  predicate that requires filtering," reused here rather than a new heuristic. A displacement that
  would flip the sign of either adjacent segment's local turn is, by definition, caught and refused.
- **No disconnection.** Regions in this engine are connected components of `component_map`
  (pixel-domain, frozen long before `topology`/`arcgraph`/`simplify` ever run). Gap Repair operates
  purely on `ArcGraph.arcs`' point coordinates — it never touches region adjacency, `Face.outer_walk`
  /`hole_walks` (which reference `arc_id`s, not geometry), or region ids. A displacement bounded to
  not cross other geometry (guard, above) cannot change which `arc_id`s bound which face, and since
  faces are defined purely by arc-id walks, the face structure — and therefore region
  connectivity — is invariant by construction, not merely by the guard. This is the strongest of
  the four guarantees: it holds even before considering the guard, purely from the fact that
  `Face` objects are never read or written by this pass (see the `geometry_normalize` architecture
  review's requirement that the pass's core transform cannot accept `Face` at all).

### 5.3 What happens when the guard fails

The specific pair's repair is **skipped** (§8) — not retried at a smaller magnitude, not
approximated. This is a deliberate, explicit divergence from the old pinch-repair's tolerance-halving
retry loop, which the task and the accepted architecture review both name as the exact anti-pattern
to eliminate.

---

## 6. Determinism proof

**Claim.** Same `ArcGraph` input, same `geometry_normalize` config, always produces bit-identical
output `ArcGraph`.

1. **Detection is deterministic** (proven in §2.3): candidate generation, exact distance
   computation, and witness selection all reduce to pure functions of arc geometry plus fixed total
   orders on `(arc_id, segment_index)` — no hashing of unordered collections in any
   output-affecting path, no wall-clock, no PRNG (this pass introduces no stochastic step at all,
   consistent with the engine's I2 contract that only genuinely stochastic steps use the seeded-PRNG
   pattern — this pass has nothing to seed).
2. **The confirmed-gap list is deterministic**: it is the filtered, exactly-computed subset of a
   deterministically-generated candidate set (§2.3), so the *list itself* — its contents and,
   because of the canonicalized `(min,max)` ordering, its processing order (§3.3) — is fixed for a
   given input.
3. **Each repair is deterministic**: the separating direction `u`, the split `Δ/2`, the taper
   coefficients, and the guard predicate (`orient`, exact/filtered per MATH_SPEC §7.2) are all
   closed-form computations on fixed inputs — no iterative solver with a convergence-dependent
   stopping point, no floating-point reduction over an unordered set.
4. **Sequential application preserves determinism**: because gaps are processed in a fixed total
   order (§3.3) and each repair immediately updates the two affected arcs before the next gap in
   the list is processed, the *sequence* of intermediate states is fixed — there is no "if repair A
   ran before repair B, the numeric result differs from B-before-A" ambiguity, because the order is
   pinned, not left to scheduler/iteration whim.
5. **No retry, no re-scan**: because this pass performs exactly one detection sweep and one ordered
   repair pass with no feedback loop (unlike the old design), there is no risk of a repair changing
   the *set* of confirmed gaps mid-pass in a way that depends on how many repairs happened to run
   first in a given execution — the confirmed-gap list is fixed before any repair is applied.

Conclusion: every step from input to output is a composition of pure, order-fixed functions;
therefore the composition is itself a pure function of `(ArcGraph, config)`, and two runs with
identical inputs produce identical outputs.

---

## 7. Complexity

- **Broad phase.** Building the spatial grid: O(A) where A = total arc count (each arc inserted by
  its bounding box). Candidate pair generation: expected O(A) under the engine's existing bounded
  boundary-density assumption (same assumption MATH_SPEC §8.2 already relies on for the VW guard:
  "boundary density is bounded by construction: arcs are disjoint except at junctions").
- **Narrow phase.** For each candidate pair, O(P_a · P_b) segment-to-segment comparisons where P_a,
  P_b are each arc's (post-simplify, already vertex-reduced) point counts. Expected small since
  `simplify` has already reduced staircase vertex counts by the documented "≥80% reduction"
  (ENGINE_SPEC §16) before this pass ever runs.
- **Worst case.** If the broad-phase density assumption is violated (a pathological fixture with
  many arcs clustered in one small region — e.g. extremely fine parallel hatching), candidate count
  degrades toward O(A²), and per-pair cost is bounded by the longest two arcs' point counts. This
  worst case is explicitly named as a benchmark tracking concern (§10), not assumed away.
- **Repair phase.** O(1) amortized per confirmed gap (a fixed number of vertices displaced, one
  guard query each against the same spatial hash already built for broad phase).
- **Memory.** O(A + P_total) for the spatial grid and per-arc point storage — same order as the
  existing VW pass's own memory bound (MATH_SPEC §8.2: "Memory. O(P_total)").

---

## 8. Failure conditions

Following the engine's established idiom (VW: "Failure modes. None; skipping all removals is always
legal" — ENGINE_SPEC §16) — **skip, never retry, fail only on a genuine precondition violation**:

- **Repair (default path).** A confirmed gap whose guard (§5.1) passes at the full `Δ/2` symmetric
  displacement (bounded per §4.1) is repaired.
- **Skip (legal, expected, not an error).** A confirmed gap whose guard fails at the required
  displacement is left unrepaired — recorded (see provenance counts in the `geometry_normalize`
  architecture review) but not escalated. Skipping is always topologically legal, exactly as VW's
  own guard treats a blocked removal: the arc pair remains exactly as narrow as it was, which is the
  same shape the pipeline would have produced with this pass disabled — never worse than doing
  nothing.
- **Skip.** A confirmed gap where the required `Δ/2` exceeds `max_displacement_pt` (§4.1) — the
  fix is refused outright rather than applied partially, because a partial fix does not clear the
  threshold and provides no benefit while still incurring geometric change; this is a "do nothing or
  do the full correct thing" pass, never "do part of it."
- **Fail (`StageError`, engine-wide convention — `topology.py`'s "never repaired silently").** Only
  if a genuine precondition of the *input* is violated — e.g. `ArcGraph`'s own constructor
  invariants somehow failed to hold before this pass runs (should be structurally impossible given
  `ArcGraph.__post_init__`, but this pass does not re-derive or bypass that check, it inherits it),
  or a config value violates the frozen ceiling rule (`min_gap_mm > simplify.tolerance_mm`) —
  the latter is a `ConfigError` at stage construction, not a runtime `StageError`, following the
  exact pattern `MergeTinyStage`/`CurveFitStage` already use for self-validating their own config at
  `__init__` time, before any geometry is ever touched.
- **Never:** a retry loop, a tolerance-halving fallback, or a straight-line degradation of the kind
  the old pinch-repair used. These are explicitly excluded by design, not merely undocumented.

---

## 9. Property tests required

1. **Symmetry.** For a synthetic fixture with one narrow gap between two arcs, the displacement
   magnitude applied to each side is exactly equal (within floating-point epsilon) regardless of
   which arc is labeled `a` vs `b` in the input ordering.
2. **Order independence / determinism.** Running the pass on `ArcGraph.arcs` in its natural order
   vs. a shuffled-then-restored-id order (i.e., permuting internal iteration while preserving the
   dense `arc_id` contract) produces byte-identical output `ArcGraph`.
3. **Idempotence.** Running the pass twice in sequence (`normalize(normalize(g)) ==
   normalize(g)`) produces identical output the second time — once a gap is repaired to clear
   `min_gap_pt`, a second pass detects no gap at that site and makes no further change. (Note: this
   is a *stronger* guarantee than `simplify`'s own documented policy of not requiring idempotence,
   ENGINE_SPEC §17 — worth flagging explicitly as a design decision: Gap Repair should target
   idempotence because, unlike smoothing, "already fixed" is a crisply decidable predicate
   (`d(a,b) ≥ min_gap_pt`), not a matter of degree.)
4. **Bounded displacement.** No vertex in the output differs from its input position by more than
   `max_displacement_pt` (§4.1), for every fixture in the golden/property-test corpus.
5. **Topology preservation.** For every property-test fixture, `topology`/`arcgraph`'s own
   independent re-derivation (Euler identity, area-partition identity, per-arc-reference count ≤ 2)
   holds on the pre-normalize and post-normalize `ArcGraph` alike — i.e., feeding the repaired
   `ArcGraph` back through the existing `_check_arc_references`-style checks never fails.
6. **Sidedness preservation.** For every displaced vertex, the two adjacent segments' `orient` sign
   is unchanged before and after displacement (direct test of §5.2's orientation-inversion claim).
7. **Junction immovability.** No vertex that is a shared endpoint between two or more arcs (a
   junction) is ever displaced by this pass, for any fixture.
8. **Guard soundness (negative test).** A constructed fixture where the "correct" displacement would
   cross a third arc must result in that gap being **skipped**, not partially repaired or repaired
   by crossing the third arc.
9. **Ceiling enforcement.** Constructing a `geometry_normalize` config with `min_gap_mm >
   simplify.tolerance_mm` raises `ConfigError` at stage construction, before any `ArcGraph` is
   processed.
10. **No-op on clean input.** Running the pass on an `ArcGraph` with no pair violating `min_gap_pt`
    returns an output identical to the input (same object identity for every arc, per the
    `geometry_normalize` architecture's "unchanged arcs pass through as the same object" contract).
11. **Monotone improvement.** For every repaired pair, post-repair `d(a,b) ≥ min_gap_pt` exactly
    (not merely "improved") — the repair either fully clears the threshold or is not attempted
    (ties to §8's "do the full fix or nothing" rule).

---

## 10. Benchmark requirements

Following the existing `benchmarks/perf` / `benchmarks/quality` split (as already used by
`merge_tiny`, ENGINE_SPEC's per-module benchmark rows):

- **Repaired-gap count.** On the golden fixture ladder, report count of confirmed gaps detected vs.
  repaired vs. skipped, per fixture category (the thin-linear-detail categories this whole sprint
  arc exists for: cables, branches, hair, decorative borders). A near-zero repaired count on
  fixtures with no thin features is a required sanity check (a non-zero count there indicates a
  threshold or symmetry defect, per the earlier architecture review's benchmark strategy).
- **Runtime.** Wall-clock on the golden ladder at increasing arc counts, with an explicit budget
  comparable to sibling stages — `merge_tiny`'s documented "20,000 → ~800 regions ≤ 1.0 s" is the
  reference scale; Gap Repair's budget should be stated once real fixture arc-counts are measured,
  but must include the worst-case density scenario named in §7 (dense parallel hatching) as a
  distinct benchmark row, not folded into the average case.
- **Fidelity delta.** I1 fidelity score before/after enabling this pass, across the full golden
  dataset — expected ≈ 0 delta given the bounded-displacement argument in §4.2; any measured delta
  beyond noise is a signal the ceiling relationship to `simplify.tolerance_mm` needs revisiting.
- **Printability improvement.** The actual target metric this sprint exists to move: count of
  faces failing I4 (inscribed-diameter-based printability) before/after, and, once the earlier
  distance-transform-based printability validator extensions land, minimum-printable-width
  statistics before/after — the direct proof that Gap Repair converts previously-unprintable thin
  double-wall features into single, cleanly separated, printable boundaries.
- **Bezier self-intersection regression check.** Re-run the exact fixture set that originally
  exposed the old pinch-repair's necessity; confirm zero `bezier`-stage self-intersections occur
  downstream with Gap Repair enabled and the old pinch-repair code removed — this is the closing
  proof that the redesign actually fixes the regression that motivated it, not merely that it is
  architecturally cleaner.

---

## Summary of scope boundary (for the implementing engineer)

This document defines Pass 3 (`minimum_gap_enforcement`) only. It assumes Pass 1
(duplicate-point cleanup) and Pass 2 (spike removal) have already run on the arc set it receives,
per the fixed execution order frozen in the `geometry_normalize` architecture review. It never
reads `Face`, never reads `component_map`, never reads or produces `Curve`/`CurveSet`/`BezierSegment`,
and never retries with a different tolerance. Its only output is a new or pass-through `Arc` per
input arc, assembled into a new `ArcGraph` with the same `faces` and `work_scale`, re-stamped
`Provenance`.
