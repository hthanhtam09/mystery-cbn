# Mystery Color-by-Number Engine — Mathematical Specification

**Status:** v1.0 — authoritative mathematical reference. Companion to [ENGINE_SPEC.md](ENGINE_SPEC.md) (algorithms & modules) and [ARCHITECTURE.md](ARCHITECTURE.md) (structure). Every formula, constant, and predicate used by the engine is defined here; ENGINE_SPEC.md references these definitions and must not restate them differently.
**Audience:** graduate-level engineers. **Rule:** changing any definition here is a versioned event requiring an ADR.

---

## 0. Notation and global conventions

| Symbol | Meaning |
|---|---|
| `⟦·⟧` | Iverson bracket (1 if true, 0 otherwise) |
| `‖·‖` | Euclidean norm unless subscripted |
| `Λ ⊂ ℤ²` | pixel lattice of the working raster, `H×W` |
| `Λ* ` | crack (dual corner) lattice, `(H+1)×(W+1)` |
| `L(p)` | region id of pixel `p`; `ℓ(p)` palette label |
| `∂` | boundary operator |
| `α(n)` | inverse Ackermann function |

All floating-point computation is IEEE-754 binary64 unless a section states otherwise. Reductions over unordered collections must fix a total order first (stable id sort) — determinism is a system invariant (ENGINE_SPEC §1.3), so *every* formula below must be evaluated in a reproducible order.

---

## 1. Coordinate systems

### 1.1 Raster frame `𝖱`

**Definition.** Pixel `p = (i, j)` (row `i` ∈ [0, H), column `j` ∈ [0, W)) is a unit square; its center has coordinates `(x, y) = (j, i)` with x rightward, y **downward**, origin at the top-left pixel center.
**Formula.** Pixel square: `S(p) = [j − ½, j + ½] × [i − ½, i + ½]`.
**Variables.** `i, j` integers; `x, y` reals. **Units.** working pixels (px).
**Complexity.** O(1) per conversion. **Numerical stability.** Exact (integers and halves are dyadic).
**Failure cases.** None; frame is total.
**Implementation notes.** y-down matches array indexing and SVG; no intermediate y-up frame may exist anywhere in the raster or vector pipeline (a single flip lives in the PDF canvas transform).

### 1.2 Crack frame `𝖢`

**Definition.** Corners of pixel squares. Corner `c = (u, v) ∈ Λ*` sits at raster coordinates `(v − ½, u − ½)`. A **crack edge** is the unit segment between 4-adjacent corners; it separates two pixel squares (or a pixel from the exterior, treated as region id −1).
**Formula.** Vertical crack between `p=(i,j)` and `q=(i,j+1)` occupies `x = j + ½, y ∈ [i − ½, i + ½]`; it exists in the boundary set iff `L(p) ≠ L(q)`.
**Variables.** `u ∈ [0, H]`, `v ∈ [0, W]`. **Units.** px.
**Complexity.** Enumeration of all crack edges: O(N). **Numerical stability.** Exact if corners are stored as **doubled integers** `(2u−1, 2v−1)`; the engine mandates doubled-integer arithmetic through §7 (topology) so that shoelace areas and orientation tests are exact integer computations.
**Failure cases.** None.
**Implementation notes.** All of ENGINE_SPEC §13–15 operates in `𝖢` with doubled integers; conversion to floats happens only at the §1.3 map below.

### 1.3 Page frame `𝖯` and the working map

**Definition.** Physical page coordinates in points, origin at the top-left of the trim box, y down.
**Formula.** With content box origin `(m_x, m_y)` pt and scale `s` (pt/px):

```
Φ: 𝖱 → 𝖯,   Φ(x, y) = (m_x + (x + ½)·s,  m_y + (y + ½)·s)
s = min( C_w / W , C_h / H )      (aspect-preserving letterbox)
m_x = margin_pt + ( C_w − sW )/2,  m_y analogous
```

where `C_w × C_h` is the content box (page minus margins minus legend band) in pt.
**Variables.** `s` pt/px; `m` pt. **Units.** pt (1 pt = 1/72 in = 25.4/72 mm).
**Complexity.** O(1)/point. **Numerical stability.** `s` is a single multiplication; relative error ≤ 2 ulp. Because Φ is affine and orientation-preserving, all topological predicates proven in `𝖢` are preserved under Φ (affine maps preserve incidence and sidedness); nothing needs re-proving after scaling except float-tolerance area sums.
**Failure cases.** `C_w ≤ 0` or `C_h ≤ 0` → `ConfigError` (config cross-field rule).
**Implementation notes.** Φ is applied exactly once (ENGINE_SPEC §15.5). The +½ centers pixel content in its square so the border loop maps to the exact content-box rectangle.

---

## 2. Units

**Definition.** Three unit systems: millimetres (user-facing config), points (vector/page domain), working pixels (raster/graph domain).
**Formulas.**

```
pt = mm · 72 / 25.4          mm = pt · 25.4 / 72
px = mm · ppmm,   ppmm = W / content_width_mm     (working pixels per printed mm)
A_min[px²] = π (d_min/2)² · ppmm²                  (printability area floor)
```

**Variables.** `d_min` mm; `ppmm` px/mm. **Units.** as stated.
**Complexity.** O(1). **Numerical stability.** All conversion factors are exact rationals scaled by binary64; error ≤ 1 ulp per conversion. The single-conversion-site rule (`foundation/units` only) prevents error *accumulation* by construction.
**Failure cases.** Division by zero impossible (config validation enforces positive page dims).
**Implementation notes.** Quality thresholds are stated in mm and converted once per run; never store a threshold in px in any artifact (resolution independence).

---

## 3. Color spaces

### 3.1 sRGB → linear RGB

**Definition.** Inverse of the IEC 61966-2-1 transfer function.
**Formula.** For channel `c ∈ [0,1]`:

```
lin(c) = c / 12.92                      if c ≤ 0.04045
lin(c) = ((c + 0.055) / 1.055)^2.4      otherwise
```

**Variables.** `c` normalized channel. **Units.** dimensionless.
**Complexity.** O(1)/channel; vectorized O(N). **Numerical stability.** `x^2.4 = exp(2.4·ln x)`; for `x ∈ [0.04045, 1]` relative error < 4 ulp. The piecewise joint is continuous to ~1e−7 — acceptable; do not "smooth" the joint (destroys standard compliance).
**Failure cases.** Inputs outside [0,1] (from filtering overshoot) must be clamped **before** linearization.
**Implementation notes.** Precompute a 256-entry LUT for 8-bit sources (exact for those inputs); direct evaluation for float inputs.

### 3.2 Linear RGB → XYZ → CIELAB (D65, 2°)

**Definition.** Linear map to CIE XYZ then the CIELAB nonlinearity, white point D65.
**Formula.**

```
[X Y Z]ᵀ = M · [R G B]ᵀ,   M = sRGB D65 matrix (IEC 61966-2-1):
    ⎡0.4124564 0.3575761 0.1804375⎤
    ⎢0.2126729 0.7151522 0.0721750⎥
    ⎣0.0193339 0.1191920 0.9503041⎦
f(t) = t^(1/3)                    if t > (6/29)³
f(t) = t/(3(6/29)²) + 4/29        otherwise
L* = 116 f(Y/Yn) − 16,  a* = 500 (f(X/Xn) − f(Y/Yn)),  b* = 200 (f(Y/Yn) − f(Z/Zn))
(Xn, Yn, Zn) = (0.95047, 1.0, 1.08883)
```

**Variables.** `L* ∈ [0,100]`, `a*, b*` ≈ [−128, 127]. **Units.** CIELAB units (perceptually ~1 JND ≈ 2.3 ΔE76).
**Complexity.** O(N). **Numerical stability.** Cube root is well-conditioned; the piecewise `f` joint is C¹ by construction. Round trips sRGB→LAB→sRGB must satisfy max channel error ≤ 1e−4 (property-tested bound, ENGINE_SPEC §10).
**Failure cases.** Out-of-gamut LAB (from center averaging) maps to RGB outside [0,1]: clamp per channel after inverse transform; clamping error is recorded nowhere because palette centers are means of in-gamut points (convexity of the gamut in linear RGB bounds the excursion to the nonlinearity's curvature — empirically < 0.5 ΔE00).
**Implementation notes.** Matrix constants are pinned above to 7 digits; implementations must use these exact values (not library defaults, which vary in the 7th digit and break byte-determinism).

---

## 4. Distance metrics

### 4.1 ΔE76

**Definition/Formula.** `ΔE76(x, y) = ‖x − y‖₂` in LAB.
**Variables.** LAB triples. **Units.** ΔE.
**Complexity.** O(1). **Numerical stability.** Trivial; use `hypot`-style guarded sums only if inputs can be huge (they cannot; skip).
**Failure cases.** None.
**Implementation notes.** Permitted only in k-means inner loops (ENGINE_SPEC §1.3); never a pass/fail gate.

### 4.2 CIEDE2000 (ΔE00)

**Definition.** The CIE 2000 perceptual color-difference formula with lightness/chroma/hue weighting and rotation term.
**Formula (normative form).** With `C̄` the mean chroma of the pair, `G = ½(1 − √(C̄⁷/(C̄⁷+25⁷)))`, `a′ = (1+G)a`, then `C′, h′` polar forms:

```
ΔE00² = (ΔL′/(k_L S_L))² + (ΔC′/(k_C S_C))² + (ΔH′/(k_H S_H))² + R_T (ΔC′/(k_C S_C))(ΔH′/(k_H S_H))
S_L = 1 + 0.015(L̄′−50)² / √(20+(L̄′−50)²),  S_C = 1 + 0.045C̄′,  S_H = 1 + 0.015C̄′T
T = 1 − 0.17cos(h̄′−30°) + 0.24cos(2h̄′) + 0.32cos(3h̄′+6°) − 0.20cos(4h̄′−63°)
R_T = −2√(C̄′⁷/(C̄′⁷+25⁷)) · sin(60°·exp(−((h̄′−275°)/25)²))
k_L = k_C = k_H = 1
```

with `ΔH′ = 2√(C₁′C₂′) sin(Δh′/2)` and the standard hue-mean case analysis (Sharma, Wu & Dalal 2005, Eqs. 7–14).
**Variables.** LAB pairs. **Units.** ΔE00 (≈1 = JND).
**Complexity.** O(1), ~40 flops. **Numerical stability.** The hue arithmetic has branch discontinuities at `C′ = 0`; define `h′ = 0` when `C′ = 0` (Sharma convention). `C̄⁷` overflows float32 — compute in float64 always.
**Failure cases.** Naïve `atan2` degree wrapping produces ±360° errors in `Δh′`; the Sharma case analysis is normative, and the implementation must reproduce the 34-pair Sharma test dataset to 1e−4.
**Implementation notes.** This is the only gate metric. Vectorize over pairs; for K ≤ 64 the full K×K table is 4 KB — precompute once per palette.

### 4.3 Discrete Hausdorff distance

**Definition.** For point sets `P, Q`: `d_H = max( max_p min_q ‖p−q‖, max_q min_p ‖p−q‖ )`.
**Units.** px or pt per context. **Complexity.** O(|P|·|Q|) naïve; O((|P|+|Q|) log) with spatial hashing — acceptable at fixture sizes.
**Numerical stability.** Exact up to distance rounding. **Failure cases.** Empty sets — undefined; callers must guard.
**Implementation notes.** Used only in tests (boundary-displacement bounds, ENGINE_SPEC §8/§16), never in the pipeline.

---

## 5. Graph theory

### 5.1 Region adjacency graph (RAG)

**Definition.** Undirected weighted graph `G = (V_R, E_R)`; vertices are 4-connected regions; edge `{a,b}` exists iff ∃ 4-adjacent pixel pair `(p,q)` with `L(p)=a, L(q)=b`.
**Formulas.**

```
w_len(a,b)  = |{ crack edges separating a and b }|            (shared boundary, px)
w_col(a,b)  = ΔE00( μ_LAB(a), μ_LAB(b) )
perim(a)    = Σ_b w_len(a,b) + border_len(a)
```

**Variables.** μ_LAB = pixel-count-weighted LAB mean of the region's palette color (palette color, not raw pixels — regions are label-homogeneous). **Units.** px, ΔE00.
**Complexity.** Construction O(N); |E_R| ≤ 2N. **Memory** O(R + |E_R|).
**Numerical stability.** `w_len` is an exact integer. **Failure cases.** None on valid label maps.
**Implementation notes.** The identity `Σ_e w_len(e) + Σ_r border_len(r) = B` (total crack-edge count) is asserted in CI (double-entry check with §7).

### 5.2 Planar boundary multigraph

**Definition.** `G* = (V_J, A)`: vertices are junctions (crack corners where ≥3 region ids meet in the 2×2 pixel block, plus the 4 page corners), edges are **arcs** (maximal crack paths with constant (left, right) region pair). Faces `F` are regions plus the exterior.
**Formula (Euler).** For the connected planar multigraph induced on the sphere:

```
V_J − A + F = 1 + C
```

where `C` = number of connected components of `G*` (each island boundary adds a component; the classical `V−E+F=2` holds for `C=1`). The engine asserts the general form with `C` counted by union-find over arc endpoints, closed arcs counting as their own component with `V=1, A=1, F` contribution via their two sides.
**Complexity.** Face walking O(A) (junction degree ≤ 4 on the crack lattice). **Numerical stability.** Pure combinatorics — exact.
**Failure cases.** Any Euler violation is a `StageError` (constructive bug), never repairable.
**Implementation notes.** Angular order of half-arcs at a junction is one of {E, N, W, S} — comparisons are table lookups, not `atan2` (exactness; no ties possible).

---

## 6. Topology

### 6.1 The crack complex

**Definition.** The label map induces a CW-complex on `[−½, W−½] × [−½, H−½]`: 0-cells = junction corners, 1-cells = arcs, 2-cells = faces. **Watertightness** (invariant I3) is the statement that this complex is a valid planar partition: 2-cells are pairwise disjoint open sets whose closures cover the rectangle.
**Formula (partition test).** With shoelace area `A(f)` (holes negative, §7.1):

```
Σ_f  A(f) = W·H            (exact, in doubled-integer arithmetic, pre-scaling)
∀ arc a: |faces(a)| = 2     (the exterior counts as a face)
```

**Complexity.** O(A + total vertices). **Numerical stability.** Exact in `𝖢`; after Φ the identity is re-checked with relative tolerance 1e−4 (±0.01 %).
**Failure cases.** A partition violation post-simplification means the §8.3 guard failed — FATAL, no repair.
**Implementation notes.** Because arcs are *shared* (each drawn once, bounding exactly two faces), gaps/overlaps are representationally impossible pre-simplification; the checks exist to catch violations *introduced* by geometry-modifying stages (§9, §10 of this doc).

### 6.2 Sidedness invariant

**Definition.** For every arc with pair (left ℓ, right r): every point of the arc's polyline, offset by +ε along the local left normal, must lie in face ℓ (resp. −ε in r).
**Formula.** For consecutive vertices `v_k, v_{k+1}` with direction `d`, left normal `n = (d_y, −d_x)/‖d‖` (y-down frame).
**Complexity.** Sampled check O(vertices). **Numerical stability.** Use ε = ¼ px; point-in-face by winding of the exact face walk.
**Failure cases.** Simplification/smoothing that flips a segment's sidedness — prevented a priori by the guards (§9.2, §10) and re-proved by validation.
**Implementation notes.** Sidedness, not distance, is the correct invariant: a boundary may legally move up to the guard tolerance, but it may never cross other geometry or invert its faces.

---

## 7. Polygon operations

### 7.1 Signed area (shoelace)

**Definition/Formula.** For a closed polygon `v_0 … v_{n−1}`:

```
2·A = Σ_k ( x_k · y_{k+1} − x_{k+1} · y_k )        (indices mod n)
```

In the y-down frame, **clockwise** loops (as produced by the left-hand-rule trace of a region's outer boundary) have positive area; hole loops come out negative — no sign fix-ups.
**Units.** px² (doubled-int: 4·A is the exact integer) or pt².
**Complexity.** O(n). **Numerical stability.** Exact in doubled integers up to |coord| ≤ 2²⁵ (far above 6000-px working rasters). In floats, use the translated form `Σ (x_k−x̄)(y_{k+1}−ȳ) − …` to avoid catastrophic cancellation for far-from-origin polygons.
**Failure cases.** Self-intersecting input yields algebraic (meaningless) area — inputs here are simple by construction.
**Implementation notes.** All I3 area identities are computed pre-scaling in integers; post-scaling only for the tolerance re-check.

### 7.2 Orientation predicate

**Definition/Formula.** `orient(a,b,c) = sign( (b−a) × (c−a) ) = sign((b_x−a_x)(c_y−a_y) − (b_y−a_y)(c_x−a_x))`.
**Complexity.** O(1). **Numerical stability.** Exact for doubled-integer inputs. For float inputs (post-smoothing geometry) use the Shewchuk-style semi-static filter: if `|det| > 4ε_mach·(|terms sum|)` trust the sign, else re-evaluate with extended precision (two-sum/Dekker). This is the **only** predicate that requires filtering.
**Failure cases.** Collinear triples return 0 — callers must handle the degenerate branch explicitly (no epsilon-fudged "almost collinear").
**Implementation notes.** Point-in-polygon, segment intersection, and the simplification guard all reduce to `orient`; concentrating robustness here is the entire strategy.

### 7.3 Segment intersection test

**Definition.** Segments `ab`, `cd` properly intersect iff `orient(a,b,c)·orient(a,b,d) < 0` and `orient(c,d,a)·orient(c,d,b) < 0`; endpoint/collinear cases via on-segment tests.
**Complexity.** O(1); all-pairs via spatial hash O(S) expected (cell size = max segment length).
**Numerical stability.** Inherits §7.2. **Failure cases.** Near-parallel near-touching segments are decided by the filtered predicate — no tolerance parameter exists, by design.
**Implementation notes.** Used by the topology validator (ENGINE_SPEC §25.2) and the VW guard.

### 7.4 Point-in-polygon (winding)

**Definition/Formula.** Winding number `ω(p) = (1/2π) Σ_k Δθ_k` computed combinatorially by signed edge crossings of the upward ray; `ω ≠ 0` ⇔ inside (faces are simple, so ω ∈ {0, ±1}).
**Complexity.** O(n) per query; with the face's bbox pre-check, amortized far less.
**Numerical stability.** Crossing tests reduce to `orient`. **Failure cases.** Query on the boundary: defined as *inside* for label anchoring (consistent tie rule).
**Implementation notes.** Even-odd equals winding for simple polygons; the engine standardizes on winding with explicit hole rings subtracted.

---

## 8. Simplification mathematics

### 8.1 Visvalingam–Whyatt effective area

**Definition.** The importance of interior vertex `v_k` is the area of the triangle formed with its current neighbors.
**Formula.**

```
EA(v_k) = ½ | (v_{k−1} − v_k) × (v_{k+1} − v_k) |
Remove v_k while EA(v_k) < ε,   ε = (τ · 72/25.4)²  pt²,  τ = simplify tolerance in mm (default 0.15)
```

After removal, recompute EA of the two neighbors (their triangles changed).
**Variables.** τ mm; ε pt². **Units.** area.
**Complexity.** O(P log P) with a min-heap and lazy deletion. **Memory** O(P).
**Numerical stability.** Cross product of short vectors — well-conditioned; areas near ε are decided in float64 (no exactness needed: the threshold is a quality knob, not a topological predicate).
**Failure cases.** Monotone removal can collapse an arc to its chord; legal (guards below protect topology, not shape).
**Implementation notes.** The staircase from crack tracing consists of unit right triangles with EA = ½ px² ≈ (0.5·s²) pt² — far below ε at default resolutions, hence the ≥80 % reduction guarantee (ENGINE_SPEC §16).

### 8.2 Sidedness/topology guard

**Definition.** Vertex removal replaces the wedge `(v_{k−1}, v_k, v_{k+1})` by the chord; the removal is admissible iff the closed triangle `T = (v_{k−1}, v_k, v_{k+1})` contains no vertex or segment crossing of any *other* geometry (other arcs, this arc's non-adjacent segments, label anchors).
**Formula.** Containment/crossing via §7.2–7.4 against a uniform spatial hash with cell size `2τ_pt`.
**Complexity.** Expected O(1) candidates per removal (boundary density is bounded by construction: arcs are disjoint except at junctions and simplification only shrinks the occupied cells).
**Numerical stability.** Filtered predicates (§7.2).
**Failure cases.** A guard that queries only vertices (not segment crossings) misses thin spikes piercing T — the segment test is mandatory.
**Implementation notes.** Rejected vertices are marked permanently unremovable in this pass (no livelock); arcs processed in id order for determinism.

### 8.3 Douglas–Peucker (non-default, documented for comparison)

**Definition.** Recursively keep the vertex of max perpendicular distance from the chord until below tolerance `δ`.
**Formula.** `d(v, chord ab) = |(b−a) × (v−a)| / ‖b−a‖`.
**Complexity.** O(P log P) expected, O(P²) worst.
**Why not default.** DP preserves extremal points, i.e., precisely the staircase spikes VW removes first; its L∞ tolerance correlates worse with perceived smoothness than VW's area criterion. Kept in the registry for CAD-like rectilinear inputs.

---

## 9. Bézier mathematics

### 9.1 Cubic Bézier form

**Definition.** `B(t) = Σ_{i=0}^{3} b_i B_i³(t)`, `B_i³(t) = C(3,i) tⁱ (1−t)^{3−i}`, `t ∈ [0,1]`, control points `b_0..b_3`.
**Derivatives.** `B′(t) = 3 Σ_{i=0}^{2} (b_{i+1} − b_i) B_i²(t)`; end tangents `B′(0)=3(b_1−b_0)`, `B′(1)=3(b_3−b_2)`.
**Units.** pt. **Complexity.** O(1) eval (de Casteljau, 6 lerps).
**Numerical stability.** De Casteljau is forward-stable (convex combinations only); prefer it over the power-basis Horner form for evaluation and subdivision.
**Failure cases.** `b_1 = b_0` or `b_2 = b_3` gives zero end tangent (cusp risk); the fitter forbids zero tangents (§9.2 step constraints).
**Implementation notes.** Flattening error bound for subdivision: for a cubic with second-difference bound `M = max(‖b_0−2b_1+b_2‖, ‖b_1−2b_2+b_3‖)`, the chord deviation after k bisections is ≤ `(3/4)·M / 4^k` — used to pick the §24 flattening depth for tolerance 0.1 px in closed form (no adaptive recursion needed for the preview).

### 9.2 Least-squares fitting (Schneider)

**Definition.** Given points `p_0..p_m` with chord-length parameters `t_k` and unit end tangents `t̂_L, t̂_R`, find scalars `α_L, α_R ≥ 0` minimizing `Σ_k ‖B(t_k) − p_k‖²` where `b_0 = p_0, b_3 = p_m, b_1 = b_0 + α_L t̂_L, b_2 = b_3 + α_R t̂_R`.
**Formula.** Normal equations of the 2×2 system:

```
A_L(k) = t̂_L B_1³(t_k),  A_R(k) = t̂_R B_2³(t_k)
[ Σ A_L·A_L   Σ A_L·A_R ] [α_L]   [ Σ A_L·(p_k − Q(t_k)) ]
[ Σ A_L·A_R   Σ A_R·A_R ] [α_R] = [ Σ A_R·(p_k − Q(t_k)) ]
Q(t) = b_0 B_0³ + b_0 B_1³ + b_3 B_2³ + b_3 B_3³   (the α=0 baseline curve)
```

**Parameterization.** Chord length: `t_k = Σ_{j≤k} ‖p_j − p_{j−1}‖ / total`. **Reparameterization** (Newton–Raphson on `f(t) = (B(t) − p_k)·B′(t)`): `t ← t − f(t)/f′(t)`, `f′ = ‖B′‖² + (B−p)·B″`, ≤ 4 iterations.
**Variables.** α pt; t dimensionless. **Units.** pt.
**Complexity.** O(m) per fit; total O(P log P) typical with adaptive splitting.
**Numerical stability.** The 2×2 system is near-singular when `t̂_L ≈ ±t̂_R` and points are nearly collinear (det → 0). Guard: if `det < 1e−12·‖A‖²` or any `α ≤ 0` or `α > 3·chord`, fall back to the heuristic `α_L = α_R = chord/3` (Wu/Schneider fallback). Newton steps that leave [0,1] are clamped; non-improving steps are discarded.
**Failure cases.** Oscillating data (post-smoothing prevents this) can produce loops: detected by checking `B′(t)·chord_dir > 0` at samples — violation forces a split.
**Implementation notes.** Max error measured at the `t_k` (not resampled) is what drives splitting, at the point of max error; split tangent = centripetal average of adjacent chords.

### 9.3 Curve continuity classes

**Definition.**

| Class | Requirement at joint `B¹(1)=B²(0)` |
|---|---|
| C0 | positional equality |
| G1 | `B¹′(1) ∥ B²′(0)`, same direction (unit tangents equal) |
| C1 | `B¹′(1) = B²′(0)` (equal vectors incl. magnitude) |
| G2 | G1 + equal signed curvature `κ` |

**Curvature formula.** `κ(t) = (B′ × B″) / ‖B′‖³` (scalar z-component in 2-D).
**Engine contract.** Within a corner-free run: **G1** (mirrored tangent directions at shared endpoints; magnitudes free — C1 is *not* required and would over-constrain the LSQ fit). At corners and junctions: **C0** by design. G2 is not promised (registered-alternative territory).
**Numerical stability.** κ is ill-conditioned where `‖B′‖ → 0`; the no-zero-tangent rule (§9.1) bounds it.
**Failure cases.** Enforcing G1 across a junction shared by 3+ arcs is over-determined — junctions are always C0 (each arc's tangent is free).
**Implementation notes.** G1 at interior splits is enforced structurally (shared unit tangent with opposite signs), not numerically checked.

---

## 10. Boundary smoothing mathematics

**Definition.** Corner-preserving Gaussian vertex filter (ENGINE_SPEC §17).
**Formula.** Turn angle at `v_k`: `θ_k = ∠(v_k − v_{k−1}, v_{k+1} − v_k) = atan2(cross, dot)`. Corner iff `|θ_k| > θ_c` (default 60°). For non-corners:

```
v_k ← Σ_{j=−2}^{2} g_j v_{k+j} / Σ g_j,   g_j = exp(−j²/2σ²), σ = 1
```

with corner/junction terms excluded from the window (weights renormalized), followed by the displacement clamp `‖v_k′ − v_k‖ ≤ δ_max` (0.2 mm in pt): if exceeded, `v_k′ = v_k + δ_max·(v_k′−v_k)/‖v_k′−v_k‖`.
**Variables.** σ in vertex-index units; δ_max pt. **Units.** pt.
**Complexity.** O(P). **Numerical stability.** Convex combination — unconditionally stable; `atan2` of short segments is noisy, so segments shorter than 1e−6 pt inherit the previous direction.
**Failure cases.** Iterating the filter shrinks curvature systematically (Gaussian smoothing is curvature flow in the limit) — exactly one pass is normative.
**Smoothness metric (quality).** Discrete curvature energy `E_κ = Σ_k θ_k²` per arc; the stage must reduce Σ_arcs E_κ by ≥ 40 % on photo fixtures (ENGINE_SPEC §17). E_κ is scale-invariant (angles only), making it comparable across resolutions.

---

## 11. Region merge cost function

**Definition.** The greedy printability merge (ENGINE_SPEC §11) folds sub-floor region `r` into `argmin_n C(r,n)` over RAG neighbors.
**Formula.**

```
C(r, n) = ΔE00( μ(r), μ(n) ) − λ · w_len(r, n) / perim(r),      λ = 15
```

**Variables.** First term ∈ [0, ~100] ΔE00; second term ∈ [0, λ] (the ratio is ≤ 1). λ calibrates "one full-perimeter hug is worth 15 ΔE00 of color mismatch."
**Units.** ΔE00 (the boundary term is rendered dimensionless by the perimeter ratio, scaled by λ).
**Complexity.** O(deg(r)) per pop; total O(R log R + M·d) with heap updates.
**Numerical stability.** Trivial. Ties broken by (larger neighbor area, lower id) — total order guaranteed.
**Failure cases.** λ too large → slivers merge into geometrically-adjacent but perceptually wrong regions (visible color bleeding); λ too small → long thin slivers jump to distant-colored large fields, creating speckle at print scale. λ = 15 was chosen so that for a typical sliver (hug ratio ≈ 0.6) the geometry term (≈ 9) dominates the median inter-palette ΔE00 (≈ 20) only when color difference is below ~9 — i.e., geometry breaks near-ties, color decides clear cases.
**Termination.** Each merge strictly decreases region count; heap re-entries only occur for regions still < A_min whose area *grew* — bounded by R total insertions ⇒ the loop terminates in ≤ R−1 merges.
**Implementation notes.** After merging, `w_len` sums and `perim` update by inclusion–exclusion: `perim(n′) = perim(n) + perim(r) − 2 w_len(r,n)`.

---

## 12. Region split mathematics

### 12.1 Oversize criterion and target cell count

**Formula.** Split region `r` iff `area(r) > A_max = 40·A_min`; target `k = ⌈ area(r) / (A_max/2) ⌉`.
**Rationale.** Products average `A_max/2`, safely above `A_min` with a 20× margin.

### 12.2 Farthest-point sampling (seeding)

**Definition.** Greedy 2-approximation of the k-center problem on the region mask under Euclidean distance.
**Formula.** `s_1 = argmax_p DT(p)` (distance transform argmax; ties → smallest pixel index); `s_{i+1} = argmax_p min_{j≤i} ‖p − s_j‖`.
**Complexity.** O(k·N_r) with incremental min-distance array. **Numerical stability.** Integer squared distances — exact.
**Failure cases.** Extremely thin regions: seeds collapse onto the medial line — acceptable (cells become segments of the shape).
**Implementation notes.** The exact Euclidean distance transform (Felzenszwalb–Huttenlocher, O(N_r)) supplies both `DT` and the §14 clearance machinery.

### 12.3 Watershed on gradient (textured branch)

**Definition.** Marker-based watershed: minimize, over label assignments with the k seeds as markers, the topographic flooding of `g = ‖∇L*‖` (Sobel) restricted to the mask; equivalently, each pixel joins the marker whose minimax path weight `min over paths max g` is smallest.
**Complexity.** O(N_r log N_r) (priority-flood). **Numerical stability.** Flood order ties broken by (g, insertion counter) — deterministic.
**Failure cases.** Flat g (σ_LAB < 2): catchments are noise-driven → use the Voronoi branch instead (normative switch, ENGINE_SPEC §12).

### 12.4 Discrete Voronoi split (flat branch)

**Definition/Formula.** `cell(p) = argmin_i ‖p − s_i‖²`, ties → lowest seed index. Restricted to the mask (geodesic effects ignored deliberately: cells may be disconnected across narrow necks, then reconnected by the §12 fold-back rule).
**Complexity.** O(k·N_r) naïve (fine at region scale) or O(N_r) by jump-flooding if profiled.
**Compactness guarantee (quality).** Isoperimetric quotient `Q = 4πA/P² ≥ 0.5` per cell on convex flat regions (tested property; Voronoi cells of well-separated seeds in convex domains are convex ⇒ Q high).

---

## 13. Largest empty circle, medial axis, and the pole of inaccessibility

### 13.1 Distance function of a face

**Definition.** For face `Ω` (with holes), `d_Ω(p) = inf_{q ∈ ∂Ω} ‖p − q‖` for `p ∈ Ω`, else 0.
**Largest empty circle / inscribed circle.** `r* = max_p d_Ω(p)`; **pole of inaccessibility** `c* = argmax d_Ω`. Inscribed diameter `= 2r*` is the printability metric (I4).
**Units.** pt (vector domain) or px (raster probes).

### 13.2 Medial axis (context)

**Definition.** `MA(Ω) = { p ∈ Ω : |nearest-point set on ∂Ω| ≥ 2 }` — the ridge set of `d_Ω`; `c*` is the global maximum of `d_Ω`, always on `MA(Ω)`.
**Why not computed exactly.** Exact MA of Bézier boundaries requires algebraic bisector curves (degree ≥ 6) — numerically fragile and unnecessary: only the *maximum* of `d_Ω` is needed, not the whole ridge. The engine therefore never constructs MA; this section exists to justify that omission.
**Failure cases (of exact MA, for the record).** MA is unstable under C¹ boundary perturbation (hairs from micro-features) — another reason to avoid it.

### 13.3 Polylabel (quadtree branch-and-bound) — the normative algorithm

**Definition.** Branch-and-bound maximization of `d_Ω` over axis-aligned cells.
**Formula.** For a square cell with center `c`, half-diagonal `h`: upper bound `U = d_Ω(c) + h`, lower bound `d_Ω(c)`. Pop cells from a max-heap by `U`; prune when `U ≤ best + ε_p` (precision `ε_p` = 0.5 pt); split else into 4 children.
**Distance evaluation.** `d_Ω(c)` = min distance to the flattened boundary polylines (0.1 mm flattening tolerance), sign by winding (§7.4).
**Complexity.** O(n) per distance eval over boundary size n; empirically O(n log(diam/ε_p)) total.
**Numerical stability.** Distances are guarded `hypot`; the bound argument tolerates approximate distances as long as they are *under*-estimates of true clearance by ≤ flattening tolerance — hence `ε_p` must exceed the flattening tolerance (0.5 pt ≈ 0.176 mm > 0.1 mm ✓; this inequality is a config cross-rule).
**Failure cases.** Multi-modal faces (dumbbells) have ties; heap order (U, then cell center lex order) makes the returned pole deterministic.
**Implementation notes.** Result `(c*, r*)` feeds both label placement (§14) and the printability validator; computing it once and sharing is mandated (single source of the number).

### 13.4 Voronoi diagrams (context and uses)

**Definition.** For sites `S`, `Vor(s) = { p : ‖p−s‖ ≤ ‖p−s′‖ ∀ s′ }`. Continuous Voronoi of the boundary *segments* is the medial-axis machinery (§13.2, avoided); discrete point-site Voronoi is used in the flat split (§12.4).
**Complexity.** Fortune's sweep O(n log n) — not used; discrete evaluation suffices at region scale.
**Implementation notes.** Any future exact-Voronoi need (e.g., G2 offsetting) must go through `foundation/geometry` with robust predicates; this document intentionally scopes Voronoi to the discrete case.

---

## 14. Label placement mathematics

### 14.1 Font-fit model

**Definition.** A printed number (≤ 2 digits) must fit inside the clearance circle `(c*, r*)` of its face.
**Formula.** With the bundled font's metrics — cap height ratio `κ_f = capheight/em`, digit advance ratio `ω_f = advance/em` — a string of `n` digits at size `S` (pt) occupies a bounding box `(n·ω_f·S) × (κ_f·S)`. Fit condition (box inscribed in circle of radius r*):

```
S ≤ 2 r* / √( (n·ω_f)² + κ_f² )
S* = clip( S_fit , font_min , font_max ),  seeded by the closed form S₀ = 1.35 r* (n=2, DejaVu metrics)
```

**Variables.** DejaVu Sans: `ω_f = 0.636`, `κ_f = 0.729` (pinned; changing fonts changes these constants → this section).
**Units.** pt.
**Complexity.** O(1). **Numerical stability.** Trivial.
**Failure cases.** `S_fit < font_min` triggers the leader path (§14.2); never scale below `font_min` (readability floor is a product invariant, I4).
**Implementation notes.** The 1.35 factor: `2/√((2·0.636)² + 0.729²) = 2/1.466 = 1.364`, rounded down for safety.

### 14.2 Leader-line placement

**Definition.** Number placed outside the face at anchor `q`, connected by segment `q → c*`.
**Formula.** Candidates on the ring `‖q − c*‖ = r* + ρ` (ρ = 4 mm in pt), sampled at 16 fixed angles from 0 in π/8 steps; feasible iff (i) the number's bbox at `q` (size `font_min`) has clearance > its own half-diagonal from all geometry (spatial-hash query), and (ii) segment `q c*` crosses < 3 arcs (§7.3 count). Choose the feasible candidate minimizing crossings, then angle index (determinism).
**Complexity.** O(16·(query + crossing count)). **Failure cases.** No feasible candidate → FATAL finding (validator decides).
**Implementation notes.** The crossing budget (< 3) bounds visual ambiguity; it is a normative constant, not a knob.

### 14.3 Overlap resolution

**Definition.** Pairwise label bbox disjointness via greedy-by-clearance with bounded displacement.
**Formula.** Process labels by descending r*; a conflicting label may translate along `∇d_Ω` (ascent direction of its face's distance function, estimated by central differences on the flattened boundary distance) by at most `r*/2`; if still conflicting → demote to leader.
**Complexity.** O(F log F + conflicts). **Numerical stability.** Gradient estimation needs no precision (any uphill direction works; step is clamped).
**Failure cases.** Dense sliver clusters can cascade into many leaders — bounded by the A_min floor which limits label density to ≤ content_area/A_min.

---

## 15. Graph optimization

### 15.1 Greedy heap-based merge (analysis)

The §11 merge is a greedy matroid-free heuristic; no optimality claim is made. Its guarantees are: (i) termination in ≤ R−1 merges (§11), (ii) invariant maintenance (all products of merges remain valid RAG nodes), (iii) determinism. Approximation quality is regulated empirically by the fidelity metric (mean ΔE00 of merged pixels ≤ 15, ENGINE_SPEC §11) rather than by a ratio bound — a deliberate engineering trade documented here to preempt "why not optimal?" reviews: the exact objective (perceptual fidelity of the final *page*) is not expressible as a graph functional.

### 15.2 Palette permutation objective

**Definition.** Anti-mystery leakage ordering (ENGINE_SPEC §20).
**Formula.** Greedy sequence maximizing `min(ΔE00(π_i, π_{i−1}), ΔE00(π_i, π_{i−2}))`; acceptance metric Spearman rank correlation

```
ρ_s = 1 − 6 Σ d_i² / (K(K²−1)),   d_i = rank_number(i) − rank_L*(i)
```

with acceptance `|ρ_s| ≤ 0.4`.
**Complexity.** O(K²). **Numerical stability.** Exact rank arithmetic (integers).
**Failure cases.** K = 2: ρ_s ∈ {±1} necessarily; the acceptance rule is waived for K < 4 (normative exception).
**Implementation notes.** The derangement fallback (interleave halves) is a fixed permutation — applying it twice returns the original; the min-|ρ_s| choice is therefore well-defined.

---

## 16. Quality metrics

### 16.1 SSIM (fidelity probe, I1)

**Definition.** Structural similarity on luminance between solved preview `x` and quantized working raster `y`, both resampled to the common grid.
**Formula.** Per 8×8 window (uniform, stride 1), with `C₁ = (0.01·255)², C₂ = (0.03·255)²` on 8-bit L:

```
SSIM(w) = (2μ_x μ_y + C₁)(2σ_xy + C₂) / ( (μ_x²+μ_y²+C₁)(σ_x²+σ_y²+C₂) )
SSIM = mean over windows;  gate: ≥ 0.985
```

**Complexity.** O(pixels) via integral images. **Numerical stability.** C₁, C₂ regularize flat windows; use float64 accumulators (float32 integral images overflow at 4k×4k).
**Failure cases.** Resampling filter must be identical for both images (area-average) or the metric measures the filter, not fidelity.
**Implementation notes.** Luminance = L* of §3.2, rescaled to [0,255]; window statistics via the standard integral-image identities.

### 16.2 Face–label agreement (fidelity audit, I1)

**Formula.** For face f with rasterization `Pix(f)`: `agree(f) = |{p ∈ Pix(f) : ℓ(p) = label(f)}| / |Pix(f)| ≥ 0.99`, and the majority label must equal `label(f)`.
**Complexity.** O(N). **Failure cases.** Faces thinner than one working pixel rasterize empty: treated as pass with a WARNING (they exist legally after simplification of 1-px necks).

### 16.3 Compactness (isoperimetric quotient)

**Formula.** `Q(f) = 4π A(f) / P(f)²  ∈ (0, 1]` (1 = disk). Tracked per-fixture mean in the quality benchmarks (no hard gate; regression-monitored).
**Numerical stability.** Perimeter of a Bézier face by flattened polyline length (0.1 mm tolerance) — consistent across runs by fixed flattening depth (§9.1 bound).

### 16.4 Boundary smoothness (curvature energy)

As §10: `E_κ = Σ θ_k²` on flattened geometry, reported per benchmark fixture; scale-invariant, so comparable across page sizes.

---

## 17. Printability metrics (I4)

### 17.1 Inscribed diameter

**Formula.** `D(f) = 2 r*(f)` from §13.3. Gate: `D(f) ≥ d_min` (mm→pt) for in-region labeling; else leader (§14.2); else FATAL.
**Units.** mm (config), pt (computation).
**Numerical stability.** r* is exact to `ε_p + flattening tolerance` — both fixed; the gate comparison includes no additional epsilon (the precision budget is already inside r*).
**Failure cases.** Annular faces: r* correctly measures the ring thickness (distance to *both* boundary components) — the metric is hole-aware by construction of `d_Ω`.

### 17.2 Colorability model (why d_min = 3.5 mm)

**Definition.** The floor models the physical tool: a sharpened pencil tip contact patch is ~0.5–1 mm; controlled fill without bleeding over a 0.3 pt outline requires ≥ ~3× tool diameter of maneuvering room; 3.5 mm also exceeds the 2-digit label at `font_min` = 6 pt (cap height 6·0.729 pt ≈ 1.54 mm) with the §14.1 fit rule at r* = 1.75 mm → S_fit ≈ 4.8 pt < 6 pt — meaning *labels*, not pencils, are usually the binding constraint near the floor; the leader mechanism resolves exactly that band. Presets: easy 5.0 mm, medium 3.5 mm, hard 2.5 mm (hard accepts more leaders).
**Failure cases.** d_min below ~2 mm makes the leader mechanism dominate (>25 % leaders) — flagged by the quality benchmark, not forbidden.

### 17.3 Number readability

**Formula.** Gate: every printed number `S ≥ font_min = 6 pt` (≈ 2.1 mm cap height — the accepted floor for children's print material). In-region numbers must additionally satisfy §14.1's fit inequality.

---

## 18. Numerical policy summary

| Concern | Policy |
|---|---|
| Topology predicates | doubled-integer exact in `𝖢`; filtered `orient` (§7.2) for float geometry |
| Areas (I3 gate) | exact integer pre-scaling; 1e−4 relative tolerance post-scaling |
| Color math | float64 end-to-end; pinned matrix constants (§3.2); Sharma test-set conformance for ΔE00 |
| Quality thresholds | float64 comparisons, no hidden epsilons — precision budgets live inside the computed quantities |
| Reductions | fixed iteration order everywhere (determinism I2) |
| Randomness | none in geometry; seeded PRNG only in k-means init (ENGINE_SPEC §1.3) |

## 19. References

1. G. Sharma, W. Wu, E. Dalal, *The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations*, Color Res. Appl. 30(1), 2005.
2. P. J. Schneider, *An Algorithm for Automatically Fitting Digitized Curves*, Graphics Gems I, 1990.
3. M. Visvalingam, J. D. Whyatt, *Line Generalisation by Repeated Elimination of Points*, Cartographic J. 30(1), 1993.
4. V. Agafonkin, *Polylabel: A Fast Algorithm for Finding the Pole of Inaccessibility*, Mapbox engineering, 2016.
5. P. F. Felzenszwalb, D. P. Huttenlocher, *Distance Transforms of Sampled Functions*, Theory of Computing 8, 2012.
6. J. R. Shewchuk, *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates*, Discrete Comput. Geom. 18, 1997.
7. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 13(4), 2004.
8. D. Hasler, S. Süsstrunk, *Measuring Colourfulness in Natural Images*, SPIE 5007, 2003.
9. F. Meyer, *Topographic Distance and Watershed Lines*, Signal Processing 38, 1994.

## 20. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-06 | Initial complete mathematical specification. |

