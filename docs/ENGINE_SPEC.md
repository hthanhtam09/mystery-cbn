# Mystery Color-by-Number Engine — Engine Specification

**Status:** v1.0 — authoritative algorithm-level specification. Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (system architecture). Where the two disagree, ARCHITECTURE.md governs structure; this document governs algorithms.
**Rule:** every implementation decision below is normative. Changing a default algorithm, formula, threshold, or parameter name requires a revision of this document *first*, plus an ADR in `docs/adr/`.
**Horizon:** 10-year maintenance life.

---

## 1. Scope, conventions, and notation

### 1.1 Scope

This specification defines all 22 processing modules of the deterministic image → vector converter: their purposes, inputs, outputs, default algorithms, alternatives with rationale, complexity, failure modes, edge cases, configuration, quality requirements, benchmarks, and unit tests. It does not define code structure (see ARCHITECTURE.md §2–3) or HTTP/CLI surfaces (ARCHITECTURE.md §5).

### 1.2 Notation

| Symbol | Meaning |
|---|---|
| `H, W` | working-raster height/width in pixels |
| `N = H·W` | pixel count of the working raster |
| `K` | palette size (number of colors) |
| `R` | number of regions (connected components) |
| `A` | number of arcs; `V` junctions; `F` faces |
| `ΔE00(a,b)` | CIEDE2000 color difference between LAB colors a, b |
| `ΔE76(a,b)` | CIE76 Euclidean LAB distance |
| `d_min` | minimum printable inscribed diameter (mm) |
| `s` | `work_scale`: points per working pixel |

### 1.3 Global mathematical conventions

- **Color.** All perceptual math in CIELAB, D65, 2° observer, computed from linearized sRGB (IEC 61966-2-1 transfer function; linearization threshold 0.04045, exponent 2.4). LAB is authoritative for palettes; sRGB is derived. ΔE00 is the normative difference metric everywhere a threshold is stated; ΔE76 may be used **only** inside inner loops explicitly permitted below, never for pass/fail gates.
- **Units.** Raster and graph domains: working pixels. The Arc Graph module (§15) applies `s = content_width_pt / W_working` exactly once. All modules after §15 operate in points. Millimetre thresholds in this document convert via `pt = mm / 25.4 × 72`. Only `foundation/units` performs conversions.
- **Determinism (invariant I2).** Every stochastic step uses a PRNG seeded from the run seed (default `seed = 0`) plus a stage-name hash: `stage_seed = SHA-256(seed ‖ stage_name)[:8]` interpreted as uint64. No wall-clock, no hash randomization, no unordered-set iteration in any output-affecting path. Floating-point reductions over unordered collections must fix iteration order (sorted by stable id).
- **Connectivity.** Pixel regions use **4-connectivity**; background/crack topology therefore uses 8-connectivity implicitly. This choice is load-bearing for watertightness (§12–15) and may never be configured.
- **Coordinate frames.** Raster: pixel centers at integer coordinates, origin top-left, y down. Crack grid: pixel *corners* at half-integer offsets, i.e. corner `(i, j)` sits at raster coordinate `(i − 0.5, j − 0.5)`. Vector/page: points, origin top-left of the trim box, y down (SVG convention); the PDF exporter flips y once.

### 1.4 Pipeline order and artifact flow

```
 §4 Raster Load ──▶ §5 Preprocess ──▶ §6 Color Analysis (advisory)
                          │
                          ▼
                    §7 Quantization ──▶ §8 Noise Removal
                          │
                          ▼
        §9 Connected Components ──▶ §10 Region Graph
                          │
                          ▼
        §11 Tiny Region Merge ──▶ §12 Large Region Split (optional)
                          │
                          ▼  (raster → vector boundary)
        §13 Contour Extraction ─▶ §14 Topology Graph ─▶ §15 Arc Graph
                          │
                          ▼
        §16 Polygon Simplification ─▶ Geometry Normalize (Sprint 36A.5) ─▶ §17 Curve Smoothing ─▶ §18 Bézier Fitting
                          │
                          ▼
        §19 Label Placement · §20 Palette Optimization · §21 Legend Generation
                          │
                          ▼
        §22 SVG Export ─▶ §23 PDF Export · §24 PNG Preview
                          │
                          ▼
                    §25 Validation (I1–I4 gates)
```

Each module below follows one fixed template: **Purpose · Responsibilities · Input · Output · Dependencies · Algorithm · Algorithm Alternatives · Reason for Choosing Default · Complexity · Memory Complexity · Failure Modes · Edge Cases · Configuration Parameters · Quality Requirements · Benchmarks · Unit Tests · Future Improvements.** Benchmarks are budgets on the reference fixture: 2 MP photo, single core, on the CI reference machine recorded in `benchmarks/baselines/`.

---

## 2. Global quality gates (summary)

| Gate | Metric | Threshold | Enforced by |
|---|---|---|---|
| I1 Fidelity | face↔label-map majority agreement | ≥ 99.0 % of face pixels | §25.1 |
| I1 Fidelity | SSIM(solved preview, quantized raster) | ≥ 0.985 | §25.1 |
| I2 Determinism | SVG byte hash across two runs | identical | CI |
| I3 Topology | Σ face areas vs content area | within ±0.01 % | §25.2 |
| I3 Topology | arcs bordering ≠ 2 faces; self-intersections | 0 | §25.2 |
| I4 Printability | regions with inscribed dia < `d_min` and no leader | 0 after repair | §25.3 |
| Palette | min pairwise ΔE00 | ≥ `palette_min_delta_e` | §25.4 |
| Speed | 2 MP photo → full bundle | ≤ 15 s single core | §9 of ARCHITECTURE.md |

## 3. Global configuration keys

Cross-module keys (every module also has its own section, listed per module):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `seed` | int ≥ 0 | 0 | run seed for all stochastic steps |
| `page.width_mm / height_mm` | float | 215.9 / 279.4 | trim size (US Letter) |
| `page.margin_mm` | float | 12.7 | uniform margin |
| `page.dpi` | int | 300 | raster export resolution |
| `quality.d_min_mm` | float | 3.5 | printability floor: min inscribed diameter |
| `quality.font_min_pt` | float | 6.0 | smallest in-region number |
| `quality.font_max_pt` | float | 14.0 | largest number |

---

## 4. Module: Raster Load

**Purpose.** Decode any supported source into the engine's canonical raster: H×W×3 float32 sRGB in [0, 1], correctly oriented and color-managed.

**Responsibilities.** Format detection; decode; EXIF orientation; ICC → sRGB conversion; alpha flattening; bit-depth normalization; provenance (source SHA-256, applied transforms).

**Input.** File path or bytes. Supported containers: PNG, JPEG, WebP, TIFF (first frame), BMP.
**Output.** `RasterImage` artifact: float32 array, `work_scale = 0` (unset), provenance.
**Dependencies.** Pillow (implementation detail). `foundation/errors`.

**Algorithm (default).**
1. Decode with Pillow; reject if decoder fails or format unsupported → `InputError`.
2. Apply EXIF orientation tag (transpose-based, lossless).
3. If an embedded ICC profile exists, convert to sRGB with relative colorimetric intent via littleCMS; if absent, assume sRGB.
4. If alpha present: composite over opaque white (`out = α·rgb + (1−α)·1`).
5. Convert palette/gray/16-bit modes to 8- or 16-bit RGB, then to float32 `v/255` (or `v/65535`).
6. Compute SHA-256 of the raw source bytes for provenance.

**Algorithm alternatives.** pyvips (faster streaming decode for > 50 MP); imageio; OpenCV `imread` (no ICC, no EXIF — disqualified).
**Reason for default.** Pillow is the only pure-Python-friendly decoder with mature EXIF + ICC handling; input decode is < 2 % of pipeline time, so vips's speed does not pay for its native-dependency cost.

**Complexity.** O(N) time. **Memory.** O(N) — peak 2 copies (decoded + float), ≤ 24 bytes/pixel transient.

**Failure modes.** Corrupt file → `InputError`; unsupported mode (CMYK without profile) → `InputError`; zero-area image → `InputError`.
**Edge cases.** Animated GIF/TIFF: first frame only, warning recorded. Grayscale: replicated to 3 channels. Images with orientation *and* profile: orientation first, then color. 1×1 to 63-px inputs: rejected (min side 64 px).

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `load.max_pixels` | int | 100_000_000 | 1e6 – 4e8; larger → `InputError` (decompression-bomb guard) |
| `load.assume_srgb` | bool | true | if false, missing profile → `InputError` |

**Quality requirements.** Round-trip: loading an 8-bit sRGB PNG must reproduce `v/255` exactly. ICC conversion error ≤ 1 ΔE00 vs reference lcms output.
**Benchmarks.** 2 MP JPEG decode + normalize ≤ 0.15 s; 24 MP ≤ 1.5 s.
**Unit tests.** EXIF orientations 1–8 produce identical pixel content; alpha-over-white formula on synthetic RGBA; 16-bit PNG scaling; bomb guard triggers at limit; corrupt file raises `InputError`.
**Future improvements.** pyvips backend as a registered alternative for > 50 MP; HEIC/AVIF support; multi-page selection.

---

## 5. Module: Preprocessing

**Purpose.** Produce the working raster: bounded resolution, edge-preserving flattened colors, so quantization sees clean color fields.

**Responsibilities.** Downscale to working resolution; record `work_scale` basis; edge-preserving smoothing; optional local contrast enhancement. Never introduces colors outside the convex hull of local input colors.

**Input.** `RasterImage`. **Output.** `RasterImage'` (working resolution) + `resize_factor` in provenance.
**Dependencies.** OpenCV (detail); `foundation/color` for CLAHE's L-channel work.

**Algorithm (default).**
1. **Resize:** if `max(H, W) > max_working_px`, scale by `f = max_working_px / max(H, W)` using area-averaging interpolation (OpenCV `INTER_AREA`). Never upscale.
2. **Smooth:** `smooth_passes` iterations of the bilateral filter, `σ_color = 0.08` (in [0,1] RGB units), `σ_space = 5 px`, kernel radius `⌈2σ_space⌉`.
3. **Optional CLAHE:** if enabled, convert to LAB, apply CLAHE to L only (clip 2.0, 8×8 tiles), convert back.

**Algorithm alternatives.** Guided filter (faster, slightly weaker flattening at strong edges); mean-shift filtering (best flattening, 10–30× slower, nondeterministic convergence ordering); edge-aware pyramid / domain transform; ML denoisers (nondeterministic across hardware — disqualified for the default path).
**Reason for default.** Iterated bilateral gives the strongest color flattening per millisecond among deterministic options, creates no new colors beyond local mixtures, and its two parameters map directly to user-meaningful knobs ("how much detail survives").

**Complexity.** Resize O(N); bilateral O(N·r²) with r = kernel radius (≈ O(N·100) at defaults). **Memory.** O(N), 2 buffers.

**Failure modes.** None expected; arithmetic is total. A degenerate all-constant image passes through unchanged.
**Edge cases.** Image smaller than `max_working_px`: no resize, `f = 1`. Extreme aspect ratios (> 8:1): allowed; page-fit letterboxing is a layout concern. `smooth_passes = 0`: pass-through.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `preprocess.max_working_px` | int | 1600 | 256–6000 |
| `preprocess.smooth_passes` | int | 2 | 0–5 |
| `preprocess.bilateral_sigma_color` | float | 0.08 | (0, 1] |
| `preprocess.bilateral_sigma_space` | float | 5.0 | (0, 50] |
| `preprocess.clahe` | bool | false | — |
| `preprocess.clahe_clip` | float | 2.0 | (0, 40] |

**Quality requirements.** Edge preservation: SSIM(smoothed, input) ≥ 0.85 on the fixture ladder; gradient-magnitude correlation at strong edges (top decile) ≥ 0.9.
**Benchmarks.** 1600-px working raster, 2 passes: ≤ 0.8 s.
**Unit tests.** Resize factor math and rounding; no-upscale rule; checkerboard survives smoothing (edges intact); flat field is a fixpoint; CLAHE touches only L.
**Future improvements.** Guided-filter implementation as registered alternative; saliency-adaptive σ.

---

## 6. Module: Color Analysis

**Purpose.** Measure global image statistics and translate them into *advisory* config overrides for the auto-tune layer (may only fill values the user left unset — ARCHITECTURE.md §7).

**Responsibilities.** Compute statistics; emit `ImageStats`; emit a config fragment proposing `quantize.n_colors` and `preprocess.smooth_passes`.

**Input.** `RasterImage'`. **Output.** `ImageStats` + auto-tune config fragment.
**Dependencies.** `foundation/color`.

**Algorithm (default).**
1. **Colorfulness** (Hasler–Süsstrunk): with `rg = R−G`, `yb = ½(R+G)−B` on [0,255] scale: `M = √(σ_rg² + σ_yb²) + 0.3·√(μ_rg² + μ_yb²)`.
2. **Edge density** `ρ`: fraction of pixels whose Sobel gradient magnitude (on L) exceeds 0.1 (L in [0,100] normalized to [0,1]).
3. **Luminance histogram:** 64 uniform bins over L; entropy `H_L = −Σ p·log₂ p`.
4. **Palette-size proposal:** `k* = clip(round(6 + 0.12·M + 6·ρ + 0.8·H_L), 8, 30)`.
5. **Smoothing proposal:** `ρ > 0.25 → smooth_passes = 3`; `ρ < 0.05 → 1`; else default.

**Algorithm alternatives.** Unique-color counting after uniform quantization (unstable across photographs); gap statistic / silhouette sweep over k (10–50× cost of quantization itself); learned advisor (future plugin, §14.1 of ARCHITECTURE.md).
**Reason for default.** Closed-form statistics are O(N), fully deterministic, explainable to users ("busy image → more colors"), and empirically monotone with perceived complexity; a k-sweep would dominate total runtime for marginal gain.

**Complexity.** O(N). **Memory.** O(1) beyond input.
**Failure modes.** None; totals are defined for any raster.
**Edge cases.** Grayscale image: M ≈ 0 → k driven by entropy/edges (floor 8 applies). Flat color: `k* = 8` floor still applies; downstream merge collapses unused colors.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `analyze.enabled` | bool | true | — |
| `analyze.k_min / k_max` | int | 8 / 30 | 2–64, min ≤ max |

**Quality requirements.** Proposal stability: `k*` identical across runs (determinism); `k*` within ±2 under 90°rotation or mirroring of the input.
**Benchmarks.** ≤ 0.1 s at 1600 px.
**Unit tests.** M formula on analytic two-color images; ρ on a step edge; entropy of uniform vs delta histograms; clipping bounds; rotation invariance of proposals.
**Future improvements.** Replace formula constants with a fitted model over the quality-benchmark corpus; AI advisor plugin.

---

## 7. Module: Quantization

**Purpose.** Reduce the working raster to K perceptually well-separated palette colors and a per-pixel label map — the single most quality-determining stage.

**Responsibilities.** Palette extraction; per-pixel assignment; near-duplicate palette merge; coverage-ordered numbering; palette in LAB (authoritative) + derived sRGB.

**Input.** `RasterImage'`; `quantize` config. **Output.** `LabelMap` (H×W int32) + `Palette`.
**Dependencies.** `foundation/color`; OpenCV k-means (detail).

**Algorithm (default): seeded LAB k-means.**
1. Convert working raster to LAB.
2. **Sample:** if N > 100 000, take a deterministic uniform stride sample of exactly 100 000 pixels (stride order, no RNG).
3. **Init:** k-means++ with `stage_seed` (§1.3), `n_init = 4`, keep the run with lowest inertia (ties → lowest run index).
4. **Iterate:** Lloyd's algorithm on the sample, ΔE76 metric (permitted inner loop), until center movement < 0.05 or 50 iterations.
5. **Assign all pixels** to the nearest center (ΔE76).
6. **Merge close centers:** while any pair has ΔE00 < `merge_delta_e`, merge the pair with the smallest ΔE00 into their pixel-count-weighted LAB mean and re-assign affected pixels.
7. **Renumber** labels by descending pixel coverage; recompute exact LAB means of final classes; derive sRGB.

**Algorithm alternatives.** Median cut (fast, poor in smooth gradients); octree (fast, favors dominant hues, weak K control); Wu's variance minimization (good, but harder to seed-control and no perceptual space); neural palette extraction (nondeterministic, heavy).
**Reason for default.** K-means in LAB directly minimizes perceptual within-class variance, gives exact K control (required by difficulty presets), and is trivially determinized by seeding. Its known weakness — sensitivity to init — is neutralized by fixed-seed k-means++ with 4 restarts.

**Complexity.** O(S·K·I) for the sample fit (S = 1e5), O(N·K) for assignment. **Memory.** O(N) labels + O(K).
**Failure modes.** K > number of distinct colors → duplicate centers; step 6 collapses them (never an error). Empty cluster during Lloyd's: re-seed that center at the sample point farthest from its center (deterministic).
**Edge cases.** `n_colors = 2`: valid (silhouette pages). Pure grayscale: centers align on the L axis; merge threshold still applies. Alpha-flattened white background typically claims label 0 (highest coverage) — required by legend conventions (§21).

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `quantize.n_colors` | int | 16 (auto-tunable) | 2–64 |
| `quantize.merge_delta_e` | float | 7.0 | 0–30 |
| `quantize.sample_px` | int | 100_000 | 10⁴–10⁶ |
| `quantize.max_iter` | int | 50 | 10–200 |

**Quality requirements.** Mean per-pixel ΔE00(raster, assigned palette color) ≤ 11 on the photo fixtures; min pairwise palette ΔE00 ≥ `merge_delta_e` by construction; byte-identical `LabelMap` across runs.
**Benchmarks.** 1600 px, K = 16: ≤ 2.0 s.
**Unit tests.** Synthetic 4-color image recovers exactly 4 centers with ≤ 0.5 ΔE00 error; determinism (two runs, identical labels); merge collapses near-duplicates; coverage ordering; empty-cluster re-seed path.
**Future improvements.** Weighted k-means using the §14.2 edge-prior; octree implementation as registered alternative for `--fast` preset.

---

## 8. Module: Noise Removal

**Purpose.** Clean the label map of quantization speckle (isolated pixels, thin filaments) before component analysis, without creating labels or moving real boundaries.

**Responsibilities.** Majority smoothing; micro-component absorption. Operates purely in label space.

**Input.** `LabelMap`, `Palette`. **Output.** `LabelMap` (replaced artifact).
**Dependencies.** scikit-image (detail); `foundation/color` for tie-breaking.

**Algorithm (default).**
1. **Modal filter:** iterate a 3×3 majority filter (pixel takes the most frequent label in its 8-neighborhood incl. itself; ties broken by smallest ΔE00 to the pixel's current palette color, then lowest label id) until fixpoint or `max_modal_iters`.
2. **Area opening:** find 4-connected components with area < `speck_px = max(4, ⌊A_min/16⌋)` (A_min from §11); relabel each such component to the label with the longest shared boundary with it (ties → smallest ΔE00, then lowest id). Process components in ascending (area, min-pixel-index) order.

**Algorithm alternatives.** Per-label morphological open/close (creates label conflicts where operations overlap); median filtering on the raster before quantization only (leaves post-assignment speckle); MRF/graph-cut label smoothing (highest quality, but global solvers are 20×+ slower and harder to determinize).
**Reason for default.** Modal + area-opening is the standard label-domain-safe pair: it cannot invent labels, converges quickly, and both steps have exact deterministic tie rules. MRF's marginal quality gain does not justify its cost at working resolution.

**Complexity.** O(N) per modal iteration; components pass O(N α(N)). **Memory.** O(N).
**Failure modes.** None; the identity map is always a legal output.
**Edge cases.** Checkerboard at pixel scale: modal filter needs the tie rule — deterministic result guaranteed. Component equal to `speck_px`: kept (strictly-less-than rule). An entire label class may vanish; the palette keeps the entry until §11 compacts numbering.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `denoise.max_modal_iters` | int | 3 | 0–10 |
| `denoise.speck_divisor` | int | 16 | 4–64 (divides A_min) |

**Quality requirements.** Post-stage isolated single pixels: 0. Boundary displacement vs pre-stage: ≤ 1 px Hausdorff on the illustration fixtures.
**Benchmarks.** ≤ 0.5 s at 1600 px.
**Unit tests.** Single-pixel speck absorbed into correct neighbor; tie-break determinism on synthetic checkerboard; fixpoint termination; area threshold boundary (== kept, −1 absorbed).
**Future improvements.** Optional seam-aware filtering guided by the §14.2 edge prior.

---

## 9. Module: Connected Components

**Purpose.** Convert the label map into discrete regions: maximal 4-connected sets of equal-label pixels, each with identity and summary statistics.

**Responsibilities.** Component labeling; `RegionRecord` computation (id, palette label, area, bbox, seed pixel). Assigns the region ids used by every later stage.

**Input.** `LabelMap`. **Output.** component map (H×W int32 region ids) + `RegionRecord` list (carried inside `RegionGraph` after §10).
**Dependencies.** scikit-image `label` (detail).

**Algorithm (default).** Two-pass union-find connected-component labeling with **4-connectivity** (§1.3, non-configurable): first pass assigns provisional ids and records equivalences; second pass resolves to final ids in raster-scan order of first occurrence (guaranteeing deterministic, top-left-first numbering). Per region record: `area_px`, tight `bbox`, `seed_px` = first pixel in raster order.

**Algorithm alternatives.** One-pass flood fill (poor cache behavior, recursion depth hazards); run-length based labeling (fastest, more complex; the library call already meets budget); 8-connectivity (disqualified — breaks planarity of the crack partition: two diagonal regions would "touch" without a shared crack edge).
**Reason for default.** Two-pass union-find is O(N α(N)), cache-friendly, available as a vetted library call, and its raster-scan id order gives determinism for free.

**Complexity.** O(N α(N)). **Memory.** O(N) for the id map.
**Failure modes.** None for valid label maps.
**Edge cases.** Single region covering the page (flat input): R = 1 is legal end-to-end. Region count can reach tens of thousands pre-merge on noisy input — ids are int32 by contract.

**Configuration parameters.** None (connectivity is an invariant, not a knob).
**Quality requirements.** Exactness: output must equal the mathematical 4-connected partition (property-tested against a reference flood fill).
**Benchmarks.** ≤ 0.2 s at 1600 px, R ≤ 50 000.
**Unit tests.** Diagonal pixels are two regions; donut topology (hole region distinct); id order stability under identical input; agreement with brute-force flood fill on random small rasters (Hypothesis).
**Future improvements.** Run-length labeling if profiling ever shows this stage on the critical path.

---

## 10. Module: Region Graph

**Purpose.** Build the region adjacency graph — the data structure on which merge/split decisions and fidelity audits operate.

**Responsibilities.** Nodes = regions; edges = 4-adjacency with attributes: shared boundary length (crack-edge count) and ΔE00 between the two regions' palette colors. Also records each region's outer perimeter length.

**Input.** component map + `RegionRecord`s + `Palette`. **Output.** `RegionGraph`.
**Dependencies.** `foundation/color`; NetworkX or equivalent adjacency store (detail).

**Algorithm (default).** Single raster sweep: for every horizontal and vertical pixel pair `(p, q)` with different region ids, increment `boundary_len[(min,max)]`; pairs with the page border increment the region's border length. After the sweep, attach `ΔE00(palette[label(a)], palette[label(b)])` to each edge. Perimeter of region r = Σ boundary lengths of its incident edges + its page-border length.

**Algorithm alternatives.** Deriving adjacency from the §13 contour trace (couples graph domain to vector domain, wrong dependency direction); Shapely polygon intersection tests (O(R²) worst case, needless geometry).
**Reason for default.** The pixel-pair sweep is exact, O(N), allocation-light, and shares its definition of "boundary" with crack tracing — the two stages count the same crack edges, which §25.2 exploits as a cross-check.

**Complexity.** O(N + E) with E = adjacency edges (E ≤ 4N). **Memory.** O(R + E).
**Failure modes.** None for a valid component map.
**Edge cases.** R = 1: graph with one node, zero edges — legal. Two regions of the *same* palette label may be adjacent (after §8 tie-breaks); edge ΔE00 = 0 and §11 will merge them first.

**Configuration parameters.** None.
**Quality requirements.** Σ over edges of boundary_len + Σ border lengths = total crack-edge count of the label map (exact integer identity, asserted in CI).
**Benchmarks.** ≤ 0.3 s at 1600 px.
**Unit tests.** 2×2 four-region raster produces the 4-cycle with unit boundary lengths; donut adjacency; boundary-length identity property test; ΔE00 edge attribute against reference values.
**Future improvements.** Incremental edge updates exposed for §11/§12 (currently rebuilt by those stages' own bookkeeping).

---

## 11. Module: Tiny Region Merge

**Purpose.** Enforce the printability floor (invariant I4's area component): no region smaller than a physically colorable size survives.

**Responsibilities.** Compute the pixel-space area floor from `d_min_mm`; iteratively fold sub-floor regions into their best neighbor; keep `RegionGraph`, component map, and coverage counts consistent; compact palette numbering if a color loses all regions.

**Input.** `RegionGraph` + component map + `Palette`; page geometry (for the mm→px conversion via the *projected* work scale). **Output.** updated `RegionGraph` + component map + `Palette`.
**Dependencies.** `foundation/color`, `foundation/units`.

**Algorithm (default): smallest-first greedy merge with perceptual cost.**
1. Area floor: `A_min = π·(d_min_mm/2)² · ppmm²` where `ppmm` = working px per mm of printed content.
2. Min-heap of regions with `area < A_min`, keyed by (area, region id).
3. Pop the smallest region r; merge it into neighbor `n* = argmin_n C(r, n)` with
   `C(r, n) = ΔE00(r, n) − λ · boundary_len(r, n) / perimeter(r)`, `λ = 15`.
   Ties: larger neighbor, then lower region id.
4. Update: relabel r's pixels to n*; n* inherits r's edges (boundary lengths summed); recompute n*'s area; if n* still < A_min it re-enters the heap; if a merged neighbor was in the heap, its key updates.
5. Repeat until the heap is empty. Then re-number palette entries by final coverage, dropping colors with zero regions (renumber map recorded for provenance).

**Algorithm alternatives.** Global graph-cut / MRF region merging (optimal-ish, nondeterministic solver order, 20×+ cost); merging strictly by smallest ΔE00 (ignores geometry — produces long slivers absorbing into distant-colored large fields); watershed-style flooding from large regions (inverts control, harder to bound).
**Reason for default.** Smallest-first greedy is the standard geometric-fidelity-preserving choice: processing tiny regions first means each decision is local and near-optimal, the λ-weighted boundary term preferentially heals slivers into the neighbor they hug, and the loop provably terminates (region count strictly decreases).

**Complexity.** O(R log R + M·d) where M = merges, d = max degree. **Memory.** O(R + E).
**Failure modes.** If the whole page merges to one region (floor larger than every region): legal output; validator warns "degenerate page". If `A_min` exceeds the content area → `ConfigError` (caught at config validation, cross-field rule).
**Edge cases.** Region below floor with a single neighbor: merged regardless of cost. Enclosed region (donut hole) below floor: merges into its enclosing ring. Two sub-floor regions adjacent to each other: smallest-first order handles chains correctly. Inscribed-diameter (not area) failures are *not* this stage's job — leader lines (§19) and the validator (§25.3) own that.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `merge.lambda_boundary` | float | 15.0 | 0–50 |
| (floor derives from `quality.d_min_mm`) | | | |

**Quality requirements.** Post-stage: 0 regions with `area < A_min`. Mean ΔE00 of merged pixels vs their new color ≤ 15 on fixtures (fidelity guard).
**Benchmarks.** 20 000 → ~800 regions at 1600 px: ≤ 1.0 s.
**Unit tests.** Cost formula on a hand-built 3-region graph; chain-merge determinism; heap-key update on neighbor growth; palette compaction and renumber map; termination property test on random graphs.
**Future improvements.** Saliency-weighted per-region floors (§14.3 plugin); learned cost function.

---

## 12. Module: Large Region Split (optional plugin)

**Purpose.** Improve puzzle rhythm: break oversized monotone regions into colorable cells so the page has no boring continents. Ships as a first-party plugin (proof of the plugin API), disabled in the `easy` preset.

**Responsibilities.** Identify oversized regions; split along content seams when content exists, along compact geometric seams when flat; keep all products above `A_min`; update graph/map consistently.

**Input.** `RegionGraph` + component map + `RasterImage'` (still raster domain — legal). **Output.** updated `RegionGraph` + component map.
**Dependencies.** `foundation/geometry` (farthest-point sampling), OpenCV watershed (detail).

**Algorithm (default).**
1. Threshold: `A_max = split_factor · A_min` (default 40×). For each region with `area > A_max`, target cell count `k = ⌈area / (A_max/2)⌉`.
2. Compute region-restricted LAB std `σ`. If `σ ≥ 2.0` (textured): marker-based watershed on the gradient magnitude of L within the region mask, markers = k seeds by farthest-point sampling on the mask's distance transform (deterministic: first seed at distance-transform argmax, ties by pixel index).
3. Else (flat): assign pixels to nearest seed in (x, y) — a discrete Voronoi split yielding compact cells.
4. Any product cell < `A_min` is folded back into its lowest-cost neighbor cell using the §11 cost function.
5. New regions inherit the parent's palette label; graph and map updated as in §11 step 4.

**Algorithm alternatives.** SLIC superpixels within the mask (good compactness, but iteration count/convergence less deterministic across library versions); straight-line guillotine cuts (ugly, content-blind); seam carving (1-D idea, poor fit for 2-D partitioning).
**Reason for default.** Watershed follows real image seams, so splits look intentional; the Voronoi fallback guarantees flat regions still split into compact, round-ish cells. Both are exactly seedable.

**Complexity.** O(N_r log N_r) per oversized region (N_r = its pixels). **Memory.** O(N_r).
**Failure modes.** Watershed producing < k catchments (very flat gradient): fall through to the Voronoi branch for the remainder — never an error.
**Edge cases.** Region exactly A_max: not split (strict >). Ring-shaped regions: distance-transform seeding handles them; cells may be annular sectors. Disabled → stage is identity.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `split.enabled` | bool | true (`easy`: false) | — |
| `split.split_factor` | float | 40.0 | 4–400 |
| `split.flat_sigma` | float | 2.0 | 0–10 (LAB units) |

**Quality requirements.** All products ≥ `A_min`; products of a textured split must have mean internal gradient ≤ boundary gradient (seams sit on content edges); determinism.
**Benchmarks.** ≤ 1.0 s when ≤ 10 % of page area is oversized.
**Unit tests.** Flat disk splits into k compact cells (compactness = 4πA/P² ≥ 0.5 each); textured synthetic (two-lobe gradient) splits on the lobe boundary; sub-floor product folding; identity when disabled.
**Future improvements.** Saliency-weighted `A_max` (§14.3); curvature-aware seam scoring.

---

## 13. Module: Contour Extraction (crack tracing)

**Purpose.** The raster→vector crossing: extract the exact boundary geometry of the region partition as polylines on the crack grid, watertight **by construction** (invariant I3).

**Responsibilities.** Trace every region's boundary loops (outer boundary plus holes) along the crack grid; output closed, consistently oriented loops. Each crack edge is traversed exactly **once per bordering region** (i.e., once per directed side): at a T-junction a corner has odd undirected crack-degree, so undirected once-only loops do not exist — per-side traversal is the correct, always-well-defined formulation.

**Input.** component map. **Output.** set of closed, oriented crack polylines with per-edge (left_id, right_id) — the raw material of §14.
**Dependencies.** `foundation/geometry` only.

**Algorithm (default): boundary walking on the crack grid.**
1. Vertices are pixel corners at half-integer coordinates (§1.3). A **crack edge** is the unit segment between two adjacent corners separating two pixels with different region ids (or a pixel from the page exterior).
2. Build the set of **directed** crack edges: each crack edge yields two directed versions, each carrying the region id on its left (exterior = −1). Keep the interior-region sides (left ≠ −1).
3. From the lexicographically smallest unvisited directed edge, walk keeping the same region on the left: at each corner, among unvisited outgoing directed edges with that left region, choose by fixed turn priority **left, straight, right** relative to the incoming direction. Mark directed edges visited. A walk terminates when it returns to its start edge — always a closed loop on a finite grid.
4. Repeat until no unvisited directed edges. Each loop carries its region id; the (left, right) pair per crack edge follows directly.

**Algorithm alternatives.** OpenCV `findContours` (traces pixel *centers* per region independently → adjacent regions get two disagreeing contours, gaps/overlaps guaranteed — disqualified, this is the classic watertightness bug); marching squares on per-region masks (same double-tracing flaw); polygonization via Shapely union of pixel squares (O(N log N) with heavy constant, loses the shared-edge structure).
**Reason for default.** Crack tracing is the only approach in which the boundary between regions A and B is *one* shared polyline by construction — I3 cannot be violated because there is nothing to disagree. Everything downstream (§14–16) depends on this shared-ness.

**Complexity.** O(N) — every directed crack edge visited exactly once (≤ 2B traversals). **Memory.** O(B), B = crack-edge count (≤ 2N + 2(H+W)).
**Failure modes.** None on a valid component map (every crack edge has exactly one continuation under the turn rule — provable on the quad grid).
**Edge cases.** Single region: output is exactly the page-border rectangle loop. 8-adjacent diagonal regions: the shared corner is traversed twice with different turn decisions — the turn-priority rule resolves it deterministically (this is why 4-connectivity is an invariant). One-pixel-wide necks: produce back-to-back crack edges; legal.

**Configuration parameters.** None. This stage has no knobs by design.
**Quality requirements.** Every directed (per-side) crack edge appears in exactly one loop; Σ signed loop areas over all interior-region loops (holes opposite sign) = page area (shoelace, exact in half-integer arithmetic — use integer doubled-coordinates); per-edge (left, right) ids consistent with the component map.
**Benchmarks.** ≤ 0.6 s at 1600 px, B ≤ 400 000.
**Unit tests.** Single pixel region → unit square loop; 2-region vertical split → three loops sharing edges consistently; diagonal checkerboard corner rule; shoelace area identity (property test on random small maps); exactly-once directed-edge coverage.
**Future improvements.** Numba/Cython port (identified in ARCHITECTURE.md §13.3 as a sanctioned hotspot) with the pure-Python version retained as reference.

---

## 14. Module: Topology Graph

**Purpose.** Impose node/edge structure on the raw crack loops: find junctions and cut loops into **arcs** (maximal boundary pieces separating exactly one pair of regions).

**Responsibilities.** Junction detection; loop cutting; arc identity assignment; per-arc (left_region, right_region) annotation.

**Input.** crack loops from §13 + component map. **Output.** junction set V + arc set A (open polylines between junctions, or closed loops with a degenerate anchor when a boundary meets no junction).
**Dependencies.** `foundation/geometry`.

**Algorithm (default).**
1. **Junctions:** a crack-grid corner is a junction iff the 2×2 pixel block around it contains **≥ 3 distinct region ids** (page exterior counts as an id), or it is one of the 4 page corners.
2. Walk each loop from §13; cut it at every junction vertex. Each resulting piece is an arc; its (left, right) pair is constant along the piece (guaranteed: a change of pair implies a junction by definition 1).
3. A loop containing no junction (an island boundary) becomes a **closed arc**; its anchor vertex is its lexicographically smallest corner (determinism).
4. Arc ids: assigned in order of (min corner of arc, left id, right id) — stable across runs.

**Algorithm alternatives.** Junctions from degree-counting on the crack-edge graph (equivalent but requires building the full edge graph first — more memory, same result); DCEL construction directly during §13 (fuses two hard stages; kept separate for testability, per ARCHITECTURE.md's boundary-crossing rule).
**Reason for default.** The 2×2 test is a local, O(1)-per-corner, provably complete junction criterion on the crack grid; it needs no global structure and is independently property-testable.

**Complexity.** O(B + V). **Memory.** O(B).
**Failure modes.** An arc whose (left, right) pair changes mid-walk → internal contract violation → `StageError` (indicates a §13 bug; checked always, cheap).
**Edge cases.** Junction of degree 4 (four regions meeting at a corner): one junction, four incident arcs. Closed arcs (islands): handled by rule 3. Page corners are always junctions even between only 2 ids — keeps the border decomposable.

**Configuration parameters.** None.
**Quality requirements.** Every crack edge belongs to exactly one arc; every arc endpoint is a junction (or the arc is closed); Σ arc lengths = B.
**Benchmarks.** ≤ 0.3 s at 1600 px.
**Unit tests.** T-junction of 3 regions → 3 arcs, 2 junctions (T-point + border interactions per fixture); island → 1 closed arc, 0 junctions; degree-4 corner; pair-constancy contract; arc-id stability.
**Future improvements.** None anticipated; this stage is deliberately minimal.

---

## 15. Module: Arc Graph

**Purpose.** Assemble the planar map: faces (regions as ordered arc walks) over the arc/junction sets, verified by Euler's formula, and convert coordinates to physical units — the last stage allowed to know about pixels.

**Responsibilities.** Face construction; face↔region correspondence; outer-face handling; Euler check; single application of `work_scale`.

**Input.** arcs + junctions (§14), component map (for correspondence), page geometry. **Output.** `ArcGraph` artifact: arcs (point coordinates), faces (arc walks with orientation flags), face→region→palette-label mapping.
**Dependencies.** `foundation/geometry`, `foundation/units`.

**Algorithm (default): half-edge face walking.**
1. Represent each arc as two directed half-arcs. At every junction, sort outgoing half-arcs by the angle of their first segment (atan2, ties impossible on the crack grid).
2. Face walk: repeatedly take an unvisited half-arc, walk "next = reverse, then rotate clockwise one slot" (standard planar-map face traversal) until closure. Each walk is one face boundary; holes attach to their containing face by matching the arc's region ids (a face's hole loops are the closed walks whose *outer* side is that face's region).
3. Identify each face with its region id via the arcs' (left, right) annotations (consistency across the walk is asserted). The exterior face is the one identified with id −1.
4. **Euler check:** `V − E + F = 2` counting the exterior face, arcs as E, junctions as V (closed arcs contribute a virtual anchor vertex). Failure → `StageError` (a §13/§14 bug surfaced; never repaired silently).
5. **Scale:** map every coordinate `(x_px, y_px) → ((x_px + 0.5)·s + m_x, (y_px + 0.5)·s + m_y)` where `s = content_width_pt / W` (aspect preserved; letterboxed within content box; `m` = content offset incl. centering). This is the only place `s` is applied (§1.3). Store `s` in provenance.

**Algorithm alternatives.** Full DCEL library (e.g. CGAL bindings — heavy native dependency for functionality needed once); face assembly by point-in-polygon containment testing (O(F²) and float-fragile); skipping face structure and rendering per-region contours (reintroduces the double-boundary flaw §13 eliminated).
**Reason for default.** Half-edge face walking is the textbook O(E) planar-map algorithm; on crack-grid geometry all angle comparisons are exact (axis-aligned unit segments), so it is robust without exact-arithmetic machinery.

**Complexity.** O(E + V log d) for the angular sorts (d = max junction degree ≤ 4 on the crack grid → effectively O(E)). **Memory.** O(E).
**Failure modes.** Euler violation or face/region mismatch → `StageError` with diagnostic dump (arc-graph JSON) per ARCHITECTURE.md §11.
**Edge cases.** R = 1: two faces (region + exterior), one arc (border loop). Nested islands (region in hole in region): hole attachment recursion depth is unbounded but data-driven; handled iteratively. Faces with multiple holes: hole loops listed in deterministic order (min anchor).

**Configuration parameters.** None.
**Quality requirements.** Euler identity holds; every arc borders exactly 2 faces; Σ|face areas| (shoelace, holes negative) = content area within ±0.01 % after scaling (float tolerance; exact pre-scaling).
**Benchmarks.** ≤ 0.3 s for A ≤ 20 000.
**Unit tests.** Two-region page (V/E/F counts exact); donut (hole attachment); nested donut; Euler property test over random label maps; scale applied exactly once (provenance value vs coordinate ratio).
**Future improvements.** None anticipated; frozen with §13/§14 as "the heart".

---

## 16. Module: Polygon Simplification

**Purpose.** Reduce staircase crack polylines (unit steps) to visually smooth, low-vertex-count polylines without breaking topology (I3) or sidedness (a point of region A must stay in region A).

**Responsibilities.** Per-arc vertex reduction; junction pinning; topology guard against arc–arc crossings.

**Input.** `ArcGraph` (points). **Output.** `ArcGraph` with simplified arcs (same topology).
**Dependencies.** `foundation/geometry` (spatial hash, predicates).

**Algorithm (default): Visvalingam–Whyatt with sidedness guard.**
1. Per arc: compute each interior vertex's *effective area* (triangle area of the vertex with its neighbors). Min-heap over vertices.
2. Remove vertices in ascending effective-area order while area < `ε = (simplify_tolerance_mm · 72/25.4)²` pt² (default tolerance 0.15 mm), recomputing neighbor areas after each removal (standard VW).
3. **Guard:** before each removal, query a global spatial hash (cell = 2·tolerance) for foreign vertices/segments inside the removal triangle; if any arc's geometry or any face's label anchor region would be crossed, skip that vertex permanently.
4. Arc endpoints (junctions) are never removed. Closed arcs keep ≥ 4 vertices (min area shape). Every arc keeps ≥ 2 interior candidates removed only while its total vertex count > 2.
5. Arcs processed in arc-id order; the spatial hash updates incrementally (determinism).

**Algorithm alternatives.** Douglas–Peucker (preserves extreme points → keeps staircase spikes; tolerance is a distance not an area, correlates worse with perceived smoothness); Reumann–Witkam (fast, low quality); topology-aware simplification via full constrained triangulation (correct but an order of magnitude more machinery).
**Reason for default.** VW's area criterion removes staircase micro-triangles first — exactly the artifact crack tracing produces — and degrades gracefully; with junction pinning plus the triangle guard, topology preservation is local and cheap.

**Complexity.** O(P log P) per arc (P = arc points) + O(1) expected guard queries. **Memory.** O(P_total).
**Failure modes.** None; skipping all removals is always legal.
**Edge cases.** 2-point arcs: pass through. Arcs shorter than tolerance: collapse to a straight segment between junctions (allowed — guard still applies). High-genus areas (many islands): guard prevents island/arc collisions.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `simplify.tolerance_mm` | float | 0.15 | 0.01–2.0 |

**Quality requirements.** Vertex reduction ≥ 80 % on the photo fixtures; zero arc–arc intersections post-stage (re-proved by §25.2); max Hausdorff deviation per arc ≤ 2× tolerance.
**Benchmarks.** ≤ 0.5 s for 400 000 → ≤ 80 000 vertices.
**Unit tests.** Staircase → straight line at tolerance; junction pinning; guard blocks a removal that would cross a nearby arc (constructed fixture); closed-arc minimum vertices; determinism.
**Future improvements.** Curvature-adaptive tolerance (tighter in salient areas, §14.3 weight map).

---

## 17. Module: Curve Smoothing

**Purpose.** Remove residual angular noise from simplified polylines before Bézier fitting, while preserving genuine corners — fitting quality (§18) improves markedly on pre-smoothed input.

**Responsibilities.** Corner detection; local smoothing of non-corner vertices; displacement bounding; junction pinning.

**Input.** simplified `ArcGraph`. **Output.** `ArcGraph` with smoothed arc geometry (same vertex counts, same topology).
**Dependencies.** `foundation/geometry`.

**Algorithm (default): corner-preserving Gaussian vertex smoothing.**
1. **Corners:** interior vertex v is a corner iff its turn angle (angle between incoming and outgoing segments) > `corner_angle_deg` (default 60°). Junction endpoints are always corners.
2. For every non-corner interior vertex: replace by the Gaussian-weighted average of the 5-vertex window (weights from σ = 1.0 vertex index units, renormalized at arc ends/near corners so corners never contribute).
3. **Clamp:** if a vertex moved more than `max_shift_mm` (default 0.2 mm, in pt), scale its displacement back to the bound.
4. One pass only (iteration would erode shape systematically).

**Algorithm alternatives.** Chaikin corner cutting (doubles vertex count, undoes §16); Laplacian smoothing with shrinkage compensation (Taubin λ|μ) — better for closed shapes, tuning-sensitive, marginal gain on already-simplified arcs; no smoothing at all (Bézier fitter then chases noise, producing wobbly tangents).
**Reason for default.** A single clamped Gaussian pass is parameter-light, cannot move any point beyond a printable-precision bound (so I1/I3 risk is bounded a priori), preserves corners exactly, and measurably lowers §18 fit error on all fixtures.

**Complexity.** O(P_total). **Memory.** O(window) = O(1) extra.
**Failure modes.** None.
**Edge cases.** Arc of 2–3 points: unchanged. All-corner arcs (rectilinear art): identity. Clamp active on nearly all vertices (huge tolerance config): still topologically safe because 0.2 mm < d_min/2 by config cross-rule.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `smooth.corner_angle_deg` | float | 60.0 | 15–120 |
| `smooth.max_shift_mm` | float | 0.2 | 0.01–1.0; must be < `quality.d_min_mm`/4 (cross-rule) |

**Quality requirements.** Curvature energy (Σ turn-angle²) reduced ≥ 40 % on photo fixtures; corner positions bit-identical; max displacement ≤ bound (asserted).
**Benchmarks.** ≤ 0.1 s for 80 000 vertices.
**Unit tests.** Right-angle corner untouched; sine-noise polyline flattens; clamp activates at constructed displacement; junction pinning; single-pass idempotence is *not* required (documented) but determinism is.
**Future improvements.** Taubin smoothing as a registered alternative for closed island arcs.

---

## 18. Module: Bézier Fitting

**Purpose.** Convert each arc polyline into a compact chain of cubic Bézier segments with G1 continuity inside arcs — the final geometry that renderers consume.

**Responsibilities.** Per-arc least-squares cubic fitting with adaptive splitting; corner handling; exact junction interpolation; fit-error guarantee.

**Input.** smoothed `ArcGraph`. **Output.** `CurveSet` (per-arc Bézier chains; faces carried over unchanged).
**Dependencies.** `foundation/geometry`.

**Algorithm (default): Schneider fitting (Graphics Gems, "An Algorithm for Automatically Fitting Digitized Curves") with corner splitting.**
1. Split each arc at its corners (§17 definition, recomputed on final geometry) into corner-free runs.
2. Per run: estimate end tangents (average of first/last 3 chords); fit one cubic by least squares over chord-length parameterization; measure max squared deviation; if > `fit_error_mm` (default 0.25 mm, in pt): apply ≤ 4 Newton–Raphson reparameterization iterations, refit; if still exceeding, split at the max-error point with a centripetal tangent estimate and recurse.
3. Within a run, adjacent segments share the split point and a mirrored tangent → G1. At corners, tangents are independent → intentional C0.
4. Endpoints of every chain interpolate the arc's junction coordinates **exactly** (watertightness at junctions is positional identity, not tolerance).
5. Degenerate runs (2 points): emit a single segment with control points at ⅓ and ⅔ chord (an exact line).

**Algorithm alternatives.** Potrace-style fitting (designed for bilevel pixel outlines; brings its own tracing assumptions, license (GPL) friction); global B-spline fit then conversion (better smoothness across whole arcs, but junction interpolation and corner control get harder); circular-arc + line fitting (great for CAD, poor for organic shapes).
**Reason for default.** Schneider is the industry-standard local fitter: bounded error by construction, exact endpoint interpolation, natural corner handling, simple determinization, and 30 years of known behavior — the right choice for a 10-year document.

**Complexity.** O(P log P) typical (recursion depth log in error decay); worst case O(P²) on adversarial arcs, bounded by max recursion 32 → then per-point segments. **Memory.** O(P).
**Failure modes.** Recursion floor reached → segments equal polyline edges (always succeeds; quality degrades, never correctness).
**Edge cases.** Closed arcs: cut at anchor, fitted as a run whose two ends meet at the anchor with independent tangents (anchor is a corner by definition). Collinear runs → single segment. Runs shorter than fit error → single line segment.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `bezier.fit_error_mm` | float | 0.25 | 0.02–2.0 |
| `bezier.corner_angle_deg` | (shared with `smooth.corner_angle_deg`) | 60.0 | — |

**Quality requirements.** Max deviation chain↔polyline ≤ fit error (sampled at 32 points/segment); segment count ≤ 0.15 × input vertex count on photo fixtures; junction coordinates exact to the double-precision value.
**Benchmarks.** ≤ 1.0 s for 80 000 vertices → ≤ 12 000 segments.
**Unit tests.** Quarter-circle polyline fits within error with ≤ 2 segments; straight line → exact 1 segment; corner produces C0 break; junction exactness; reparameterization improves error on a constructed case; determinism.
**Future improvements.** Numba hotspot port (sanctioned, §13.3 of ARCHITECTURE.md); curvature-continuous (G2) fitting as a registered alternative.

---

## 19. Module: Label Placement

**Purpose.** Place every region's number where a human can read it and unambiguously associate it with the region (invariant I4's readability half).

**Responsibilities.** Per-face anchor computation; font sizing; leader-line fallback for slivers; overlap resolution.

**Input.** `CurveSet` (faces with holes), `RegionGraph` (region↔palette label), quality config. **Output.** `LabelPlan`.
**Dependencies.** `foundation/geometry` (polylabel, point-in-face, distance).

**Algorithm (default).**
1. **Anchor:** pole of inaccessibility per face via quadtree polylabel on the Bézier chains flattened at 0.1 mm tolerance, precision 0.5 pt; holes respected. Yields anchor `c` and clearance radius `r` (pt).
2. **Font size:** `size = clip(1.35·r, font_min_pt, font_max_pt)` — derived from digit aspect: two digits at cap height `h` occupy a bounding box ≲ (1.4h × h), which fits a circle of radius r when `h ≤ 1.35r/√(1.4²+1)·2` simplified with the bundled font's metrics to the 1.35 factor. The exact rule: the rendered text bbox (bundled font metrics, §23) must fit within the clearance circle; 1.35·r is the closed-form seed, then shrink-to-fit.
3. **Leader lines:** if fitting requires `size < font_min_pt`, place the number outside: candidate anchor positions on a 4 mm ring around the face, in whitespace (distance to all geometry > text bbox radius), nearest-first; connect with a straight leader from the number to the face's pole; leader may not cross ≥ 3 arcs (else pick next candidate; if all fail → finding for §25.3).
4. **Overlap resolution:** sort labels by clearance descending; greedily keep; any label whose bbox intersects a kept bbox is displaced along the gradient of its face's distance transform by up to r/2, else demoted to leader line.

**Algorithm alternatives.** Centroid anchors (fails on concave/annular faces — centroid may lie outside); medial-axis midpoint (more code, equivalent result to polylabel); ILP-based global label placement (optimal, heavy solver dependency, nondeterministic solver paths).
**Reason for default.** Polylabel is the de-facto standard for "most interior point", strictly correct on holes and concavity, has a tunable precision/time knob, and greedy-by-clearance resolves overlaps well when label density is low (guaranteed by `A_min`).

**Complexity.** O(F · q) with q = polylabel quadtree iterations (~100s); overlap pass O(F log F + F·k) via spatial hash. **Memory.** O(F).
**Failure modes.** No valid leader anchor for a face → recorded as `Finding(severity=FATAL, invariant="I4")`; the validator decides abort (§25).
**Edge cases.** Annular faces: pole lies inside the ring — correct by construction. Face smaller than any readable label but ≥ A_min with extreme aspect (long sliver): leader path. Two-digit vs one-digit numbers use their real bbox (no worst-casing). Number glyphs for K > 99 are out of contract (`n_colors ≤ 64`).

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `labels.polylabel_precision_pt` | float | 0.5 | 0.05–2 |
| `labels.leader_ring_mm` | float | 4.0 | 1–20 |
| (font bounds from `quality.font_min_pt/font_max_pt`) | | | |

**Quality requirements.** 100 % of faces labeled (in-region or leader); 0 label-bbox overlaps; 0 label bboxes crossing region boundaries (in-region labels); label fit rate (in-region without shrink) ≥ 90 % on photo fixtures.
**Benchmarks.** ≤ 1.0 s for F = 800.
**Unit tests.** Pole of a C-shape lies inside the C; annulus; font-size formula vs brute-force bbox check; leader fallback on a 1×30 mm sliver; overlap displacement determinism.
**Future improvements.** Simulated-annealing global placement (seeded) as a registered alternative for dense pages.

---

## 20. Module: Palette Optimization

**Purpose.** Renumber palette entries so the printed numbers maximize the *mystery* effect (the motif must not be guessable from number patterns) while keeping the legend pleasant and the coloring experience balanced.

**Responsibilities.** Produce the permutation `printed_number = π(palette_index)`; never touch geometry or colors.

**Input.** `Palette`, `RegionGraph` (coverage + adjacency). **Output.** the permutation (consumed by §21's `LegendPlan`).
**Dependencies.** `foundation/color`.

**Algorithm (default): deterministic greedy anti-correlation ordering.**
1. Score every palette pair: `w(a,b) = ΔE00(a,b)`.
2. Build the printed order greedily: start from the color with **median** luminance (breaks the light→dark giveaway); repeatedly append the unnumbered color maximizing `min(ΔE00 to the last 2 numbered colors)` — consecutive numbers are always perceptually distant, so adjacent numbers on paper don't hint at adjacent tones. Ties: larger total coverage, then lower original index.
3. **Spatial check:** compute the number-vs-luminance Spearman correlation |ρ|; if |ρ| > 0.4 (ordering accidentally reconstructed a tonal ramp), apply the fixed derangement `π' = π ∘ (interleave first-half/second-half)` and keep whichever of π, π' has lower |ρ| (ties → π).
4. Background rule: the highest-coverage color is never printed number 1 (swap with the next slot if it lands there) — number 1 regions are where people start; starting on the background spoils composition early.

**Algorithm alternatives.** Seeded random shuffle (destroys any structure but also any *guarantee*; can produce ρ ≈ 1 by chance); simulated annealing over a multi-term objective (better optima, more machinery and tuning for an aesthetic with weak gradients); identity/luminance ordering (actively anti-mystery — disqualified).
**Reason for default.** The greedy max-min-ΔE00 walk is O(K²), deterministic without consuming randomness, and directly encodes the two measurable proxies of "mystery": tonal-ramp destruction and neighbor-number contrast. K ≤ 64 makes optimality irrelevant.

**Complexity.** O(K²). **Memory.** O(K²) for the ΔE table.
**Failure modes.** None.
**Edge cases.** K = 2: permutation is identity or swap by rule 4 only. All-equal colors (post-merge impossible, ΔE ≥ merge threshold — documented reliance).

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `palette_order.enabled` | bool | true | false → identity numbering (debug) |
| `palette_order.max_luma_spearman` | float | 0.4 | 0–1 |

**Quality requirements.** |Spearman(number, L)| ≤ 0.4; min ΔE00 between consecutively numbered colors ≥ 0.8× the palette's global median pairwise ΔE00 on fixtures; determinism.
**Benchmarks.** ≤ 5 ms at K = 30.
**Unit tests.** Grayscale ramp input yields |ρ| ≤ 0.4; background-not-1 rule; greedy tie-break determinism; disabled → identity.
**Future improvements.** Learned mystery scorer (§14.4 plugin) as advisor proposing a permutation through the same interface.

---

## 21. Module: Legend Generation

**Purpose.** Lay out the number↔color key: chips with printed numbers, arranged in the reserved page band, in printed-number order.

**Responsibilities.** Chip grid geometry; band fitting with graceful shrink; ordering per §20's permutation; carrying the renumber map into provenance.

**Input.** `Palette`, permutation (§20), page geometry. **Output.** `LegendPlan` (chip positions/sizes in pt, palette order, renumber map).
**Dependencies.** `foundation/units`.

**Algorithm (default).**
1. Reserve the legend band at the bottom of the content box: height `band = rows·(chip + gap) + gap` (the artwork's content box shrinks accordingly *before* §15 computes `s` — the config cross-rule orders this: page layout is resolved at config time, so the band is a constant of the run).
2. Chip: square `chip_mm` (default 6) with 1.5 pt-radius corner rounding, 0.3 pt black outline, filled with the palette sRGB; number printed right of the chip at `max(font_min_pt, 0.6·chip_pt)` in black.
3. Grid: fill left→right, top→bottom, in printed-number order; per-row capacity `⌊(content_width + gap) / (cell_width + gap)⌋` where cell width = chip + number width (2-digit metrics) + inner pad.
4. If K chips need more than `max_rows` (default 3): shrink `chip` stepwise by 0.5 mm to a floor of 4 mm; if still overflowing → `QualityError` ("palette too large for page format").

**Algorithm alternatives.** Right-side vertical legend (steals width from portrait artwork — worse for the dominant page formats); separate legend page (breaks the single-page product definition); flowed text legend (harder to color-match visually).
**Reason for default.** A bottom band preserves artwork aspect on portrait pages (the dominant format), has trivially predictable geometry (config-time constant), and matches print-industry coloring-book convention.

**Complexity.** O(K). **Memory.** O(K).
**Failure modes.** Overflow after shrink floor → `QualityError` (never silent truncation).
**Edge cases.** K = 2: single short row, band still reserved at fixed 1-row height minimum. Landscape pages: same rule (band at bottom).

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `legend.chip_mm` | float | 6.0 | 3–15 |
| `legend.gap_mm` | float | 2.0 | 0.5–10 |
| `legend.max_rows` | int | 3 | 1–6 |

**Quality requirements.** All chips within the content box; chip color = palette sRGB exactly; numbers = §20 permutation exactly; deterministic layout.
**Benchmarks.** ≤ 1 ms.
**Unit tests.** Row capacity arithmetic; shrink cascade; overflow → `QualityError`; band height vs config cross-rule; permutation pass-through.
**Future improvements.** Optional per-chip region counts ("12 × №5") for difficulty labeling.

---

## 22. Module: SVG Export

**Purpose.** The canonical output renderer: byte-deterministic, structurally clean SVG line art that all other outputs must geometrically agree with.

**Responsibilities.** Serialize curves, labels, leader lines, legend, and page furniture into one SVG document; enforce the determinism contract (I2's test surface).

**Input.** `CurveSet`, `LabelPlan`, `LegendPlan`, page config. **Output.** SVG bytes (UTF-8).
**Dependencies.** none beyond stdlib string building (svgwrite is *not* used: writer libraries do not guarantee attribute ordering across versions — a direct serializer does).

**Algorithm (default).**
1. Document: `viewBox="0 0 W_pt H_pt"`, explicit `width/height` in mm. Fixed layer order: `<g id="regions">`, `<g id="labels">`, `<g id="leaders">`, `<g id="legend">`, `<g id="frame">`.
2. Regions: one `<path>` per **arc** (not per face — each boundary drawn once, half the ink, no double-stroke darkening), `d` from the Bézier chain (`M`, `C` commands), `stroke:#000; stroke-width:0.3pt; fill:none; stroke-linecap:round; stroke-linejoin:round`. Each path carries `data-left`/`data-right` printed numbers (enables downstream tooling and the solved preview).
3. Labels: `<text>` at anchor, `text-anchor:middle; dominant-baseline:central`, font family fixed to the bundled font name; leaders as `<line>` 0.25 pt.
4. Legend and frame per §21 geometry.
5. **Determinism rules:** all coordinates formatted with exactly 3 decimal places (`format(x, '.3f')`, negative-zero normalized to `0.000`); elements emitted in id order; attribute order fixed by the serializer; no timestamps, no generator comments with versions that vary per run (engine version *is* included — it is part of the reproducibility record and constant for a build); LF newlines.

**Algorithm alternatives.** svgwrite/lxml builders (attribute-order and escaping drift across library versions breaks byte-determinism); per-face `<path>` with fills (needed only for the solved preview — done there, not here); SVGZ compression (breaks human inspectability, trivial to add downstream).
**Reason for default.** A direct serializer is ~200 lines the engine fully controls; I2's "byte-identical" promise cannot be delegated to a third-party writer's internals.

**Complexity.** O(segments + F). **Memory.** O(output size).
**Failure modes.** None (inputs already validated).
**Edge cases.** Fonts: family name references the *embedded-by-PDF* bundled font; SVG itself references by name only (SVG is line art for print workflows that install the font; the PDF is the self-contained deliverable — documented product decision). Numbers > 1-digit widths already handled in §19 bboxes.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `svg.stroke_pt` | float | 0.3 | 0.05–2 |
| `svg.decimals` | int | 3 | 2–5 (I2 hash fixtures pinned to 3) |

**Quality requirements.** Byte-identical across runs and platforms (the I2 CI gate hashes this output); valid SVG 1.1 (validated in CI); each arc appears exactly once.
**Benchmarks.** ≤ 0.3 s for 12 000 segments; output ≤ 2 MB typical.
**Unit tests.** Coordinate formatting (incl. −0); layer order; arc-once property; golden byte hash on the fixture ladder; XML validity.
**Future improvements.** Optional `data-` metadata toggle for a lighter file.

---

## 23. Module: PDF Export

**Purpose.** The self-contained print deliverable: exact trim size, embedded subset font, vector geometry identical to the SVG.

**Responsibilities.** Native vector re-render of the same plans (no SVG rasterization); font embedding; PDF metadata; geometric agreement with SVG.

**Input.** `CurveSet`, `LabelPlan`, `LegendPlan`, page config. **Output.** PDF bytes.
**Dependencies.** ReportLab (detail); bundled OFL-licensed font (DejaVu Sans, in `assets/fonts/`, pinned by hash).

**Algorithm (default).**
1. Page = trim size from config in pt; y-axis flipped once at the canvas transform (§1.3).
2. Draw the same primitives as §22 in the same order: Bézier paths (`curveTo`), texts, leaders, legend chips (rounded-rect + fill + stroke), frame.
3. Embed the bundled font subset; no system font may ever be referenced (cross-machine determinism of *metrics*; PDF bytes themselves are **not** hash-gated — ReportLab object numbering is not canonical; geometric agreement is the contract instead).
4. Metadata: title, engine version, resolved-config hash in the XMP/Info dict; creation date set to the fixed epoch value derived from the input hash (no wall clock — determinism).

**Algorithm alternatives.** Rasterize SVG → embed image (destroys vector print quality — disqualified); PyMuPDF drawing API (fine, heavier native dependency); cairo (native dependency, historically painful wheels); SVG→PDF converters like CairoSVG (adds a conversion layer whose fidelity we'd have to test anyway).
**Reason for default.** ReportLab is pure-Python-installable, 20+ years stable, with first-class font subsetting; re-rendering from the plans (rather than converting the SVG) keeps both outputs downstream of the *same* geometry, which is what the renderer-agreement contract test verifies.

**Complexity.** O(segments + F). **Memory.** O(output).
**Failure modes.** Bundled font missing/hash-mismatched → `StageError` at startup (asset integrity check).
**Edge cases.** Page sizes beyond ReportLab defaults: explicit width/height always set. Very large K legends: geometry already resolved by §21.

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `pdf.enabled` | bool | true | — |
| `pdf.embed_solved_page` | bool | false | appends the solved preview as page 2 |

**Quality requirements.** Renderer agreement: sampling 1 000 deterministic points per fixture along arcs, PDF-space and SVG-space positions agree within 0.05 pt; fonts embedded (verified by parsing the output); trim box exact.
**Benchmarks.** ≤ 0.5 s for 12 000 segments.
**Unit tests.** Trim box; y-flip correctness on an asymmetric fixture; font-embedding presence; agreement contract test vs §22; fixed-date metadata.
**Future improvements.** PDF/X-1a compliance profile for print bureaus.

---

## 24. Module: PNG Preview

**Purpose.** Two raster previews: **line art** (what the customer prints) and **solved** (regions flood-filled with palette colors — also the fidelity probe for I1).

**Responsibilities.** Flatten Béziers; scanline-fill faces (solved); stroke arcs (line art); compose legend; fixed-resolution output.

**Input.** `CurveSet`, `LabelPlan`, `LegendPlan`, `Palette`, page config. **Output.** `{"lineart": png, "solved": png}` bytes.
**Dependencies.** Pillow (detail); `foundation/geometry` (flattening).

**Algorithm (default).**
1. Raster size: page at `preview_dpi` (default 150).
2. Flatten every Bézier chain with adaptive subdivision, tolerance 0.1 px at preview scale.
3. **Solved:** even-odd scanline polygon fill per face (outer ring + holes) in ascending printed-number order with the palette sRGB (8-bit, rounded half-up); then stroke arcs 1 px black; no labels.
4. **Line art:** white canvas; stroke flattened arcs with a round-capped 1-px pen; draw labels and legend using Pillow's text rendering with the bundled font.
5. Encode PNG with fixed encoder settings (compression 6, no ancillary time chunks) — previews are hash-gated like the SVG.

**Algorithm alternatives.** Rasterizing the SVG via resvg/CairoSVG (adds a native dependency and a second geometry interpretation; disqualified as *default*, sanctioned as an optional high-fidelity plugin); supersampling + downscale (2× cost; anti-aliasing is cosmetic for a preview, and the solved preview must be *un*-antialiased along fills for the SSIM probe to be meaningful — filled with hard edges, matching the quantized raster's nature).
**Reason for default.** Scanline fill over flattened faces uses the engine's own topology (holes handled by the face structure, not by winding heuristics), keeps zero native dependencies, and produces the exact per-pixel color classes §25.1 needs.

**Complexity.** O(preview pixels + segments·subdivisions). **Memory.** O(preview pixels).
**Failure modes.** None (geometry pre-validated).
**Edge cases.** Faces thinner than 1 preview px: fill may be empty, stroke still marks them — SSIM tolerance absorbs it. Holes: even-odd with explicit hole rings (never winding-dependent).

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `preview.dpi` | int | 150 | 72–300 |

**Quality requirements.** Solved preview SSIM vs quantized working raster (both resampled to the smaller grid, luminance SSIM, 8×8 window) ≥ 0.985 (the I1 proxy — this number is a release gate); byte-deterministic PNGs.
**Benchmarks.** both previews ≤ 1.0 s at 150 DPI Letter.
**Unit tests.** Even-odd hole filling on a donut; fill color exactness; flattening tolerance; PNG byte determinism; SSIM ≥ threshold on fixtures.
**Future improvements.** resvg-based "marketing quality" preview plugin.

---

## 25. Module: Validation

**Purpose.** Prove invariants I1–I4 on every run — enforcement, not convention (ARCHITECTURE.md §0). Four validators plus report aggregation; declared auto-repairs only.

**Responsibilities.** Fidelity audit; topology proof; printability check with repairs; palette distinguishability; aggregate `ValidationReport`s; decide pass/abort (`QualityError` on unrepairable FATAL).

**Input.** full pipeline context (all artifacts). **Output.** `ValidationReport` per validator; gate decision.
**Dependencies.** `foundation/geometry`, `foundation/color`.

### 25.1 Fidelity (I1)

1. **Correspondence audit:** rasterize each face id at working resolution (same scanline filler as §24); for every face, the majority label of its covered pixels in the post-§12 label map must equal the face's label, and agreement must be ≥ 99.0 % of its pixels. Below → FATAL.
2. **SSIM probe:** as specified in §24 quality requirements; < 0.985 → FATAL.

### 25.2 Topology (I3)

1. Re-prove (independently of §15's construction): every arc borders exactly 2 faces; Σ shoelace face areas (holes negative) = content area ± 0.01 %; no arc self-intersects and no two arcs intersect except at shared junction endpoints (segment sweep with spatial hash, exact orientation predicates from `foundation/geometry`). Any violation → FATAL (never repaired — a topology repair is a lie).

### 25.3 Printability (I4)

1. Per face: inscribed diameter `2r` (from §19's polylabel) ≥ `d_min_mm`? If not and the face has an in-region label → **declared repair**: demote to leader line (re-invokes §19 step 3 for that face; logged WARNING). If no leader placement exists → FATAL.
2. Every face has a label plan entry; every label ≥ `font_min_pt`. Violation → FATAL.

### 25.4 Palette

1. Min pairwise ΔE00 ≥ `quantize.merge_delta_e` (construction re-check, FATAL if violated — indicates a §7 bug).
2. Min pairwise ΔE00 < `palette_warn_delta_e` (default 12) → WARNING ("colors hard to distinguish for young solvers") — preset `easy` raises it to FATAL.

**Algorithm alternatives.** Trust-by-construction (no validator — rejected: ARCHITECTURE.md I1–I4 demand independent proof); sampling-based audits (cheaper, but the full audit fits the budget).
**Reason for default.** Each check re-derives its invariant from raw artifacts by a *different* method than the constructing stage — independent double-entry bookkeeping.

**Complexity.** O(N + E log E + F·q). **Memory.** O(N).
**Failure modes.** This module *produces* failures; its own failure mode is an exception during checking → `StageError` (a validator crash is never a pass).
**Edge cases.** Degenerate one-region page: all checks pass trivially, WARNING "degenerate page" recorded (from §11).

**Configuration parameters.**

| Key | Type | Default | Range |
|---|---|---|---|
| `validate.palette_warn_delta_e` | float | 12.0 | 0–40 |
| `validate.fidelity_min_agreement` | float | 0.99 | 0.9–1.0 |
| `validate.ssim_min` | float | 0.985 | 0.9–1.0 |

**Quality requirements.** Validators are deterministic; a repaired run re-validates from scratch (repair → full re-check loop, max 2 iterations, then FATAL).
**Benchmarks.** full validation ≤ 2.0 s on the 2 MP fixture.
**Unit tests.** Each check triggered by a purpose-built broken artifact (corrupted face label, gap injected into an arc, sub-d_min face, near-duplicate palette); repair loop convergence; validator-crash → StageError path.
**Future improvements.** Learned quality critic as an additional soft validator (§14.5 plugin).

---

## 26. Cross-module acceptance matrix

| # | Module | Hard gate(s) it owns | Budget (2 MP fixture) |
|---|---|---|---|
| 4 | Raster Load | decode correctness | 0.15 s |
| 5 | Preprocessing | edge-preservation SSIM ≥ 0.85 | 0.8 s |
| 6 | Color Analysis | proposal determinism | 0.1 s |
| 7 | Quantization | mean ΔE00 ≤ 11; palette separation | 2.0 s |
| 8 | Noise Removal | zero isolated pixels | 0.5 s |
| 9 | Connected Components | exact partition | 0.2 s |
| 10 | Region Graph | boundary-length identity | 0.3 s |
| 11 | Tiny Region Merge | zero sub-A_min regions | 1.0 s |
| 12 | Large Region Split | products ≥ A_min | 1.0 s |
| 13 | Contour Extraction | exact once-coverage; area identity | 0.6 s |
| 14 | Topology Graph | pair-constancy; Σ arc len = B | 0.3 s |
| 15 | Arc Graph | Euler identity; single scaling | 0.3 s |
| 16 | Simplification | zero intersections; ≥ 80 % reduction | 0.5 s |
| 17 | Curve Smoothing | displacement bound | 0.1 s |
| 18 | Bézier Fitting | fit error bound; exact junctions | 1.0 s |
| 19 | Label Placement | 100 % labeled; zero overlaps | 1.0 s |
| 20 | Palette Optimization | \|Spearman\| ≤ 0.4 | 0.01 s |
| 21 | Legend | fits or QualityError | 0.01 s |
| 22 | SVG Export | byte determinism (I2 gate) | 0.3 s |
| 23 | PDF Export | geometric agreement ≤ 0.05 pt | 0.5 s |
| 24 | PNG Preview | SSIM ≥ 0.985 (I1 probe) | 1.0 s |
| 25 | Validation | I1–I4 | 2.0 s |
| | **Total** | | **≤ 13.6 s < 15 s target** |

## 27. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-06 | Initial complete specification (22 modules). |

