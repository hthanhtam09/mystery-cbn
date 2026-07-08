# Mystery Color-by-Number Engine — Quality Specification

**Status:** v1.0 — authoritative quality contract. Companion to [ENGINE_SPEC.md](../ENGINE_SPEC.md) (algorithms), [MATH_SPEC.md](../MATH_SPEC.md) (formula definitions), [ARCHITECTURE.md](../ARCHITECTURE.md) (§9 benchmark framework).
**Rule:** every metric below is computed automatically by the benchmark/validation harness. No metric admits human judgement. A release ships only if every **Gate**-class metric is within its acceptable band on the full fixture ladder; **Monitor**-class metrics fail CI only on regression beyond the stated tolerance vs the committed baseline.

---

## 1. Measurement framework

### 1.1 Fixture ladder

All metrics are measured on the versioned fixture set (original, in-repo assets):

| Fixture id | Content class | Size |
|---|---|---|
| `F-photo-05` / `F-photo-2` / `F-photo-12` / `F-photo-24` | photograph | 0.5 / 2 / 12 / 24 MP |
| `F-illu-2` | flat-shaded illustration | 2 MP |
| `F-flat-2` | hard-edged flat art (logo-like) | 2 MP |
| `F-noise-2` | high-frequency noisy photo | 2 MP |
| `F-degen-1` | single flat color | 1 MP |

Unless a metric states otherwise: the **reference fixture** is `F-photo-2`, preset `medium`, seed 0, default config; "ladder" means all fixtures × presets {easy, medium, hard}.

### 1.2 Metric classes

- **Gate** — hard pass/fail band; any violation blocks release (maps to invariants I1–I4 or output validity).
- **Monitor** — tracked against `benchmarks/baselines/*.json`; CI fails on regression beyond the metric's tolerance. Baselines change only by explicit reviewed commit.

### 1.3 Template

Every metric defines: **Name · Definition · Formula · Target · Min acceptable · Max acceptable · Measurement method · Automatic benchmark.** "n/a" for a bound means the metric is one-sided. Formula symbols are defined in MATH_SPEC.md; section references given per metric.

---

## 2. Topology and geometry metrics

### QM-01 Topology Errors — **Gate**

- **Definition.** Count of violations of the planar-partition invariant I3: arcs bordering ≠ 2 faces, arc self-intersections, arc–arc intersections away from shared junctions, Euler-formula violations.
- **Formula.** `T_err = n_badarc + n_selfx + n_pairx + n_euler` (MATH_SPEC §5.2, §6.1, §7.3).
- **Target** 0 · **Min** 0 · **Max** 0.
- **Measurement.** Topology validator (ENGINE_SPEC §25.2): independent re-proof via filtered orientation predicates + spatial-hash segment sweep on final CurveSet geometry flattened at 0.1 mm.
- **Benchmark.** `benchmarks/quality/topology.py`, ladder-wide; also embedded in every production run (validator).

### QM-02 Watertightness Residual — **Gate**

- **Definition.** Relative gap between the sum of face areas and the content-box area.
- **Formula.** `W_res = | Σ_f A(f) − C_w·C_h | / (C_w·C_h)` (shoelace, holes negative; MATH_SPEC §7.1).
- **Target** 0 · **Min** n/a · **Max** 1×10⁻⁴.
- **Measurement.** Computed twice: exact integer form pre-scaling (must be exactly 0) and float form post-scaling (band above).
- **Benchmark.** `benchmarks/quality/topology.py`; per-run validator.

### QM-03 Contour Accuracy — **Gate**

- **Definition.** Maximum deviation of final vector boundaries from the exact crack-grid boundary of the label map.
- **Formula.** `C_acc = max_arc d_H(flatten(chain_arc), crack_arc)` in mm (discrete Hausdorff, MATH_SPEC §4.3), where both polylines are mapped to page frame.
- **Target** ≤ 0.30 mm · **Min** n/a · **Max** 0.55 mm (= simplify 0.15 + smooth clamp 0.20 + fit 0.25 budgets, minus expected slack; the max is the linear-sum worst case rounded down).
- **Measurement.** Per-arc Hausdorff on 32-sample flattening vs stored crack polyline (retained in debug artifacts for measurement runs).
- **Benchmark.** `benchmarks/quality/contour_accuracy.py`, ladder-wide max reported.

### QM-04 Bézier Fit Error — **Gate**

- **Definition.** Maximum distance between each fitted Bézier chain and its source (smoothed, simplified) polyline.
- **Formula.** `B_err = max_arc max_k ‖B(t_k) − p_k‖` in mm (MATH_SPEC §9.2), sampled at the fit parameters plus 32 uniform samples per segment.
- **Target** ≤ 0.25 mm (= `bezier.fit_error_mm` default) · **Min** n/a · **Max** 0.25 mm (the config value is the gate; changing config moves the gate).
- **Measurement.** Fitter self-report cross-checked by independent resampling in the validator.
- **Benchmark.** `benchmarks/quality/bezier.py`.

### QM-05 Vertex Reduction Ratio — Monitor

- **Definition.** Fraction of crack-polyline vertices removed by simplification.
- **Formula.** `V_red = 1 − P_out / P_in`.
- **Target** ≥ 0.85 (photo fixtures) · **Min acceptable** 0.80 · **Max** n/a.
- **Measurement.** Stage tracing counters (ENGINE_SPEC §16).
- **Benchmark.** `benchmarks/quality/simplify.py`; regression tolerance ±0.03 absolute.

### QM-06 Maximum Vertices per Output — Monitor

- **Definition.** Total Bézier segment count in the final SVG (proxy for file size and downstream editability).
- **Formula.** `S_total = Σ_arc segments(arc)`.
- **Target** ≤ 12 000 (`F-photo-2`, medium) · **Min** n/a · **Max acceptable** 20 000.
- **Measurement.** SVG structural parse (count `C` commands).
- **Benchmark.** `benchmarks/quality/complexity.py`; tolerance +10 % vs baseline.

### QM-07 Average Polygon Complexity — Monitor

- **Definition.** Mean Bézier segments per face boundary (outer ring + holes).
- **Formula.** `P_avg = Σ_f segments(∂f) / F`.
- **Target** 8–25 (photo fixtures) · **Min acceptable** 4 · **Max acceptable** 40.
- **Measurement.** CurveSet traversal. Below 4 indicates over-simplification (blobby output); above 40 indicates under-simplification.
- **Benchmark.** `benchmarks/quality/complexity.py`; tolerance ±20 % vs baseline.

### QM-08 Boundary Smoothness (Curvature Energy) — Monitor

- **Definition.** Scale-invariant angular noise of final boundaries.
- **Formula.** `E_κ = Σ_arcs Σ_k θ_k²` on 0.1 mm-flattened final geometry; reported as `E_κ / Σ_arcs length_mm` (rad²/mm) (MATH_SPEC §10, §16.4).
- **Target** ≤ 0.05 rad²/mm (`F-photo-2`) · **Min** n/a · **Max acceptable** 0.09 rad²/mm.
- **Measurement.** Flatten → turn angles → normalized sum; corners (|θ| > 60°) excluded from the sum (intentional corners are not noise).
- **Benchmark.** `benchmarks/quality/smoothness.py`; tolerance +15 % vs baseline.

### QM-09 Smoothing Displacement Bound — **Gate**

- **Definition.** Maximum vertex displacement introduced by curve smoothing.
- **Formula.** `D_max = max_v ‖v_smoothed − v_input‖` in mm.
- **Target** ≤ 0.20 mm · **Min** n/a · **Max** 0.20 mm (= `smooth.max_shift_mm`; clamp makes violation structurally impossible — the gate detects clamp bugs).
- **Measurement.** Stage assertion, re-verified by the benchmark from retained pre/post geometry.
- **Benchmark.** `benchmarks/quality/smoothness.py`.

---

## 3. Region and printability metrics

### QM-10 Minimum Region Diameter — **Gate**

- **Definition.** Smallest inscribed-circle diameter over all faces that carry an in-region label.
- **Formula.** `D_min_obs = min_{f ∈ in-region} 2·r*(f)` in mm (polylabel, MATH_SPEC §13.3).
- **Target** ≥ 3.5 mm (preset `medium`; 5.0 easy, 2.5 hard) · **Min** = preset `d_min_mm` · **Max** n/a.
- **Measurement.** Printability validator (ENGINE_SPEC §25.3); faces below the floor must be leader-labeled or the run aborts.
- **Benchmark.** `benchmarks/quality/printability.py`.

### QM-11 Maximum Tiny Region Percentage — **Gate**

- **Definition.** Share of faces whose area is below the printability area floor after the merge stage.
- **Formula.** `R_tiny = |{f : A(f) < A_min}| / F × 100 %` with `A_min = π(d_min/2)²` (MATH_SPEC §2).
- **Target** 0 % · **Min** n/a · **Max** 0 %.
- **Measurement.** Face-area sweep on final geometry (pt², converted to mm²).
- **Benchmark.** `benchmarks/quality/printability.py`; per-run validator.

### QM-12 Leader-Line Ratio — Monitor

- **Definition.** Fraction of faces labeled by leader line instead of in-region number.
- **Formula.** `L_ratio = n_leader / F × 100 %`.
- **Target** ≤ 5 % (medium) · **Min** n/a · **Max acceptable** 15 % (25 % for preset `hard`).
- **Measurement.** LabelPlan census.
- **Benchmark.** `benchmarks/quality/labels.py`; tolerance +3 points vs baseline.

### QM-13 Region Count Band — Monitor

- **Definition.** Total face count (puzzle richness proxy; both directions matter).
- **Formula.** `F` (exterior excluded).
- **Target** 300–900 (`F-photo-2`, medium) · **Min acceptable** 150 · **Max acceptable** 1 500.
- **Measurement.** ArcGraph census.
- **Benchmark.** `benchmarks/quality/complexity.py`; tolerance ±15 % vs baseline.

### QM-14 Mean Region Compactness — Monitor

- **Definition.** Mean isoperimetric quotient over faces.
- **Formula.** `Q̄ = (1/F) Σ_f 4π A(f)/P(f)²` (MATH_SPEC §16.3).
- **Target** ≥ 0.35 (`F-photo-2`) · **Min acceptable** 0.25 · **Max** n/a.
- **Measurement.** Flattened-perimeter shoelace sweep.
- **Benchmark.** `benchmarks/quality/complexity.py`; tolerance −0.05 vs baseline.

---

## 4. Color metrics

### QM-15 Color Accuracy (Quantization Fidelity) — **Gate**

- **Definition.** Mean perceptual error between the working raster and its palette assignment.
- **Formula.** `ΔĒ = (1/N) Σ_p ΔE00(raster(p), palette(ℓ(p)))` (MATH_SPEC §4.2).
- **Target** ≤ 9.0 (`F-photo-2`, K=16) · **Min** n/a · **Max acceptable** 11.0.
- **Measurement.** Full-raster sweep post-quantization (pre-denoise).
- **Benchmark.** `benchmarks/quality/color.py`.

### QM-16 Palette Separation — **Gate**

- **Definition.** Minimum pairwise perceptual distance within the final palette.
- **Formula.** `ΔE_min = min_{a≠b} ΔE00(pal_a, pal_b)`.
- **Target** ≥ 12.0 · **Min** = `quantize.merge_delta_e` (7.0 default; 12.0 is FATAL floor for preset `easy`) · **Max** n/a.
- **Measurement.** K×K ΔE00 table (palette validator, ENGINE_SPEC §25.4).
- **Benchmark.** `benchmarks/quality/color.py`.

### QM-17 Solved-Preview Fidelity (SSIM) — **Gate**

- **Definition.** Structural similarity between the solved preview and the quantized raster — the I1 proxy.
- **Formula.** MATH_SPEC §16.1 (8×8 windows, C₁=(0.01·255)², C₂=(0.03·255)², luminance).
- **Target** ≥ 0.992 · **Min acceptable** 0.985 · **Max** n/a.
- **Measurement.** Fidelity validator (ENGINE_SPEC §25.1), area-average resampling to common grid.
- **Benchmark.** `benchmarks/quality/fidelity.py`.

### QM-18 Face–Label Agreement — **Gate**

- **Definition.** Per-face pixel agreement between rasterized faces and the label map (I1 audit).
- **Formula.** `min_f agree(f)` with `agree` per MATH_SPEC §16.2.
- **Target** ≥ 0.995 · **Min acceptable** 0.990 · **Max** n/a.
- **Measurement.** Scanline rasterization of every face at working resolution; majority + ratio check.
- **Benchmark.** `benchmarks/quality/fidelity.py`; per-run validator.

### QM-19 Number–Luminance Correlation (Mystery Leakage) — **Gate**

- **Definition.** Rank correlation between printed numbers and palette lightness; low magnitude prevents tone-ramp guessing.
- **Formula.** `|ρ_s|` (Spearman, MATH_SPEC §15.2). Waived for K < 4.
- **Target** ≤ 0.25 · **Min** n/a · **Max acceptable** 0.40.
- **Measurement.** Exact rank arithmetic on LegendPlan vs palette L*.
- **Benchmark.** `benchmarks/quality/legend.py`.

---

## 5. Label and legend metrics

### QM-20 Label Collision Count — **Gate**

- **Definition.** Pairs of label bounding boxes (numbers, leader texts, legend items) with non-empty intersection, plus in-region label bboxes crossing any arc.
- **Formula.** `L_coll = |{(i,j): bbox_i ∩ bbox_j ≠ ∅}| + |{i: bbox_i ∩ arcs ≠ ∅}|` using real font metric bboxes (MATH_SPEC §14.1).
- **Target** 0 · **Min** 0 · **Max** 0.
- **Measurement.** Spatial-hash sweep over LabelPlan + LegendPlan geometry against CurveSet.
- **Benchmark.** `benchmarks/quality/labels.py`; per-run validator.

### QM-21 Label Coverage — **Gate**

- **Definition.** Fraction of faces with a rendered number (in-region or leader).
- **Formula.** `L_cov = n_labeled / F × 100 %`.
- **Target** 100 % · **Min** 100 % · **Max** 100 %.
- **Measurement.** LabelPlan census vs ArcGraph face census (exterior excluded).
- **Benchmark.** per-run validator; `benchmarks/quality/labels.py`.

### QM-22 In-Region Label Fit Rate — Monitor

- **Definition.** Fraction of labels placed in-region at unshrunk computed size (no shrink-to-fit, no leader).
- **Formula.** `L_fit = n_{S = clip(S_fit)} at first attempt / F × 100 %`.
- **Target** ≥ 93 % · **Min acceptable** 90 % · **Max** n/a.
- **Measurement.** LabelPlan provenance flags.
- **Benchmark.** `benchmarks/quality/labels.py`; tolerance −2 points.

### QM-23 Maximum Label Rotation — **Gate**

- **Definition.** Angular deviation of any rendered number from page-upright.
- **Formula.** `θ_label_max = max_i |rot_i|`.
- **Target** 0° · **Min** 0° · **Max** 0° (rotation is not in the layout model; the gate asserts no transform sneaks in via rendering).
- **Measurement.** SVG parse: no `rotate`/`matrix` transforms permitted on `<text>` nodes; PDF text matrix must be axis-aligned.
- **Benchmark.** `benchmarks/quality/output_validity.py`.

### QM-24 Minimum Font Size — **Gate**

- **Definition.** Smallest rendered number size.
- **Formula.** `S_min_obs = min_i S_i` in pt.
- **Target** ≥ 6.0 pt · **Min** 6.0 pt (= `quality.font_min_pt`) · **Max** n/a (per-label cap 14 pt enforced separately as config).
- **Measurement.** LabelPlan sweep + SVG attribute parse (double-entry).
- **Benchmark.** per-run validator; `benchmarks/quality/labels.py`.

### QM-25 Leader Crossing Count — **Gate**

- **Definition.** Arcs crossed by any single leader line.
- **Formula.** `X_leader = max_leaders |{arcs crossed}|` (MATH_SPEC §14.2).
- **Target** ≤ 1 · **Min** n/a · **Max** 2.
- **Measurement.** Segment-intersection sweep, leader vs flattened arcs.
- **Benchmark.** `benchmarks/quality/labels.py`; per-run validator.

---

## 6. Output validity metrics

### QM-26 SVG Validity — **Gate**

- **Definition.** Conformance of the SVG output to SVG 1.1 plus the engine's structural contract.
- **Formula.** `V_svg = n_xml_errors + n_schema_errors + n_contract_errors` where contract errors = missing/reordered layer groups, coordinate precision ≠ 3 decimals, arc drawn ≠ exactly once.
- **Target** 0 · **Min** 0 · **Max** 0.
- **Measurement.** XML well-formedness parse + RELAX NG validation against the SVG 1.1 schema + structural linter (layer order, `d`-string regex, per-arc uniqueness via `data-left/right` census).
- **Benchmark.** `benchmarks/quality/output_validity.py`.

### QM-27 Output Determinism — **Gate**

- **Definition.** Byte identity of SVG and PNG outputs across repeated runs and across supported platforms (I2).
- **Formula.** `D_out = ⟦SHA-256 run₁ = SHA-256 run₂ (per artifact)⟧`, over 2 same-machine runs + the recorded cross-platform CI matrix.
- **Target** 1 · **Min** 1 · **Max** 1. (PDF excluded — gated by QM-28 instead; MATH_SPEC/ENGINE_SPEC §23 rationale.)
- **Measurement.** Hash comparison in CI; ladder-wide.
- **Benchmark.** `benchmarks/quality/determinism.py`.

### QM-28 PDF Printability — **Gate**

- **Definition.** Print-readiness of the PDF: exact trim box, embedded subset fonts, vector-only content, geometric agreement with SVG.
- **Formula.** `P_pdf = ⟦trimbox = page ± 0.05 pt⟧ · ⟦all fonts embedded⟧ · ⟦n_images = 0⟧ · ⟦Δ_geo ≤ 0.05 pt⟧` where `Δ_geo` = max deviation over 1 000 deterministic sample points per fixture, PDF path space vs SVG path space.
- **Target** 1 · **Min** 1 · **Max** 1.
- **Measurement.** PDF object parse (boxes, font descriptors, XObject census) + renderer-agreement contract test.
- **Benchmark.** `benchmarks/quality/output_validity.py`.

### QM-29 Preview Fidelity to Vector — Monitor

- **Definition.** Agreement of the line-art PNG with the SVG rendered by an independent rasterizer.
- **Formula.** SSIM(engine PNG, reference-rasterized SVG) at 150 DPI, luminance.
- **Target** ≥ 0.97 · **Min acceptable** 0.95 · **Max** n/a.
- **Measurement.** CI-pinned reference rasterizer (version-locked resvg binary in the CI image; not a runtime dependency).
- **Benchmark.** `benchmarks/quality/output_validity.py`; tolerance −0.01.

---

## 7. Performance and resource metrics

### QM-30 Processing Time (End-to-End) — **Gate**

- **Definition.** Wall time from `convert()` entry to OutputBundle return, single core, reference machine.
- **Formula.** `T_e2e` seconds; per-stage budgets per ENGINE_SPEC §26.
- **Target** ≤ 12 s (`F-photo-2`) · **Min** n/a · **Max acceptable** 15 s. Ladder scaling gates: `F-photo-05` ≤ 6 s, `F-photo-12` ≤ 35 s, `F-photo-24` ≤ 60 s.
- **Measurement.** Tracer timings (monotonic clock), 3 runs, median reported.
- **Benchmark.** `benchmarks/perf/e2e.py`; per-stage regression tolerance +20 % vs baseline (ARCHITECTURE.md §9).

### QM-31 Peak Memory Usage — **Gate**

- **Definition.** Peak resident set size during a conversion.
- **Formula.** `M_peak` MiB (max RSS delta from process baseline).
- **Target** ≤ 600 MiB (`F-photo-2`) · **Min** n/a · **Max acceptable** 900 MiB; `F-photo-24` ≤ 2 500 MiB.
- **Measurement.** `resource.getrusage` high-water mark sampled per stage by the tracer; container-measured in CI.
- **Benchmark.** `benchmarks/perf/memory.py`; tolerance +15 % vs baseline.

### QM-32 Output File Size — Monitor

- **Definition.** Byte size of SVG and PDF deliverables.
- **Formula.** `|svg|`, `|pdf|` bytes.
- **Target** SVG ≤ 1.5 MiB, PDF ≤ 1.0 MiB (`F-photo-2`) · **Min** n/a · **Max acceptable** SVG 3 MiB, PDF 2 MiB.
- **Measurement.** Bundle byte census.
- **Benchmark.** `benchmarks/perf/size.py`; tolerance +15 %.

### QM-33 Determinism Cost — Monitor

- **Definition.** Runtime overhead of tracing/logging being enabled (must be immaterial; I2 requires *output* invariance, this bounds *speed* impact).
- **Formula.** `T_traced / T_untraced`.
- **Target** ≤ 1.03 · **Min** n/a · **Max acceptable** 1.08.
- **Measurement.** Paired e2e runs, median of 3.
- **Benchmark.** `benchmarks/perf/overhead.py`.

---

## 8. Release gate summary

A release candidate passes iff, on every fixture × preset of the ladder:

| Class | Metrics | Criterion |
|---|---|---|
| Gate | QM-01..04, 09, 10, 11, 15..21, 23..28, 30, 31 | value within band, zero exceptions |
| Monitor | QM-05..08, 12..14, 22, 29, 32, 33 | within tolerance vs committed baseline |

The harness emits `benchmarks/reports/<run-id>.json` containing every QM value, the resolved config hash, engine version, input hashes, and machine fingerprint — sufficient to reproduce any historical number (golden-ledger rule, ARCHITECTURE.md §9). Baseline updates require a reviewed commit that includes the before/after report diff.

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-06 | Initial complete quality specification (33 metrics). |

