# Module Design — Color Analysis (`stages/raster/analyze`)

**Status:** v1.0 — implemented. Governing spec: [ENGINE_SPEC.md §6](../ENGINE_SPEC.md); formulas in [MATH_SPEC.md](../MATH_SPEC.md) §3–§4, §16.

## Purpose

Measure global statistics of the working raster and translate them into *advisory* config overrides. The auto-tune layer may only fill values the user left unset (ARCHITECTURE.md §7); this stage never overrides human intent.

## Statistics (all closed-form, O(N), deterministic)

| Statistic | Definition | Source |
|---|---|---|
| `colorfulness` | Hasler–Süsstrunk `M = σ_rgyb + 0.3·μ_rgyb` on 0–255 scale | `foundation/color` (single implementation rule) |
| `edge_density` | fraction of pixels with Sobel(∇L*/100) magnitude > 0.1 | ENGINE_SPEC §6.2 |
| `luminance_histogram` | 64 uniform bins over L* ∈ [0, 100], normalized | §6.3 |
| `entropy_bits` | Shannon entropy of the histogram (≤ 6 bits) | §6.3 |
| `lab_mean / lab_std` | per-channel LAB moments | this design |
| `brightness / contrast` | mean / std of L* | this design |
| `saturation` | mean chroma C* = hypot(a*, b*) | this design |

## Proposals (§6.4–6.5)

- `quantize.n_colors ← k* = clip(round(6 + 0.12·M + 6·ρ + 0.8·H_L), k_min=8, k_max=30)`
- `preprocess.smooth_passes ← 3` if ρ > 0.25; `← 1` if ρ < 0.05; otherwise **no proposal** (absence, not a default value — the fill-only merge means a proposal that equals the default would still shadow a later preset).

## Quality requirements

- Determinism: identical stats across runs (no RNG anywhere).
- Invariance: `k*` identical under 90° rotation and mirroring (all inputs to the formula are orientation-invariant; Sobel magnitude is symmetric under the dihedral group) — property-tested.
- Budget: ≤ 0.1 s at 1600 px (ENGINE_SPEC §26).

## Artifacts

Provides `image_stats` (`ImageStats`, DATA_MODEL addendum in `model/records.py`) and `auto_tune` (`AutoTuneProposal`, a frozen config fragment consumed by the orchestrator's config resolution — ConfigLayer.AUTO_TUNE).

## Future

The learned content-aware advisor (ARCHITECTURE.md §14.1) replaces `propose_overrides` behind the same analyzer extension point; `compute_stats` remains as its deterministic fallback.
