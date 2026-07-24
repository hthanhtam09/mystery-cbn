"""Built-in default config + difficulty presets (ARCHITECTURE.md §7's
five-layer resolution, layers 1-2: BUILTIN_DEFAULTS, DIFFICULTY_PRESET).

Every stage's ``config_section`` name must have a corresponding key here
(``FrozenConfig.stage_section`` raises ``ConfigError`` on a missing
section) -- this module is the single place that enumerates the full
pipeline slot list and its per-stage defaults, so ``Orchestrator`` never
duplicates that list.
"""

from __future__ import annotations

from mysterycbn.foundation.errors import ConfigError

# Order matters: this is both the ``pipeline.stages`` list DefaultPlanResolver
# consumes and the exact Sprint 19 declared stage order. "Contour Extraction"
# is two existing stages (topology + arcgraph); "Curve Smoothing" is the
# existing bezier-fit stage -- see registry_bootstrap.py's module docstring.
#
# Sprint 36A.5: "geometry_normalize" is inserted between "simplify" and
# "bezier" -- the frozen ArcGraph -> ArcGraph normalization stage (duplicate
# cleanup / spike removal / minimum gap enforcement passes; see
# docs/modules/geometry_normalize.md, docs/modules/GAP_REPAIR_DESIGN.md).
# Its three passes are still identity placeholders (Sprint 36A.4); no
# geometry algorithm is implemented by this insertion.
#
# ADR-003: "organic_partition" is inserted between "merge_tiny" and
# "split_large" -- subdivides eligible regions into organic, spline-friendly
# cells (flowing boundaries, ribbon-like cells, nested islands) instead of
# straight/warped-Voronoi cells; see stages/graph/organic_partition.py and
# docs/adr/003-organic-region-partition.md. Disabled ("organic.enabled" =
# False) in every built-in preset -- opt-in only, so no existing golden
# fixture output changes.
PIPELINE_STAGES: tuple[str, ...] = (
    "load",
    "preprocess",
    "analyze",
    "quantize",
    "denoise",
    # "fill_holes": absorb small fully-enclosed label islands (e.g. a pupil's
    # white catchlight trapped inside the dark pupil) into their surrounding
    # region, so features don't render as broken donuts. Component-level, before
    # region building. Disabled outside dense/partial. See
    # stages/raster/fill_holes.py.
    "fill_holes",
    # "ink_detect": recover thin dark line work (whiskers, fine line-art) that
    # quantization maps into the surrounding fill. Emits a render-only overlay
    # (never regions/palette), so validators are blind to it. Disabled
    # ("ink.enabled" = False) in every preset except dense/partial -- no
    # existing golden output changes. See stages/raster/ink_detect.py.
    "ink_detect",
    "regions",
    "merge_tiny",
    # "mask" (the "partial" preset): marks the largest merge-compacted regions
    # "no_color" -- they keep their outline but bear no number and claim no
    # legend color. Placed right after "merge_tiny" (selects over clean,
    # floor-legal regions + the compacted palette) and before
    # "organic_partition" (which threads the no_color set through per-pixel,
    # like it already threads filler/rim). Disabled ("mask.enabled" = False)
    # in every preset except "partial", so no existing golden output changes.
    # See stages/graph/mask.py.
    "mask",
    "organic_partition",
    "split_large",
    "topology",
    "arcgraph",
    # "ink_overlay": vectorize the ink_mask into black centerline polylines in
    # page points (needs arcgraph's Φ). Render-only; see
    # stages/vector/ink_overlay.py.
    "ink_overlay",
    "simplify",
    "geometry_normalize",
    "bezier",
    "labels",
    "legend",
    "svg",
    "pdf",
    "png",
)

# d_min_mm per preset (QUALITY_SPEC.md QM-10: "3.5 mm medium; 5.0 easy; 2.5 hard").
# "dense" (Sprint: commercial CBN look) drops the floor far below the standard
# presets so the page keeps many small numbered cells instead of merging them
# away -- the printability gate scales with this same value, so small cells
# remain legal rather than FATAL.
D_MIN_MM_BY_PRESET: dict[str, float] = {
    "easy": 5.0,
    "medium": 3.5,
    "hard": 2.5,
    # "dense" uses a low floor to preserve small semantic detail. Raising it
    # to 3.5 was tried and merges away real subject detail (eyes/mouths on
    # character art come out mangled); even the earlier 2.5 floor still folded
    # the small *background* animals' eyes/mouths (few-mm dark dots) into the
    # surrounding fur, so the trace stopped matching the source. 1.5 keeps
    # those features as their own regions while merge_tiny still cleans genuine
    # sub-floor noise; label room comes from the organic cells' seed_density_mm2
    # (filler micro-labels, not this floor), and the "colored" preview now
    # fills sub-floor slivers with their solution color (see render/png.py) so
    # the lower floor doesn't paint white streaks.
    "dense": 1.5,
    # "partial" reuses "dense"'s tuning wholesale (it IS dense, minus the
    # numbers on the largest regions) -- keep the same floor.
    "partial": 1.5,
}

# n_colors per preset (ENGINE_SPEC quantize defaults; medium matches
# QuantizeStage's own default of 16). "dense" uses ~17 to match the
# commercial color-by-number palette size.
N_COLORS_BY_PRESET: dict[str, int] = {
    "easy": 8,
    "medium": 16,
    "hard": 24,
    "dense": 17,
    # "partial" mirrors "dense"'s palette size; the mask stage may later drop
    # any color used *only* by no_color regions from the legend, so the
    # printed count can end up a little lower than this.
    "partial": 17,
}

DEFAULT_PAGE_MM: tuple[float, float, float] = (215.9, 279.4, 12.7)  # US Letter, matches
# every stage's own _DEFAULT_PAGE_MM constant (arcgraph.py, svg.py, pdf.py).


def builtin_defaults() -> dict[str, object]:
    """Layer 1: every stage's config section, defaulted to ``{}`` (each
    stage's own ``__init__``/``from_config`` already supplies working
    defaults for every key it reads) plus the ``pipeline.stages`` list.
    """
    sections: dict[str, object] = {name: {} for name in PIPELINE_STAGES}
    # Three graph stages read a config section whose name differs from their
    # pipeline slot (MergeTinyStage.config_section == "merge",
    # SplitLargeStage.config_section == "split",
    # OrganicPartitionStage.config_section == "organic"); declare those so
    # they resolve.
    sections["merge"] = {}
    sections["split"] = {}
    sections["organic"] = {}
    # "mask" (NoColorMaskStage) reads its own section by pipeline-slot name;
    # disabled by default so every preset except "partial" is unaffected.
    sections["mask"] = {"enabled": False, "bitmap": None, "top_area_percentile": 0.5}
    # "ink" section is shared by both ink stages (ink_detect + ink_overlay);
    # disabled by default so only dense/partial (which enable it) are affected.
    sections["ink"] = {"enabled": False}
    sections["fill_holes"] = {"enabled": False}
    sections["validate"] = {}
    sections["pipeline"] = {"stages": list(PIPELINE_STAGES)}
    sections["page"] = {
        "width_mm": DEFAULT_PAGE_MM[0],
        "height_mm": DEFAULT_PAGE_MM[1],
        "margin_mm": DEFAULT_PAGE_MM[2],
    }
    sections["quality"] = {"d_min_mm": D_MIN_MM_BY_PRESET["medium"], "font_min_pt": 5.0}
    sections["quantize"] = {"n_colors": N_COLORS_BY_PRESET["medium"]}
    return sections


def difficulty_preset(preset: str) -> dict[str, object]:
    """Layer 2: the ``easy``/``medium``/``hard``/``dense`` preset overlay."""
    if preset not in D_MIN_MM_BY_PRESET:
        raise ConfigError(f"unknown preset {preset!r}; choose from {sorted(D_MIN_MM_BY_PRESET)}")
    overlay: dict[str, object] = {
        "quality": {"d_min_mm": D_MIN_MM_BY_PRESET[preset]},
        "quantize": {"n_colors": N_COLORS_BY_PRESET[preset]},
    }
    if preset in ("dense", "partial"):
        # Commercial color-by-number look: ~17 colors and many small numbered
        # cells across the whole page. Lower the color-merge threshold so
        # near-similar palette colors stay distinct (guaranteeing ~17 colors).
        # font_min stays at the default: the moderate d_min floor above means
        # split cells remain large enough to print a normal number.
        overlay["quantize"] = {"n_colors": N_COLORS_BY_PRESET[preset], "merge_delta_e": 3.0}
        # merge_tiny protects semantic dark dots from being merged away: a
        # sub-floor region whose palette L* < protect_dark_l AND every neighbour
        # is >= protect_dark_delta_l lighter (a dark pupil/nostril on a light
        # surround) is kept, so eyes keep symmetric pupils instead of one side
        # being folded into the sclera. The ink layer outlines them; only a
        # surviving region carries the dark fill. Safe under the compact-area
        # fidelity tolerance (they trace as compact discs).
        overlay["merge"] = {"protect_dark_l": 48.0, "protect_dark_delta_l": 16.0}
        # Fill small enclosed label islands (e.g. a catchlight trapped inside a
        # pupil) so eyes render as solid discs, not broken rings.
        overlay["fill_holes"] = {"enabled": True, "max_hole_mm2": 1.5}
        # merge_tiny stays ENABLED: it gives clean, flat, well-numbered color
        # cells (the "color by number" layer). Small semantic dark features
        # (eyes, pupils, whiskers, mouth/nose line work) are NOT preserved as
        # color regions here -- their shapes are still formed by the numbered
        # color cells once the page is coloured in.
        #
        # Ink-line layer: DISABLED. Its purpose was to re-trace the artwork's
        # bold outlines (eye/nose/mouth/whisker/mane line work) on top of the
        # color cells to mimic a commercial cartoon CBN look. But every ink
        # stroke lands right next to a region boundary that already traces the
        # same edge, so the page shows a doubled outline that makes the hidden
        # subject clearly recognisable before it is coloured -- the opposite of
        # a "mystery" page. Leaving ink off keeps a single set of light-gray
        # region-division lines, which is what the mystery look requires. (The
        # detection knobs below are retained, commented, so the commercial
        # look can be restored deliberately by flipping "enabled" back to True.)
        overlay["ink"] = {
            "enabled": False,
            # "max_width_mm": 1.2,
            # "contrast_l": 10.0,
            # "darkness_l": 42.0,
            # "survived_l": 0.0,
            # "min_length_mm": 1.5,
            # "stroke_mm": 0.1058,
        }
        # Complex character art needs the extra working resolution: at the
        # 1600px default a busy page's thin, high-contrast features (eyes,
        # mouths, braids) land on too few pixels to survive quantize+denoise
        # with their shapes intact — the traced lines then read as "wrong".
        # 2400 gives the small *background* animals' eyes/mouths enough pixels
        # to survive quantize's stride sampling and denoise's modal vote (the
        # lowered d_min floor above only helps once the feature reaches the
        # graph stage as its own region).
        overlay["preprocess"] = {"max_working_px": 2400}
        # Tile the whole page — background included — with organic cells (the
        # commercial "mystery" look: no boring continents, subject interior
        # subdivided too). organic_partition is the sole subdivider here;
        # split_large stays off because running both doubles the outline
        # around the subject (split_large's independent rim_mm=2.0 wraps a
        # second rim next to the silhouette — see ADR-003). skip_background
        # is off so the flat backdrop gets cells; seed_density_mm2 is the
        # target cell AREA in mm² (~16mm-wide cells at 250), and the lowered
        # min_area_mm2 lets medium-sized subject regions subdivide as well.
        # Cells are filler-exempt from the readable-font floor (micro-labels
        # down to 2pt), so printability holds despite the density.
        # seed_density_mm2 is the target cell AREA in mm²: 400 gives ~20mm-wide
        # cells, comfortably wider than a printed number plus padding (the
        # earlier 250 produced cells too small to label). warp_strength_mm
        # bends each cell boundary and noise_scale_mm sets the wavelength of
        # that bend; kept modest relative to the ~20mm cell width (a bend
        # whose amplitude/wavelength approach the cell's own size can fold a
        # boundary back on itself, which the never-repaired topology gate
        # then FATALs on as a self-intersecting arc -- see corner_angle_deg
        # note below for the other half of that failure mode).
        # min_inner_diameter_mm is the WIDTH floor (largest inscribed disk):
        # a cell can clear the area floor yet still be a ribbon too narrow to
        # carry its printed number — commercial reference sheets have no such
        # slivers. 3.2mm leaves ~1.6mm clearance radius around the label.
        # Camouflage tuning (the commercial "mystery" look): smaller, more
        # numerous cells with pronounced flowing/ribbon boundaries crossing
        # subject and background alike, so the picture only emerges once
        # colored. seed_density_mm2 250 gives ~16mm cells (still comfortably
        # wider than a printed number); ribbon_elongation biases streamline
        # pockets toward thin branching ribbons that visually break up the
        # silhouette. warp/noise stay well below the cell size so a boundary
        # cannot fold onto itself (the topology self-intersection FATAL the
        # earlier tuning notes warn about).
        # warp_strength_mm/noise_scale_mm lowered from 6.0/18.0: at ~16mm
        # cells (seed_density_mm2 120), a bend whose amplitude approaches a
        # sixth of the cell width still folds some boundaries back on
        # themselves often enough in practice to trip the never-repaired
        # topology gate (self-intersecting arc) on a meaningful slice of
        # real-world images. 4.0/22.0 keeps the same flowing-ribbon look
        # (lower amplitude, longer wavelength -- gentler curvature) while
        # giving geometry_normalize/curves.py's repair passes more slack to
        # actually converge instead of hitting their fixpoint loop's cap.
        overlay["organic"] = {
            "enabled": True,
            "mode": "streamline",
            "skip_background": False,
            "seed_density_mm2": 120.0,
            "min_area_mm2": 40.0,
            "warp_strength_mm": 4.0,
            "noise_scale_mm": 22.0,
            "ribbon_elongation": 0.7,
            "min_inner_diameter_mm": 3.2,
            # Fold the cartoon's own bold outline stroke so the subject shows a
            # SINGLE region-division line, not a doubled one. A thin ink outline
            # anti-aliased against light fur/background does NOT quantize to
            # near-black -- it lands on a mid-gray (measured L*~36 on the arctic
            # fox), well above the default fold threshold of 15.0, so the stroke
            # survived as its own ring region whose two edges (ring<->subject and
            # ring<->background) both traced -> a doubled outline on the subject
            # only (background strokes, having no such ring, stayed single).
            #
            # 37.0 (not higher): the outline gray measured L*~36 and the blue
            # coat is L*~38, so 37 makes the outline foldable while leaving every
            # blue-coat region ineligible -- the coat is protected outright, not
            # merely by the inradius gate. (Genuine mid-tone browns ~L*45 also
            # stay.)
            "skip_dark_lab_l_threshold": 37.0,
            # Raising the threshold alone did NOT fix it: the outline is ONE
            # connected region winding across the whole subject, and at its
            # thickest junction its inscribed-disk radius reaches ~2mm, above the
            # default 1.5mm "thin-only" fold gate -- so the entire outline web was
            # misclassified as a solid dark mass and spared. 2.5mm folds that
            # ~2mm web while still sparing genuine solid dark masses (the
            # near-black boots measured ~2.9mm+ inradius, so they keep their
            # black). See ADR-003 "pre-drawn cartoon outline" section.
            "dark_fold_max_inradius_mm": 2.5,
        }
        overlay["split"] = {"enabled": False}
        # Rounder line work: a higher corner threshold keeps only genuinely
        # sharp reversals as hard corners (everything else fits as one smooth
        # spline), and the looser simplify/fit tolerances let the bezier pass
        # relax the pixel staircase into curves instead of tracing it. Kept
        # well short of corner_angle_deg's 120° ceiling -- pushing it too
        # high stops registering real cusps as corners at all, and the
        # least-squares fitter then loops trying to smooth through an actual
        # reversal with too little error budget, self-intersecting (the same
        # topology FATAL the organic tuning above guards against).
        # corner_angle_deg lowered from 80.0: closer to the 120° ceiling the
        # least-squares fitter has less error budget to smooth through an
        # actual reversal without self-intersecting (see the topology-FATAL
        # note above) — 70.0 registers cusps as hard corners a bit more
        # readily, trading a slightly less rounded look at sharp reversals
        # for a lower self-intersection rate. tolerance_mm/fit_error_mm
        # unchanged: they aren't the primary driver per the tuning notes.
        overlay["simplify"] = {"tolerance_mm": 0.2}
        overlay["bezier"] = {"fit_error_mm": 0.22, "corner_angle_deg": 70.0}
        # The relaxed simplify/bezier tolerances above deliberately let the
        # fitted curves drift off the pixel-exact label boundaries (that
        # drift IS the rounded look), so the fidelity floors must budget for
        # it: agreement lands under the strict 0.99 default (which would
        # FATAL-abort the conversion for a purely cosmetic deviation). 0.93 /
        # 0.85 still catch genuine mis-registration — the corrupted-label
        # failure mode scores ~0.0 (validate/fidelity.py).
        overlay["validate"] = {
            "fidelity_min_agreement": 0.93,
            "fidelity_min_agreement_filler": 0.85,
        }
    if preset == "partial":
        # "partial" == the full "dense" treatment above, plus: mark the larger
        # half of the (merge-compacted) regions no_color -- they keep their
        # outline but carry no number and claim no legend color. Area-based
        # auto-detect is the chosen mask rule (no hand-drawn mask in the
        # engine); see stages/graph/mask.py. Everything else inherits "dense"
        # verbatim from the branch above (it IS dense, minus numbers on the
        # biggest regions).
        overlay["mask"] = {"enabled": True, "top_area_percentile": 0.5}
    return overlay
