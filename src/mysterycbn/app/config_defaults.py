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
    "regions",
    "merge_tiny",
    "organic_partition",
    "split_large",
    "topology",
    "arcgraph",
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
    # "dense" keeps the hard floor: raising it to 3.5 was tried and merges
    # away real subject detail (eyes/mouths on character art come out
    # mangled) — label room comes from the organic cells' seed_density_mm2
    # instead, and merge_tiny still cleans sub-floor noise at 2.5.
    "dense": 2.5,
}

# n_colors per preset (ENGINE_SPEC quantize defaults; medium matches
# QuantizeStage's own default of 16). "dense" uses ~17 to match the
# commercial color-by-number palette size.
N_COLORS_BY_PRESET: dict[str, int] = {
    "easy": 8,
    "medium": 16,
    "hard": 24,
    "dense": 17,
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
    if preset == "dense":
        # Commercial color-by-number look: ~17 colors and many small numbered
        # cells across the whole page. Lower the color-merge threshold so
        # near-similar palette colors stay distinct (guaranteeing ~17 colors).
        # font_min stays at the default: the moderate d_min floor above means
        # split cells remain large enough to print a normal number.
        overlay["quantize"] = {"n_colors": N_COLORS_BY_PRESET[preset], "merge_delta_e": 3.0}
        # Complex character art needs the extra working resolution: at the
        # 1600px default a busy page's thin, high-contrast features (eyes,
        # mouths, braids) land on too few pixels to survive quantize+denoise
        # with their shapes intact — the traced lines then read as "wrong".
        overlay["preprocess"] = {"max_working_px": 2000}
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
        overlay["organic"] = {
            "enabled": True,
            "mode": "streamline",
            "skip_background": False,
            "seed_density_mm2": 120.0,
            "min_area_mm2": 40.0,
            "warp_strength_mm": 6.0,
            "noise_scale_mm": 18.0,
            "ribbon_elongation": 0.7,
            "min_inner_diameter_mm": 3.2,
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
        overlay["simplify"] = {"tolerance_mm": 0.2}
        overlay["bezier"] = {"fit_error_mm": 0.22, "corner_angle_deg": 80.0}
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
    return overlay
