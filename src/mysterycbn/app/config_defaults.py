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
PIPELINE_STAGES: tuple[str, ...] = (
    "load",
    "preprocess",
    "analyze",
    "quantize",
    "denoise",
    "regions",
    "merge_tiny",
    "topology",
    "arcgraph",
    "simplify",
    "bezier",
    "labels",
    "legend",
    "svg",
    "pdf",
    "png",
)

# d_min_mm per preset (QUALITY_SPEC.md QM-10: "3.5 mm medium; 5.0 easy; 2.5 hard").
D_MIN_MM_BY_PRESET: dict[str, float] = {
    "easy": 5.0,
    "medium": 3.5,
    "hard": 2.5,
}

# n_colors per preset (ENGINE_SPEC quantize defaults; medium matches
# QuantizeStage's own default of 16).
N_COLORS_BY_PRESET: dict[str, int] = {
    "easy": 8,
    "medium": 16,
    "hard": 24,
}

DEFAULT_PAGE_MM: tuple[float, float, float] = (215.9, 279.4, 12.7)  # US Letter, matches
# every stage's own _DEFAULT_PAGE_MM constant (arcgraph.py, svg.py, pdf.py).


def builtin_defaults() -> dict[str, object]:
    """Layer 1: every stage's config section, defaulted to ``{}`` (each
    stage's own ``__init__``/``from_config`` already supplies working
    defaults for every key it reads) plus the ``pipeline.stages`` list."""
    sections: dict[str, object] = {name: {} for name in PIPELINE_STAGES}
    sections["pipeline"] = {"stages": list(PIPELINE_STAGES)}
    sections["page"] = {
        "width_mm": DEFAULT_PAGE_MM[0],
        "height_mm": DEFAULT_PAGE_MM[1],
        "margin_mm": DEFAULT_PAGE_MM[2],
    }
    sections["quality"] = {"d_min_mm": D_MIN_MM_BY_PRESET["medium"], "font_min_pt": 6.0}
    sections["quantize"] = {"n_colors": N_COLORS_BY_PRESET["medium"]}
    return sections


def difficulty_preset(preset: str) -> dict[str, object]:
    """Layer 2: the ``easy``/``medium``/``hard`` preset overlay."""
    if preset not in D_MIN_MM_BY_PRESET:
        raise ConfigError(f"unknown preset {preset!r}; choose from {sorted(D_MIN_MM_BY_PRESET)}")
    return {
        "quality": {"d_min_mm": D_MIN_MM_BY_PRESET[preset]},
        "quantize": {"n_colors": N_COLORS_BY_PRESET[preset]},
    }
