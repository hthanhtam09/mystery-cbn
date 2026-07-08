"""Engine configuration.

A single frozen pydantic model tree. Difficulty presets provide sensible
defaults; any explicit field the caller sets wins over preset and over
auto-analysis suggestions.

All physical dimensions are expressed in millimetres and converted to
points/pixels by the stages that need them, so page geometry stays
printer-oriented.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigError

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PageConfig(_Frozen):
    """Output page geometry. Defaults: US Letter with 12.7 mm margins."""

    width_mm: float = Field(default=215.9, gt=0)
    height_mm: float = Field(default=279.4, gt=0)
    margin_mm: float = Field(default=12.7, ge=0)
    dpi: int = Field(default=300, ge=72, le=1200)

    @property
    def content_width_mm(self) -> float:
        return self.width_mm - 2 * self.margin_mm

    @property
    def content_height_mm(self) -> float:
        return self.height_mm - 2 * self.margin_mm

    @property
    def width_pt(self) -> float:
        return self.width_mm / MM_PER_INCH * PT_PER_INCH

    @property
    def height_pt(self) -> float:
        return self.height_mm / MM_PER_INCH * PT_PER_INCH

    @model_validator(mode="after")
    def _check_content_area(self) -> PageConfig:
        if self.content_width_mm <= 0 or self.content_height_mm <= 0:
            raise ConfigError("margins leave no printable content area")
        return self


class PreprocessConfig(_Frozen):
    """Working-resolution and edge-preserving smoothing parameters."""

    # Longest side of the working raster. Quality/speed trade-off knob:
    # boundaries are traced at this resolution, then scaled to page space.
    max_working_px: int = Field(default=1600, ge=256, le=6000)
    # Edge-preserving smoothing: number of bilateral passes and strength.
    smooth_passes: int = Field(default=2, ge=0, le=5)
    bilateral_sigma_color: float = Field(default=0.08, gt=0)  # in [0,1] RGB units
    bilateral_sigma_space: float = Field(default=5.0, gt=0)  # pixels
    clahe: bool = False
    clahe_clip: float = Field(default=2.0, gt=0)


class QuantizeConfig(_Frozen):
    """Perceptual color quantization in CIELAB."""

    n_colors: int = Field(default=18, ge=2, le=64)
    seed: int = 1337
    # k-means detail
    max_iter: int = Field(default=50, ge=5)
    attempts: int = Field(default=3, ge=1)
    # Chroma boost >1 makes hue differences count more than lightness,
    # which yields palettes that read better as distinct pencil colors.
    chroma_weight: float = Field(default=1.0, gt=0, le=3.0)
    # Merge palette entries closer than this ΔE76 after clustering.
    min_delta_e: float = Field(default=6.0, ge=0)


class RegionConfig(_Frozen):
    """Region size constraints, expressed physically."""

    # Minimum colorable feature: inscribed-circle diameter on the page.
    min_region_mm: float = Field(default=3.0, gt=0)
    # Regions smaller than this fraction of min area are merged silently;
    # larger-but-still-small regions may keep a leader-lined label instead.
    denoise_area_frac: float = Field(default=0.35, gt=0, le=1)


class CurveConfig(_Frozen):
    """Simplification and Bézier smoothing."""

    # Visvalingam–Whyatt effective area threshold in output millimetres².
    simplify_mm2: float = Field(default=0.04, ge=0)
    # Max Bézier fitting error in output millimetres.
    fit_error_mm: float = Field(default=0.15, gt=0)
    # Corner detection: angles sharper than this (degrees) stay sharp.
    corner_angle_deg: float = Field(default=65.0, gt=0, lt=180)
    stroke_width_pt: float = Field(default=0.9, gt=0)


class LabelConfig(_Frozen):
    min_font_pt: float = Field(default=5.5, gt=0)
    max_font_pt: float = Field(default=11.0, gt=0)
    font_family: str = "Helvetica"


class EngineConfig(_Frozen):
    """Root configuration for one conversion run."""

    difficulty: Difficulty = Difficulty.MEDIUM
    page: PageConfig = PageConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    quantize: QuantizeConfig = QuantizeConfig()
    regions: RegionConfig = RegionConfig()
    curves: CurveConfig = CurveConfig()
    labels: LabelConfig = LabelConfig()
    debug: bool = False

    @classmethod
    def preset(cls, difficulty: Difficulty | str, **overrides: object) -> EngineConfig:
        """Build a config from a difficulty preset plus explicit overrides.

        Overrides are nested dicts matching the model tree, e.g.
        ``EngineConfig.preset("easy", quantize={"n_colors": 10})``.
        """
        difficulty = Difficulty(difficulty)
        presets: dict[Difficulty, dict[str, dict[str, object]]] = {
            Difficulty.EASY: {
                "quantize": {"n_colors": 10, "min_delta_e": 10.0},
                "regions": {"min_region_mm": 5.0},
            },
            Difficulty.MEDIUM: {
                "quantize": {"n_colors": 18},
                "regions": {"min_region_mm": 3.0},
            },
            Difficulty.HARD: {
                "quantize": {"n_colors": 27, "min_delta_e": 4.0},
                "regions": {"min_region_mm": 2.2},
            },
        }
        base: dict[str, object] = {"difficulty": difficulty}
        for section, values in presets[difficulty].items():
            base[section] = dict(values)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **value}  # type: ignore[dict-item]
            else:
                base[key] = value
        return cls.model_validate(base)
