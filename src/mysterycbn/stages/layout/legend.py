"""Legend stage: palette-order permutation + chip band layout
(ENGINE_SPEC.md §20-21, ARCHITECTURE.md §15 "Layout stages"; Sprint 19
orchestration gap).

No prior implementation existed for this responsibility (confirmed absent:
``find src -iname "*legend*" -o -iname "*palette_order*"`` returned nothing
before this file, per the Sprint 18 architecture audit). This module is new
code, not a redesign of any existing stage -- ``render_svg``/``render_pdf``
already declare ``legend`` as a required input artifact
(``SvgExportStage.requires``/``PdfExportStage.requires`` both include
``"legend"``) but nothing ever produced it.

Default policy (identity permutation, single-row chip band) is the simplest
contract-satisfying implementation: printed numbers equal palette index + 1,
in coverage-descending order (the order ``Palette.colors`` is already sorted
in by the quantize stage, DATA_MODEL_SPEC §4's "coverage-descending renumber"
rule) -- i.e. no numbering-obfuscation ("mystery") shuffle is applied. A
future ``palette_order`` advisor (ENGINE_SPEC §20's Spearman-constrained
shuffle) can replace ``_default_permutation`` without touching this stage's
contract.
"""

from __future__ import annotations

from collections.abc import Mapping

from mysterycbn.foundation.errors import ConfigError
from mysterycbn.model.context import PipelineContext
from mysterycbn.model.layout import Legend
from mysterycbn.model.records import Palette, Provenance

STAGE_NAME = "legend"
STAGE_VERSION = "1.0.0"
_UNSET_HASH = "0" * 64

CHIP_MM_DEFAULT = 6.0
GAP_MM_DEFAULT = 2.0
NUMBER_FONT_PT_DEFAULT = 8.0
_MM_TO_PT = 72.0 / 25.4


def build_legend(
    palette: Palette,
    *,
    band_origin_pt: tuple[float, float] = (0.0, 0.0),
    band_width_pt: float,
    chip_mm: float = CHIP_MM_DEFAULT,
    gap_mm: float = GAP_MM_DEFAULT,
    number_font_pt: float = NUMBER_FONT_PT_DEFAULT,
    config_hash: str = _UNSET_HASH,
) -> Legend:
    """Lay out one chip per palette entry, wrapping rows to fit
    ``band_width_pt``; identity permutation (printed_number = index + 1)."""
    if palette.size < 2:
        raise ConfigError(f"legend requires a palette of >= 2 colors, got {palette.size}")
    chip_pt = chip_mm * _MM_TO_PT
    gap_pt = gap_mm * _MM_TO_PT
    if chip_pt <= 0.0 or gap_pt < 0.0:
        raise ConfigError(f"legend: chip_mm must be > 0 and gap_mm >= 0, got {chip_mm}/{gap_mm}")

    ox, oy = band_origin_pt
    per_row = max(1, int((band_width_pt + gap_pt) // (chip_pt + gap_pt)))
    n_rows = -(-palette.size // per_row)  # ceil division
    row_height = chip_pt + gap_pt

    chips: list[tuple[int, tuple[float, float], float]] = []
    for i in range(palette.size):
        row, col = divmod(i, per_row)
        cx = ox + col * (chip_pt + gap_pt)
        cy = oy + row * row_height
        chips.append((i, (cx, cy), chip_pt))

    band_height_pt = n_rows * row_height
    permutation = tuple(range(palette.size))  # identity: no mystery shuffle (see module docstring)

    return Legend(
        permutation=permutation,
        chips=tuple(chips),
        band_rect=(ox, oy, band_width_pt, band_height_pt),
        number_font_pt=number_font_pt,
        provenance=Provenance(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config_hash=config_hash,
            source_hash=palette.provenance.source_hash,
        ),
    )


class LegendStage:
    """Stage wrapper: ``palette`` -> ``legend``.

    Reads the ``page`` section for the content-box width (the legend band
    spans the printable width, mirroring ``ArcGraphStage``'s page geometry)
    -- this is the one config key this stage reads outside its own section,
    justified because legend placement is inherently a page-geometry concern.
    """

    def __init__(
        self,
        section: Mapping[str, object] | None = None,
        *,
        page_width_mm: float = 215.9,
        margin_mm: float = 12.7,
        config_hash: str = _UNSET_HASH,
    ) -> None:
        section = section or {}
        chip_mm = section.get("chip_mm", CHIP_MM_DEFAULT)
        gap_mm = section.get("gap_mm", GAP_MM_DEFAULT)
        font_pt = section.get("number_font_pt", NUMBER_FONT_PT_DEFAULT)
        if not isinstance(chip_mm, (int, float)) or not 1.0 <= float(chip_mm) <= 30.0:
            raise ConfigError(f"legend config: chip_mm must be in [1, 30], got {chip_mm!r}")
        if not isinstance(gap_mm, (int, float)) or not 0.0 <= float(gap_mm) <= 10.0:
            raise ConfigError(f"legend config: gap_mm must be in [0, 10], got {gap_mm!r}")
        if not isinstance(font_pt, (int, float)) or float(font_pt) <= 0.0:
            raise ConfigError(f"legend config: number_font_pt must be > 0, got {font_pt!r}")
        self._chip_mm = float(chip_mm)
        self._gap_mm = float(gap_mm)
        self._font_pt = float(font_pt)
        self._band_width_pt = (page_width_mm - 2 * margin_mm) * _MM_TO_PT
        self._margin_pt = margin_mm * _MM_TO_PT
        self._config_hash = config_hash

    @property
    def name(self) -> str:
        return STAGE_NAME

    @property
    def version(self) -> str:
        return STAGE_VERSION

    @property
    def requires(self) -> tuple[str, ...]:
        return ("palette",)

    @property
    def provides(self) -> tuple[str, ...]:
        return ("legend",)

    @property
    def config_section(self) -> str:
        return STAGE_NAME

    def run(self, ctx: PipelineContext) -> None:
        palette = ctx.get("palette")
        if not isinstance(palette, Palette):
            raise ConfigError("legend requires a Palette artifact")
        ctx.put(
            "legend",
            build_legend(
                palette,
                band_origin_pt=(self._margin_pt, self._margin_pt),
                band_width_pt=self._band_width_pt,
                chip_mm=self._chip_mm,
                gap_mm=self._gap_mm,
                number_font_pt=self._font_pt,
                config_hash=self._config_hash,
            ),
        )
