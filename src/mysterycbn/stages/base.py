"""Abstract per-domain stage bases: shared identity plumbing, domain-specific run contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from mysterycbn.model.context import PipelineContext


class StageBase(ABC):
    """Common base satisfying the kernel Stage protocol structurally."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def requires(self) -> Sequence[str]: ...

    @property
    @abstractmethod
    def provides(self) -> Sequence[str]: ...

    @property
    @abstractmethod
    def config_section(self) -> str: ...

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None: ...


class RasterStage(StageBase):
    """Operates on dense arrays at working resolution; may not touch graph or vector artifacts."""


class GraphStage(StageBase):
    """Operates on RegionGraph/LabelMap; the label raster remains authoritative geometry."""


class VectorStage(StageBase):
    """Operates on ArcGraph/CurveSet in physical units; may never consult the raster."""


class LayoutStage(StageBase):
    """Produces LabelPlan/LegendPlan from vector-domain artifacts."""
