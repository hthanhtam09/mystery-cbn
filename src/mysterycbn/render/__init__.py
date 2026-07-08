"""Rendering backends behind one Renderer interface (ARCHITECTURE.md §6).

SVG is canonical; PDF and PNG must agree geometrically (contract-tested).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mysterycbn.model.artifacts import CurveSet, LabelPlan, LegendPlan


@runtime_checkable
class PageConfig(Protocol):
    """Physical page geometry in points."""

    @property
    def width_pt(self) -> float: ...

    @property
    def height_pt(self) -> float: ...

    @property
    def margin_pt(self) -> float: ...


@runtime_checkable
class Renderer(Protocol):
    """Public plugin interface: plans in, bytes out."""

    @property
    def name(self) -> str: ...

    @property
    def media_type(self) -> str: ...

    def render(
        self,
        curves: CurveSet,
        labels: LabelPlan,
        legend: LegendPlan,
        page: PageConfig,
    ) -> bytes: ...
