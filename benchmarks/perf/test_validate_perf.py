"""Benchmarks for the Validation Engine (budget: full validation <= 2.0 s on
the 2 MP fixture, ENGINE_SPEC §25/§26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn import validate as V
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import place_labels
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph

RNG = np.random.default_rng(0)
PROV = Provenance("bezier", "1.0.0", "0" * 64, "1" * 64)
PAGE_MM = (215.9, 279.4, 12.7)

_K = 8
# Hues spread around a fixed-L, fixed-chroma wheel: well-separated (min
# pairwise DeltaE00 ~16.6, clearing the default 12.0 warn threshold), unlike
# a straight line through LAB which CIEDE2000 compresses at high lightness.
_PALETTE = Palette(
    colors=tuple(
        PaletteColor.from_lab(
            i, (55.0, 40.0 * np.cos(2 * np.pi * i / _K), 40.0 * np.sin(2 * np.pi * i / _K)), 1000
        )
        for i in range(_K)
    ),
    provenance=PROV,
)

# ~800 faces on a letter page: 30x27 blocks of _K distinct labels (same
# scale as the labels-stage benchmark, ENGINE_SPEC §26's face-count target).
_BASE = np.repeat(np.repeat(RNG.integers(0, _K, (27, 30)), 12, axis=0), 12, axis=1).astype(np.int32)
_RG = build_region_graph(LabelMap(labels=_BASE, provenance=PROV), _PALETTE)
_BOX = content_box_pt(PAGE_MM)
_TOPOLOGY = build_topology_graph(_RG.component_map)
_AG = build_arc_graph(_TOPOLOGY, _RG, content_box=_BOX)
_CS = fit_curves(_AG)
_PLAN, _FINDINGS = place_labels(_CS, _RG)
assert _FINDINGS == ()
assert 600 <= len(_CS.faces) <= 1000


def _fresh_ctx() -> InMemoryContext:
    ctx = InMemoryContext(seed=0)
    ctx.put("region_graph", _RG)
    ctx.put("arc_graph", _AG)
    ctx.put("curve_set", _CS)
    ctx.put("label_plan", _PLAN)
    ctx.put("palette", _PALETTE)
    return ctx


def test_bench_full_validation_gate_800_faces(benchmark: Any) -> None:
    def run() -> tuple[object, ...]:
        return V.run_validation(_fresh_ctx())

    reports = benchmark(run)
    assert all(r.passed for r in reports)


def test_bench_topology_validator_800_faces(benchmark: Any) -> None:
    ctx = _fresh_ctx()
    report = benchmark(V.validate_topology, ctx)
    assert report.passed


def test_bench_fidelity_validator_800_faces(benchmark: Any) -> None:
    ctx = _fresh_ctx()
    report = benchmark(V.validate_fidelity, ctx)
    assert report.passed


def test_bench_printability_validator_800_faces(benchmark: Any) -> None:
    def run() -> object:
        return V.validate_printability(_fresh_ctx())

    report = benchmark(run)
    assert report.passed


def test_bench_palette_validator_8_colors(benchmark: Any) -> None:
    ctx = _fresh_ctx()
    report = benchmark(V.validate_palette, ctx)
    assert report.passed
