"""Benchmarks for the SVG renderer (budget: ≤ 0.3 s for 12 000 segments,
output ≤ 2 MB typical, ENGINE_SPEC §22/§26)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mysterycbn.model.layout import Label, LabelMode, LabelPlan, Legend
from mysterycbn.model.records import Palette, PaletteColor, Provenance
from mysterycbn.model.vector import BezierSegment, Curve, CurveSet, Face
from mysterycbn.render.svg import render_svg, validate_svg

RNG = np.random.default_rng(0)
PROV = Provenance("bezier", "1.0.0", "0" * 64, "1" * 64)

# Synthetic CurveSet at contract scale: 1 500 arcs × 8 segments = 12 000
# segments, one face per arc referencing it once.
_N_ARCS, _SEGS = 1500, 8
_curves = []
_faces = []
for arc_id in range(_N_ARCS):
    x0, y0 = 40.0 + (arc_id % 40) * 12.0, 40.0 + (arc_id // 40) * 16.0
    pts = RNG.random((_SEGS, 4, 2)) * 10.0 + np.array([x0, y0])
    segments = []
    prev = pts[0][0]
    for s in range(_SEGS):
        ctrl = np.vstack([prev[None, :], pts[s][1:]])
        segments.append(BezierSegment(control=ctrl))
        prev = ctrl[3]
    _curves.append(
        Curve(arc_id=arc_id, segments=tuple(segments), corner_indices=(), max_fit_error_pt=0.1)
    )
    _faces.append(Face(face_id=arc_id, label=arc_id % 16, outer_walk=((arc_id, False),)))

_CURVE_SET = CurveSet(curves=tuple(_curves), faces=tuple(_faces), provenance=PROV)
_PLAN = LabelPlan(
    labels=tuple(
        Label(
            region_id=i,
            printed_number=(i % 16) + 1,
            anchor=(45.0 + (i % 40) * 12.0, 45.0 + (i // 40) * 16.0),
            font_size_pt=6.0,
            mode=LabelMode.IN_REGION,
            clearance_pt=5.0,
        )
        for i in range(_N_ARCS)
    ),
    provenance=PROV,
)
_PALETTE = Palette(
    colors=tuple(PaletteColor.from_lab(i, (5.0 + 6.0 * i, 0.0, 0.0), 1000) for i in range(16)),
    provenance=PROV,
)
_LEGEND = Legend(
    permutation=tuple(range(16)),
    chips=tuple((i, (40.0 + 33.0 * i, 745.0), 15.0) for i in range(16)),
    band_rect=(36.0, 740.0, 545.0, 25.0),
    number_font_pt=8.0,
    provenance=PROV,
)


def test_bench_svg_12k_segments(benchmark: Any) -> None:
    data = benchmark(render_svg, _CURVE_SET, _PLAN, _LEGEND, _PALETTE)
    validate_svg(data, _CURVE_SET)
    assert len(data) <= 2 * 1024 * 1024  # ≤ 2 MB typical
