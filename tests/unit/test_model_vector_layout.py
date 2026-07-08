"""Unit tests for the vector- and layout-domain data model (DATA_MODEL_SPEC §9–§16)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mysterycbn.model.layout import Label, LabelMode, Legend
from mysterycbn.model.records import Provenance
from mysterycbn.model.vector import (
    Arc,
    ArcGraph,
    BezierSegment,
    Curve,
    CurveSet,
    Face,
    TopologyGraph,
)

PROV = Provenance("test", "1.0.0", "ab" * 32, "cd" * 32)
LINE = np.array([[0.0, 0.0], [1.0, 0.0]])


def _arc(arc_id: int = 0, left: int = 0, right: int = -1) -> Arc:
    return Arc(arc_id=arc_id, points=LINE, left_region=left, right_region=right)


def test_arc_validation() -> None:
    assert json.dumps(_arc().to_dict())
    with pytest.raises(ValueError, match="differ"):
        Arc(0, LINE, left_region=1, right_region=1)
    with pytest.raises(ValueError, match="distinct"):
        Arc(0, np.array([[0.0, 0.0], [0.0, 0.0]]), 0, 1)
    with pytest.raises(ValueError, match="≥ 4"):
        Arc(0, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), 0, 1, closed=True)


def test_topology_graph_endpoint_check() -> None:
    junctions = np.array([[0, 0], [1, 0]], dtype=np.int64)
    graph = TopologyGraph(
        junctions=junctions,
        arcs=(Arc(0, np.array([[0.0, 0.0], [1.0, 0.0]]), 0, -1),),
        provenance=PROV,
    )
    assert json.dumps(graph.to_dict())
    with pytest.raises(ValueError, match="not a junction"):
        TopologyGraph(
            junctions=np.array([[0, 0]], dtype=np.int64),
            arcs=(Arc(0, np.array([[0.0, 0.0], [5.0, 5.0]]), 0, -1),),
            provenance=PROV,
        )


def test_arc_graph_reference_counting() -> None:
    arcs = (_arc(0), _arc(1, left=1))
    faces = (
        Face(0, 0, outer_walk=((0, False),)),
        Face(1, 1, outer_walk=((1, False),)),
    )
    graph = ArcGraph(arcs=arcs, faces=faces, work_scale=0.4, provenance=PROV)
    assert json.dumps(graph.to_dict())
    with pytest.raises(ValueError, match="unknown arc"):
        ArcGraph(arcs, (Face(0, 0, ((7, False),)),), 0.4, PROV)
    over = (
        Face(0, 0, ((0, False),)),
        Face(1, 1, ((0, True), (0, False))),  # arc 0 referenced 3× total
    )
    with pytest.raises(ValueError, match="max 2"):
        ArcGraph(arcs, over, 0.4, PROV)
    with pytest.raises(ValueError, match="work_scale"):
        ArcGraph(arcs, faces, 0.0, PROV)


def _segment(x0: float = 0.0, x1: float = 3.0) -> BezierSegment:
    return BezierSegment(control=np.array([[x0, 0.0], [x0 + 1.0, 0.0], [x1 - 1.0, 0.0], [x1, 0.0]]))


def test_bezier_segment_validation() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        BezierSegment(control=np.array([[0.0, 0.0]] * 4))
    with pytest.raises(ValueError, match="finite"):
        BezierSegment(control=np.array([[0.0, np.nan], [1, 0], [2, 0], [3, 0]]))


def test_curve_chain_continuity() -> None:
    curve = Curve(
        arc_id=0,
        segments=(_segment(0.0, 3.0), _segment(3.0, 6.0)),
        corner_indices=(1,),
        max_fit_error_pt=0.1,
    )
    assert json.dumps(curve.to_dict())
    with pytest.raises(ValueError, match="share endpoints"):
        Curve(0, (_segment(0.0, 3.0), _segment(4.0, 7.0)), (), 0.1)
    with pytest.raises(ValueError, match="corner_indices"):
        Curve(0, (_segment(),), corner_indices=(1,), max_fit_error_pt=0.1)


def test_curve_set_density() -> None:
    curves = (Curve(0, (_segment(),), (), 0.0),)
    cs = CurveSet(curves=curves, faces=(Face(0, 0, ((0, False),)),), provenance=PROV)
    assert json.dumps(cs.to_dict())
    with pytest.raises(ValueError, match="dense"):
        CurveSet((Curve(1, (_segment(),), (), 0.0),), (), PROV)


def test_label_modes() -> None:
    ok = Label(0, 1, (10.0, 10.0), 8.0, LabelMode.IN_REGION, clearance_pt=6.0)
    assert json.dumps(ok.to_dict())
    with pytest.raises(ValueError, match="leader segment"):
        Label(0, 1, (10.0, 10.0), 8.0, LabelMode.LEADER, clearance_pt=1.0)
    with pytest.raises(ValueError, match="must not carry"):
        Label(
            0,
            1,
            (10.0, 10.0),
            8.0,
            LabelMode.IN_REGION,
            1.0,
            leader=((0.0, 0.0), (1.0, 1.0)),
        )
    with pytest.raises(ValueError, match="1-based"):
        Label(0, 0, (10.0, 10.0), 8.0, LabelMode.IN_REGION, 1.0)


def test_legend_bijection_and_band_containment() -> None:
    legend = Legend(
        permutation=(1, 0),
        chips=((0, (0.0, 0.0), 10.0), (1, (12.0, 0.0), 10.0)),
        band_rect=(0.0, 0.0, 30.0, 12.0),
        number_font_pt=8.0,
        provenance=PROV,
    )
    assert legend.printed_number(0) == 2
    assert legend.printed_number(1) == 1
    assert json.dumps(legend.to_dict())
    with pytest.raises(ValueError, match="bijection"):
        Legend((0, 0), legend.chips, legend.band_rect, 8.0, PROV)
    with pytest.raises(ValueError, match="outside the legend band"):
        Legend(
            (1, 0),
            ((0, (0.0, 0.0), 10.0), (1, (25.0, 0.0), 10.0)),  # 25+10 > 30
            (0.0, 0.0, 30.0, 12.0),
            8.0,
            PROV,
        )
