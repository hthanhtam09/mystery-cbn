"""Unit tests for the SVG Export renderer (ENGINE_SPEC §22)."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mysterycbn.foundation.codes import code_for_number
from mysterycbn.foundation.errors import ConfigError, StageError
from mysterycbn.kernel.context import InMemoryContext
from mysterycbn.model.layout import Legend
from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.render.svg import (
    SvgDocument,
    SvgExportStage,
    format_coord,
    render_svg,
    validate_svg,
)
from mysterycbn.stages.graph.components import build_region_graph
from mysterycbn.stages.layout.labels import place_labels
from mysterycbn.stages.vector.arcgraph import build_arc_graph, content_box_pt
from mysterycbn.stages.vector.curves import fit_curves
from mysterycbn.stages.vector.topology import build_topology_graph

PROV = Provenance("labels", "1.0.0", "0" * 64, "1" * 64)
PAGE_MM = (215.9, 279.4, 12.7)
PAL4 = Palette(
    colors=tuple(PaletteColor.from_lab(i, (10.0 + 25.0 * i, 0.0, 0.0), 100) for i in range(4)),
    provenance=PROV,
)
NS = "{http://www.w3.org/2000/svg}"


def _fixture(rows: list[list[int]]):
    lm = LabelMap(labels=np.array(rows, dtype=np.int32), provenance=PROV)
    rg = build_region_graph(lm, PAL4)
    ag = build_arc_graph(
        build_topology_graph(rg.component_map), rg, content_box=content_box_pt(PAGE_MM)
    )
    curve_set = fit_curves(ag)
    plan, findings = place_labels(curve_set, rg)
    assert findings == ()
    legend = Legend(
        permutation=(0, 1, 2, 3),
        chips=tuple((i, (40.0 + 40.0 * i, 740.0), 17.0) for i in range(4)),
        band_rect=(36.0, 736.0, 540.0, 25.0),
        number_font_pt=8.0,
        provenance=PROV,
    )
    return curve_set, plan, legend


ROWS = [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]


def test_coordinate_formatting_including_negative_zero() -> None:
    assert format_coord(1.23456) == "1.235"
    assert format_coord(-0.0001) == "0.000"
    assert format_coord(-0.0) == "0.000"
    assert format_coord(-1.5) == "-1.500"
    assert format_coord(2.0, decimals=2) == "2.00"


def test_layer_order_and_validity() -> None:
    curve_set, plan, legend = _fixture(ROWS)
    data = render_svg(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)
    validate_svg(data, curve_set)  # raises on any structural violation
    root = ET.fromstring(data)
    assert [g.get("id") for g in root if g.tag == f"{NS}g"] == [
        "regions",
        "blackout",
        "ink",
        "labels",
        "leaders",
        "legend",
        "frame",
    ]
    assert root.get("width") == "215.900mm" and root.get("height") == "279.400mm"


def test_arc_once_property_and_data_sides() -> None:
    curve_set, plan, legend = _fixture(ROWS)
    data = render_svg(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)
    root = ET.fromstring(data)
    regions = root.find(f"{NS}g[@id='regions']")
    assert regions is not None
    paths = regions.findall(f"{NS}path")
    assert [p.get("id") for p in paths] == [f"arc-{c.arc_id}" for c in curve_set.curves]
    numbers = {str(lb.printed_number) for lb in plan.labels} | {"0"}
    for p in paths:
        assert p.get("data-left") in numbers and p.get("data-right") in numbers
    # Interior wall between regions carries two nonzero printed numbers.
    assert any(p.get("data-left") != "0" and p.get("data-right") != "0" for p in paths)


def test_labels_legend_and_frame_content() -> None:
    curve_set, plan, legend = _fixture(ROWS)
    data = render_svg(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)
    root = ET.fromstring(data)
    texts = root.find(f"{NS}g[@id='labels']").findall(f"{NS}text")  # type: ignore[union-attr]
    assert [t.text for t in texts] == [code_for_number(lb.printed_number) for lb in plan.labels]
    chips = root.find(f"{NS}g[@id='legend']").findall(f"{NS}rect")  # type: ignore[union-attr]
    assert len(chips) == 4
    # Chip fill equals the palette sRGB exactly.
    hex0 = chips[0].get("fill")
    r, g, b = (round(255 * v) for v in PAL4.colors[0].srgb)
    assert hex0 == f"#{r:02x}{g:02x}{b:02x}"
    frame = root.find(f"{NS}g[@id='frame']/{NS}rect")
    assert frame is not None
    margin_pt = 12.7 * 72 / 25.4
    assert frame.get("x") == format_coord(margin_pt)


def test_byte_determinism() -> None:
    curve_set, plan, legend = _fixture(ROWS)
    a = render_svg(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)
    b = render_svg(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
    assert b"\r" not in a  # LF newlines only
    assert not re.search(rb"\d\.\d{4,}", a)  # no coordinate exceeds 3 decimals


def test_print_safety_contract() -> None:
    curve_set, plan, legend = _fixture(ROWS)
    data = render_svg(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)
    text = data.decode()
    assert "<script" not in text and "href" not in text  # no scripts/external refs
    regions = re.search(r'<g id="regions"[^>]*>(.*?)</g>', text, re.DOTALL).group(0)
    region_strokes = set(re.findall(r'stroke="([^"]+)"', regions))
    assert region_strokes == {"#999"}  # single uniform gray line art, no subject/filler distinction


def test_validate_svg_catches_corruption() -> None:
    curve_set, plan, legend = _fixture(ROWS)
    data = render_svg(curve_set, plan, legend, PAL4, page_mm=PAGE_MM)
    with pytest.raises(StageError, match="well-formed"):
        validate_svg(data[:-20])
    tampered = data.replace(b'<g id="labels"', b'<g id="labelz"', 1)
    with pytest.raises(StageError, match="layer order"):
        validate_svg(tampered)
    missing_arc = data.replace(b'id="arc-0" ', b'id="arc-x" ', 1)
    with pytest.raises(StageError, match="exactly once"):
        validate_svg(missing_arc, curve_set)
    no_mm = data.replace(b'width="215.900mm"', b'width="215.900"', 1)
    with pytest.raises(StageError, match="physical"):
        validate_svg(no_mm)


def test_stage_wrapper_contract() -> None:
    with pytest.raises(ConfigError, match="stroke_pt"):
        SvgExportStage({"stroke_pt": 9.0})
    with pytest.raises(ConfigError, match="decimals"):
        SvgExportStage({"decimals": 7})

    stage = SvgExportStage({}, page_mm=PAGE_MM)
    assert stage.name == "svg"
    assert stage.requires == ("curve_set", "label_plan", "legend", "palette")
    assert stage.provides == ("svg",)
    curve_set, plan, legend = _fixture(ROWS)
    ctx = InMemoryContext(seed=0)
    ctx.put("curve_set", curve_set)
    ctx.put("label_plan", plan)
    ctx.put("legend", legend)
    ctx.put("palette", PAL4)
    stage.run(ctx)
    doc = ctx.get("svg")
    assert isinstance(doc, SvgDocument)
    assert doc.provenance.stage_name == "svg"
    assert doc.data.startswith(b"<?xml")

    bad = InMemoryContext(seed=0)
    bad.put("curve_set", "nope")
    bad.put("label_plan", plan)
    bad.put("legend", legend)
    bad.put("palette", PAL4)
    with pytest.raises(ConfigError):
        stage.run(bad)
