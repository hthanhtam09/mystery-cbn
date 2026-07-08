"""Unit tests for the raster/graph-domain data model (DATA_MODEL_SPEC §2–§8)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mysterycbn.model.records import (
    LabelMap,
    Palette,
    PaletteColor,
    Provenance,
    RasterImage,
    Region,
    RegionGraph,
)

PROV = Provenance(
    stage_name="test", stage_version="1.0.0", config_hash="ab" * 32, source_hash="cd" * 32
)


def test_provenance_rejects_bad_hashes() -> None:
    with pytest.raises(ValueError, match="hex"):
        Provenance("s", "1", "not-a-hash", "cd" * 32)
    with pytest.raises(ValueError, match="non-empty"):
        Provenance("", "1", "ab" * 32, "cd" * 32)


def test_raster_image_valid_and_immutable() -> None:
    img = RasterImage(
        pixels=np.zeros((64, 64, 3), dtype=np.float32),
        work_scale=0.0,
        resize_factor=1.0,
        icc_applied=False,
        exif_orientation=1,
        provenance=PROV,
    )
    with pytest.raises(ValueError, match="read-only"):
        img.pixels[0, 0, 0] = 1.0
    assert json.dumps(img.to_dict())  # serializable
    assert img.to_dict()["pixels"]["shape"] == [64, 64, 3]  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pixels": np.zeros((8, 8, 3), dtype=np.float32)}, "≥ 64"),
        ({"pixels": np.full((64, 64, 3), 2.0, dtype=np.float32)}, r"\[0, 1\]"),
        ({"exif_orientation": 9}, "EXIF"),
        ({"resize_factor": 0.0}, "resize_factor"),
    ],
)
def test_raster_image_validation(kwargs: dict, match: str) -> None:  # type: ignore[type-arg]
    base: dict = {  # type: ignore[type-arg]
        "pixels": np.zeros((64, 64, 3), dtype=np.float32),
        "work_scale": 0.0,
        "resize_factor": 1.0,
        "icc_applied": False,
        "exif_orientation": 1,
        "provenance": PROV,
    }
    with pytest.raises(ValueError, match=match):
        RasterImage(**{**base, **kwargs})


def test_palette_color_from_lab_and_mismatch_rejection() -> None:
    color = PaletteColor.from_lab(0, (50.0, 20.0, -30.0), coverage_px=100)
    assert min(color.srgb) >= 0.0 and max(color.srgb) <= 1.0
    with pytest.raises(ValueError, match="from_lab"):
        PaletteColor(0, (50.0, 20.0, -30.0), (0.9, 0.9, 0.9), 100)


def _palette(labs: list[tuple[float, float, float]], min_delta_e: float = 0.0) -> Palette:
    colors = tuple(PaletteColor.from_lab(i, lab, 10) for i, lab in enumerate(labs))
    return Palette(colors=colors, provenance=PROV, min_delta_e=min_delta_e)


def test_palette_delta_table_and_separation() -> None:
    pal = _palette([(20.0, 0.0, 0.0), (80.0, 0.0, 0.0)], min_delta_e=10.0)
    assert pal.size == 2
    assert pal.delta_e_table.shape == (2, 2)
    assert pal.delta_e_table[0, 0] == 0.0
    assert pal.delta_e_table[0, 1] > 10.0
    with pytest.raises(ValueError, match="read-only"):
        pal.delta_e_table[0, 1] = 0.0


def test_palette_rejects_separation_violation_and_sparse_indices() -> None:
    with pytest.raises(ValueError, match="separation"):
        _palette([(50.0, 0.0, 0.0), (50.5, 0.0, 0.0)], min_delta_e=5.0)
    colors = (
        PaletteColor.from_lab(0, (20.0, 0.0, 0.0), 1),
        PaletteColor.from_lab(2, (80.0, 0.0, 0.0), 1),  # gap in indices
    )
    with pytest.raises(ValueError, match="dense"):
        Palette(colors=colors, provenance=PROV)


def test_label_map_pairing() -> None:
    lm = LabelMap(labels=np.array([[0, 1], [1, 2]], dtype=np.int32), provenance=PROV)
    pal2 = _palette([(20.0, 0.0, 0.0), (80.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="out of range"):
        lm.validate_against(pal2)
    with pytest.raises(ValueError, match="≥ 0"):
        LabelMap(labels=np.array([[-1]], dtype=np.int32), provenance=PROV)


def test_region_validation() -> None:
    region = Region(
        0, 0, area_px=4, bbox=(0, 0, 1, 1), seed_px=(0, 0), perimeter_px=8, centroid=(0.5, 0.5)
    )
    assert json.dumps(region.to_dict())
    with pytest.raises(ValueError, match="centroid"):
        Region(0, 0, 4, (0, 0, 1, 1), (0, 0), 8, (5.0, 0.5))
    with pytest.raises(ValueError, match="bbox"):
        Region(0, 0, 4, (1, 1, 0, 0), (0, 0), 8, (0.5, 0.5))
    with pytest.raises(ValueError, match="seed"):
        Region(0, 0, 4, (0, 0, 1, 1), (3, 3), 8, (0.5, 0.5))


def test_region_graph_accessors_and_validation() -> None:
    regions = (
        Region(0, 0, 2, (0, 0, 0, 1), (0, 0), 6, (0.0, 0.5)),
        Region(1, 1, 2, (0, 2, 0, 3), (0, 2), 6, (0.0, 2.5)),
    )
    graph = RegionGraph(
        regions=regions,
        component_map=np.array([[0, 0, 1, 1]], dtype=np.int32),
        edges=((0, 1, 25.0, 1),),
        provenance=PROV,
    )
    assert graph.neighbors(0) == (1,)
    assert graph.edge_weight(1, 0) == (1.0, 25.0)
    with pytest.raises(KeyError):
        graph.edge_weight(0, 0)
    with pytest.raises(ValueError, match="dense"):
        RegionGraph(regions, np.array([[0, 0, 2, 2]], dtype=np.int32), (), PROV)
    with pytest.raises(ValueError, match="a < b"):
        RegionGraph(regions, np.array([[0, 0, 1, 1]], dtype=np.int32), ((1, 0, 25.0, 1),), PROV)
