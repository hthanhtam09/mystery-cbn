"""Unit tests for golden on-disk storage (round-trip, no network/engine)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.golden import storage
from benchmarks.golden.topology_compare import TopologyFingerprint


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "GOLDEN_STORE_ROOT", tmp_path)


def test_has_golden_false_before_write() -> None:
    assert not storage.has_golden("D-fake-fixture")


def test_write_then_read_golden_round_trip() -> None:
    topology = TopologyFingerprint(region_count=4, arc_count=10, face_count=4)
    storage.write_golden(
        "D-fake-fixture",
        svg_bytes=b"<svg/>",
        preview_png_bytes=b"\x89PNG",
        topology=topology,
        engine_version="0.1.0",
        dataset_version=1,
        category="animals",
    )
    assert storage.has_golden("D-fake-fixture")
    assert storage.read_golden_svg("D-fake-fixture") == b"<svg/>"
    assert storage.read_golden_preview("D-fake-fixture") == b"\x89PNG"
    assert storage.read_golden_topology("D-fake-fixture") == topology


def test_write_golden_without_preview() -> None:
    topology = TopologyFingerprint(region_count=1, arc_count=1, face_count=1)
    storage.write_golden(
        "D-fake-nopreview",
        svg_bytes=b"<svg/>",
        preview_png_bytes=None,
        topology=topology,
        engine_version="0.1.0",
        dataset_version=1,
        category="food",
    )
    assert storage.read_golden_preview("D-fake-nopreview") is None
