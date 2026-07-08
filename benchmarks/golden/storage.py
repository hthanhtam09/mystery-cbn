"""On-disk golden storage for the dataset golden-test framework
(docs/GOLDEN_TEST_STANDARDS.md §3).

Goldens live under ``benchmarks/golden_store/<fixture_id>/`` -- kept
separate from both ``tests/golden/`` (the engine's own hand-picked golden
suite) and ``benchmarks/goldens/`` (the synthetic-ladder benchmark harness'
goldens, ``benchmarks/framework/visual.py``) since this tier is scoped to
the Sprint 20 categorized dataset specifically.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.golden.topology_compare import TopologyFingerprint

GOLDEN_STORE_ROOT = Path(__file__).resolve().parents[1] / "golden_store"


def golden_dir(fixture_id: str) -> Path:
    return GOLDEN_STORE_ROOT / fixture_id


def has_golden(fixture_id: str) -> bool:
    return (golden_dir(fixture_id) / "page.svg").is_file()


def read_golden_svg(fixture_id: str) -> bytes:
    return (golden_dir(fixture_id) / "page.svg").read_bytes()


def read_golden_preview(fixture_id: str) -> bytes | None:
    path = golden_dir(fixture_id) / "preview.png"
    return path.read_bytes() if path.is_file() else None


def read_golden_topology(fixture_id: str) -> TopologyFingerprint:
    data = json.loads((golden_dir(fixture_id) / "topology.json").read_text())
    return TopologyFingerprint.from_dict(data)


def write_golden(
    fixture_id: str,
    *,
    svg_bytes: bytes,
    preview_png_bytes: bytes | None,
    topology: TopologyFingerprint,
    engine_version: str,
    dataset_version: int,
    category: str,
) -> None:
    """Bless the current run's output as the new golden. Only called by the
    explicit update workflow (``update.py``), never implicitly during a
    comparison run (BENCHMARK_SPEC.md §4.3)."""
    directory = golden_dir(fixture_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "page.svg").write_bytes(svg_bytes)
    (directory / "topology.json").write_text(
        json.dumps(topology.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    if preview_png_bytes is not None:
        (directory / "preview.png").write_bytes(preview_png_bytes)
    manifest = {
        "fixture_id": fixture_id,
        "category": category,
        "engine_version": engine_version,
        "dataset_version": dataset_version,
        "has_preview": preview_png_bytes is not None,
        **topology.to_dict(),
    }
    (directory / "GOLDEN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
