"""Visual/golden comparison (BENCHMARK_SPEC.md §4): SVG byte-hash and
structural diff, plus PNG luminance SSIM when a PDF preview is available.

Goldens are stored under ``benchmarks/goldens/<fixture_id>/`` (this
framework's analogue of ``tests/golden/`` -- kept separate since these are
synthetic-fixture goldens for the *benchmark* harness, not the engine's own
golden test suite) as ``page.svg`` + optional ``preview.png`` plus a
``GOLDEN_MANIFEST.json`` recording the hash and producing run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.framework.pipeline import PipelineRun
from mysterycbn.model.reports import GoldenOutcome

GOLDENS_ROOT = Path(__file__).resolve().parents[1] / "goldens"

_SSIM_PASS_THRESHOLD = 0.97
_COORD_RMS_MAX_MM = 0.3
_SEGMENT_COUNT_TOLERANCE = 0.10


@dataclass(frozen=True)
class GoldenComparison:
    """One fixture's comparison outcome (BENCHMARK_SPEC §4.2/§11 ``golden`` block)."""

    fixture_id: str
    svg_outcome: GoldenOutcome
    ssim_solved: float | None
    details: dict[str, object]


def _golden_dir(fixture_id: str) -> Path:
    return GOLDENS_ROOT / fixture_id


def has_golden(fixture_id: str) -> bool:
    return (_golden_dir(fixture_id) / "page.svg").is_file()


def write_golden(run: PipelineRun, *, engine_version: str, config_hash: str) -> None:
    """Bless the current run's output as the new golden (BENCHMARK_SPEC §4.3:
    goldens regenerate only via an explicit call, never implicitly during a
    comparison run)."""
    directory = _golden_dir(run.fixture_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "page.svg").write_bytes(run.svg_bytes)
    manifest: dict[str, object] = {
        "fixture_id": run.fixture_id,
        "svg_sha256": hashlib.sha256(run.svg_bytes).hexdigest(),
        "engine_version": engine_version,
        "config_hash": config_hash,
    }
    if run.pdf_bytes is not None:
        preview = _rasterize_preview(run.pdf_bytes)
        if preview is not None:
            _write_png(directory / "preview.png", preview)
            manifest["preview_shape"] = list(preview.shape)
    (directory / "GOLDEN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _rasterize_preview(pdf_bytes: bytes) -> np.ndarray | None:
    try:
        from mysterycbn.render.pdf import render_preview_png
    except ImportError:
        return None
    png_bytes = render_preview_png(pdf_bytes, dpi=72)
    return _decode_png(png_bytes)


def _decode_png(png_bytes: bytes) -> np.ndarray:
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    return np.asarray(img, dtype=np.float64)


def _encode_png(luminance: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(luminance.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _write_png(path: Path, luminance: np.ndarray) -> None:
    path.write_bytes(_encode_png(luminance))


_ARC_ID_RE = re.compile(r'id="arc-(\d+)"')
_D_ATTR_RE = re.compile(r'd="([^"]*)"')
_LEFT_RIGHT_RE = re.compile(r'data-left="(\d+)" data-right="(\d+)"')


@dataclass(frozen=True)
class _StructuralFeatures:
    """Cheap structural fingerprint of an SVG page (arc count, face-side
    multiset, segment counts) -- avoids a full XML diff for the common
    "nothing changed but whitespace" case, and gives the §4.2 structural
    diff its inputs when bytes do differ."""

    arc_count: int
    face_sides: list[tuple[str, str]]
    segment_counts: dict[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "arc_count": self.arc_count,
            "face_sides": self.face_sides,
            "segment_counts": self.segment_counts,
        }


def _structural_features(svg_bytes: bytes) -> _StructuralFeatures:
    text = svg_bytes.decode("utf-8")
    arc_ids = [int(m) for m in _ARC_ID_RE.findall(text)]
    left_right = _LEFT_RIGHT_RE.findall(text)
    segment_counts: dict[int, int] = {}
    for arc_id, d in zip(arc_ids, _D_ATTR_RE.findall(text), strict=False):
        segment_counts[arc_id] = d.count(" C ") + d.count(",C ")
    return _StructuralFeatures(
        arc_count=len(arc_ids), face_sides=sorted(left_right), segment_counts=segment_counts
    )


def _structural_diff_ok(
    golden_bytes: bytes, candidate_bytes: bytes
) -> tuple[bool, dict[str, object]]:
    """BENCHMARK_SPEC §4.2's structural-diff pass conditions when the byte
    hash differs: arc count, face-side multiset, and per-arc segment count
    within +-10% all match."""
    g = _structural_features(golden_bytes)
    c = _structural_features(candidate_bytes)
    details: dict[str, object] = {"golden": g.to_dict(), "candidate": c.to_dict()}

    if g.arc_count != c.arc_count:
        return False, details
    if g.face_sides != c.face_sides:
        return False, details

    if set(g.segment_counts) != set(c.segment_counts):
        return False, details
    for arc_id, g_n in g.segment_counts.items():
        c_n = c.segment_counts[arc_id]
        if g_n == 0:
            if c_n != 0:
                return False, details
            continue
        if abs(c_n - g_n) / g_n > _SEGMENT_COUNT_TOLERANCE:
            return False, details
    return True, details


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    if a.shape != b.shape:
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    return float(structural_similarity(a, b, data_range=255.0))  # type: ignore[no-untyped-call]


def compare_to_golden(run: PipelineRun) -> GoldenComparison:
    """Compare ``run``'s output against the stored golden. If no golden
    exists yet, the outcome is reported as ``INCOMPATIBLE`` with a clear
    reason (never silently treated as a pass)."""
    directory = _golden_dir(run.fixture_id)
    golden_svg_path = directory / "page.svg"
    if not golden_svg_path.is_file():
        return GoldenComparison(
            fixture_id=run.fixture_id,
            svg_outcome=GoldenOutcome.INCOMPATIBLE,
            ssim_solved=None,
            details={"reason": "no golden recorded for this fixture"},
        )

    golden_bytes = golden_svg_path.read_bytes()
    if golden_bytes == run.svg_bytes:
        svg_outcome = GoldenOutcome.IDENTICAL
        details: dict[str, object] = {"byte_identical": True}
    else:
        ok, diff_details = _structural_diff_ok(golden_bytes, run.svg_bytes)
        svg_outcome = GoldenOutcome.CHANGED_COMPATIBLE if ok else GoldenOutcome.INCOMPATIBLE
        details = {"byte_identical": False, **diff_details}

    ssim_solved: float | None = None
    preview_path = directory / "preview.png"
    if preview_path.is_file() and run.pdf_bytes is not None:
        golden_preview = _decode_png(preview_path.read_bytes())
        candidate_preview = _rasterize_preview(run.pdf_bytes)
        if candidate_preview is not None:
            ssim_solved = _ssim(golden_preview, candidate_preview)
            if ssim_solved < _SSIM_PASS_THRESHOLD and svg_outcome is not GoldenOutcome.IDENTICAL:
                svg_outcome = GoldenOutcome.INCOMPATIBLE
            details["ssim_threshold"] = _SSIM_PASS_THRESHOLD

    return GoldenComparison(
        fixture_id=run.fixture_id, svg_outcome=svg_outcome, ssim_solved=ssim_solved, details=details
    )
