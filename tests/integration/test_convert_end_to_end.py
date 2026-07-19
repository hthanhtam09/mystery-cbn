"""Integration test for Sprint 19's orchestration layer: ``convert()`` is
the engine's only public entry point and must run the full pipeline
(Load -> Preprocess -> Analyze -> Quantize -> Denoise -> Region Graph ->
Merge Tiny Regions -> Contour Extraction -> Simplify -> Curve Smoothing ->
Label Placement -> Legend -> Validation -> SVG -> PDF -> PNG) end-to-end
from raw image bytes to a validated ``OutputBundle``.

Prior to Sprint 19, no code path in the repository constructed a full
``Pipeline``/``Registry`` and ran it end-to-end (Sprint 18 architecture
audit: ``Orchestrator`` was an abstract class with no concrete subclass;
``adapters/cli``/``adapters/api`` were empty; no stage was ever registered
into ``InMemoryStageRegistry`` outside of unit tests using fake stages).
These tests are the first in the repository to exercise the real
end-to-end conversion.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mysterycbn.app import convert
from mysterycbn.app.reports_io import write_bundle_reports
from mysterycbn.foundation.errors import CancelledError, EngineError, InputError
from mysterycbn.kernel.cancellation import ManualCancelToken
from mysterycbn.kernel.progress import ProgressKind
from mysterycbn.model.reports import OutputBundle


def _synthetic_photo_bytes(size: tuple[int, int] = (96, 96)) -> bytes:
    """A smooth RGB gradient -- stands in for a real photograph without
    requiring a committed binary fixture (no copyrighted imagery anywhere
    in this repo, ARCHITECTURE.md §10 legal invariant)."""
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(w - 1, 1)).astype(np.uint8)
    g = (y * 255 // max(h - 1, 1)).astype(np.uint8)
    b = (((x + y) * 255) // max(w + h - 2, 1)).astype(np.uint8)
    arr = np.stack([r, g, b], axis=2)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _two_tone_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    """A hard two-region split -- exercises the flat-art path (few, large,
    high-contrast regions) as a fast smoke fixture."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (220, 30, 30)
    arr[:, w // 2 :] = (30, 30, 220)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


# --------------------------------------------------------------- golden path


def test_convert_end_to_end_from_bytes_produces_a_valid_bundle() -> None:
    bundle = convert(_two_tone_bytes(), preset="medium")

    assert isinstance(bundle, OutputBundle)
    assert len(bundle.svg) > 0
    assert bundle.svg.startswith(b"<?xml")
    assert bundle.pdf is not None and len(bundle.pdf) > 0
    assert set(bundle.previews) == {"lineart", "solved"}
    assert len(bundle.previews["lineart"]) > 0
    assert len(bundle.previews["solved"]) > 0

    # Both previews must decode as valid PNGs of the same page size.
    lineart = Image.open(io.BytesIO(bundle.previews["lineart"]))
    solved = Image.open(io.BytesIO(bundle.previews["solved"]))
    assert lineart.size == solved.size
    assert lineart.format == "PNG"
    assert solved.format == "PNG"


def test_convert_embeds_all_four_canonical_validation_reports() -> None:
    bundle = convert(_two_tone_bytes(), preset="medium")
    names = [v.validator_name for v in bundle.report.validation]
    assert names == ["fidelity", "topology", "printability", "palette"]
    assert all(v.passed for v in bundle.report.validation)


def test_convert_populates_sprint23_quality_metrics() -> None:
    """OutputBundle.quality (Sprint 23) is populated end-to-end and is
    purely observational: it never blocks the bundle, regardless of value."""
    bundle = convert(_two_tone_bytes(), preset="medium")
    expected = {
        "QM-13", "QM-14", "QM-11", "QM-08", "QM-16", "QM-22",
        "label_overlap_rate_pct", "QM-26", "QM-28", "printability_score",
    }  # fmt: skip
    assert expected <= set(bundle.quality.metrics)
    assert bundle.quality.metrics["QM-26"].value == 1.0
    assert bundle.quality.metrics["QM-28"].value == 1.0

    with tempfile.TemporaryDirectory() as tmp:
        metrics_path, report_path = write_bundle_reports(bundle, Path(tmp))
        assert metrics_path.is_file()
        assert report_path.is_file()


def test_convert_report_carries_reproducibility_record() -> None:
    bundle = convert(_two_tone_bytes(), preset="medium", seed=7)
    report = bundle.report
    assert report.seed == 7
    assert len(report.input_hash) == 64  # sha256 hex
    assert report.engine_version
    # Every one of the 14 Stage-protocol slots produced a timing entry
    # (plus the orchestrator's own "_total_s"), proving the full plan ran.
    expected_stages = {
        "load", "preprocess", "analyze", "quantize", "denoise", "regions",
        "merge_tiny", "topology", "arcgraph", "simplify", "bezier",
        "labels", "legend", "svg", "pdf", "png",
    }  # fmt: skip
    assert expected_stages.issubset(report.stage_timings_s.keys())


def test_convert_accepts_a_realistic_photo_like_gradient() -> None:
    """Not just a synthetic 2-region flat image -- a smooth-gradient
    fixture exercises quantize/denoise/merge on a richer palette."""
    bundle = convert(_synthetic_photo_bytes(), preset="medium")
    assert len(bundle.svg) > 0
    assert all(v.passed for v in bundle.report.validation)


@pytest.mark.parametrize("preset", ["easy", "medium", "hard", "dense"])
def test_convert_runs_under_every_difficulty_preset(preset: str) -> None:
    bundle = convert(_two_tone_bytes(), preset=preset)
    assert all(v.passed for v in bundle.report.validation)


# --------------------------------------------------------------- determinism


def test_convert_is_deterministic_given_the_same_seed() -> None:
    """I2: same input + config => byte-identical SVG (ARCHITECTURE.md §0)."""
    data = _two_tone_bytes()
    first = convert(data, preset="medium", seed=0)
    second = convert(data, preset="medium", seed=0)
    assert first.svg == second.svg


# ------------------------------------------------------------- progress API


def test_convert_emits_progress_from_zero_to_one_across_every_stage() -> None:
    events: list[tuple[str, str, float]] = []

    class _Listener:
        def on_progress(self, event: object) -> None:
            events.append(
                (event.kind.name, event.stage_name, event.fraction_complete)  # type: ignore[attr-defined]
            )

    convert(_two_tone_bytes(), on_progress=_Listener())

    assert events, "expected at least one progress event"
    assert events[0] == ("STAGE_STARTED", "load", 0.0)
    assert events[-1][0] == "STAGE_FINISHED"
    assert events[-1][2] == pytest.approx(1.0)
    kinds = {ProgressKind[k] for k, _, _ in events}
    assert kinds == {ProgressKind.STAGE_STARTED, ProgressKind.STAGE_FINISHED}
    fractions = [f for _, _, f in events]
    assert fractions == sorted(fractions)  # monotonically non-decreasing


# ------------------------------------------------------------- cancellation


def test_convert_raises_cancelled_error_when_token_is_pre_cancelled() -> None:
    token = ManualCancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        convert(_two_tone_bytes(), cancel_token=token)


def test_convert_stops_mid_pipeline_on_cancellation() -> None:
    """Cancellation requested after the first stage starts must still abort
    before a later stage runs (cooperative check between every stage)."""
    token = ManualCancelToken()
    seen_stages: list[str] = []

    class _CancelAfterLoad:
        def on_progress(self, event: object) -> None:
            seen_stages.append(event.stage_name)  # type: ignore[attr-defined]
            if event.stage_name == "load" and event.kind.name == "STAGE_FINISHED":  # type: ignore[attr-defined]
                token.cancel()

    with pytest.raises(CancelledError):
        convert(_two_tone_bytes(), on_progress=_CancelAfterLoad(), cancel_token=token)
    assert "load" in seen_stages
    assert "png" not in seen_stages  # aborted long before the final stage


# ---------------------------------------------------------- error propagation


def test_convert_raises_input_error_for_a_missing_file() -> None:
    with pytest.raises(InputError):
        convert("/nonexistent/path/does-not-exist.jpg")


def test_convert_raises_engine_error_for_garbage_bytes() -> None:
    with pytest.raises(EngineError):
        convert(b"not an image at all")


def test_convert_rejects_an_unknown_preset() -> None:
    with pytest.raises(EngineError):
        convert(_two_tone_bytes(), preset="extreme")


# ------------------------------------------------------------- OutputBundle


def test_output_bundle_is_atomic_and_never_partially_constructed() -> None:
    """A failed convert() must raise, never return a partially-built bundle."""
    try:
        convert("/nonexistent/path.jpg")
    except EngineError:
        pass
    else:
        pytest.fail("expected an EngineError for a missing source file")
