"""Tests for the debug-mode pipeline runner."""

from __future__ import annotations

from tools.visual_debugger.runner import run_pipeline_for_debug


def test_run_pipeline_for_debug_binds_every_expected_artifact(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")

    expected = {
        "source_bytes",
        "raster_source",
        "raster_working",
        "image_stats",
        "label_map",
        "palette",
        "region_graph",
        "topology_graph",
        "arc_graph",
        "curve_set",
        "label_plan",
        "legend",
        "svg",
        "pdf",
        "png_previews",
    }
    for name in expected:
        assert run.ctx.has(name), f"missing artifact {name!r}"


def test_run_pipeline_for_debug_records_stage_timings(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    assert len(run.stage_timings_s) > 0
    assert all(v >= 0.0 for v in run.stage_timings_s.values())


def test_run_pipeline_for_debug_reports_validation_passed(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    assert run.validation_passed is True


def test_stages_are_in_pipeline_order(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    names = [stage.name for stage in run.stages]
    assert names[0] == "load"
    assert names[-1] == "png"
