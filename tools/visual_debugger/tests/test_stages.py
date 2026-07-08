"""Tests for artifact-name -> stage-view dispatch."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from mysterycbn.kernel.context import InMemoryContext
from tools.visual_debugger.runner import run_pipeline_for_debug
from tools.visual_debugger.stages import STAGE_LABELS, _render_one, build_stage_views


def test_every_declared_stage_is_available_for_a_full_run(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    stage_views = build_stage_views(run.ctx)

    assert len(stage_views) == len(STAGE_LABELS)
    unavailable = [sv.label for sv in stage_views if not sv.available]
    assert unavailable == [], f"stages missing views: {unavailable}"


def test_every_view_has_nonempty_download_bytes(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    stage_views = build_stage_views(run.ctx)

    for sv in stage_views:
        for name, view in sv.views.items():
            assert len(view.download_bytes) > 0, f"{sv.label}/{name} has empty download"
            assert view.download_filename


def test_image_views_produce_decodable_png(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    stage_views = build_stage_views(run.ctx)

    for sv in stage_views:
        for name, view in sv.views.items():
            if view.kind == "image" and view.preview_png is not None:
                img = Image.open(io.BytesIO(view.preview_png))
                img.load()
                assert img.size[0] > 0 and img.size[1] > 0, f"{sv.label}/{name}"


def test_unknown_artifact_name_raises_key_error() -> None:
    ctx = InMemoryContext(seed=0)

    class _Fake:
        provenance = None

    ctx.put("mystery_artifact", _Fake())  # type: ignore[arg-type]

    with pytest.raises(KeyError, match="no renderer registered"):
        _render_one(ctx, "mystery_artifact")
