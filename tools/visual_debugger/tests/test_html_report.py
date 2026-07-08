"""Tests for the self-contained HTML report generator."""

from __future__ import annotations

import re

from tools.visual_debugger.html_report import render_html_report
from tools.visual_debugger.runner import run_pipeline_for_debug
from tools.visual_debugger.stages import STAGE_LABELS


def test_report_is_self_contained_html(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    report = render_html_report(run)

    assert report.startswith("<!doctype html>")
    assert "</html>" in report
    # No external network dependency: no non-namespace http(s) URL anywhere.
    external = [url for url in re.findall(r'https?://[^"\s<]+', report) if "w3.org" not in url]
    assert external == []


def test_report_contains_every_stage_label(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    report = render_html_report(run)

    for label, _ in STAGE_LABELS:
        assert label in report, f"missing stage label {label!r}"


def test_report_embeds_downloadable_data_uris(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    report = render_html_report(run)

    assert "data:image/png;base64," in report
    assert "data:application/octet-stream;base64," in report
    assert 'class="download"' in report


def test_report_reflects_validation_status(two_tone_image_bytes: bytes) -> None:
    run = run_pipeline_for_debug(two_tone_image_bytes, preset="medium")
    report = render_html_report(run)

    assert ("validation: PASSED" in report) or ("validation: FAILED" in report)
    assert f"validation: {'PASSED' if run.validation_passed else 'FAILED'}" in report
