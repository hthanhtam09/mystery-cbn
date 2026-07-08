"""Renders one self-contained HTML file from a ``DebugRun``
(docs/VISUAL_DEBUGGER.md §4). No UI framework, no build step, no external
network assets -- every image/download is a base64 ``data:`` URI embedded
directly in the file, so the report opens in any browser standalone.
"""

from __future__ import annotations

import base64
import html
from urllib.parse import quote

from tools.visual_debugger.render_artifact import ArtifactView
from tools.visual_debugger.runner import DebugRun
from tools.visual_debugger.stages import build_stage_views

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 2rem; background: #fafafa; color: #1a1a1a; }
@media (prefers-color-scheme: dark) { body { background: #16181c; color: #e6e6e6; } }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }
@media (prefers-color-scheme: dark) { .meta { color: #999; } }
.stage { border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem;
         margin-bottom: 1rem; background: white; }
@media (prefers-color-scheme: dark) { .stage { background: #1f2228; border-color: #333; } }
.stage.missing { opacity: 0.45; }
.stage h2 { font-size: 1.1rem; margin: 0 0 0.4rem 0;
            display: flex; align-items: center; gap: 0.6rem; }
.badge { font-size: 0.7rem; font-weight: normal; color: #888; }
.summary { font-size: 0.85rem; color: #555; margin-bottom: 0.6rem; }
@media (prefers-color-scheme: dark) { .summary { color: #aaa; } }
.view { margin-bottom: 0.6rem; }
.view img { max-width: 100%; max-height: 480px; border: 1px solid #eee; display: block;
            image-rendering: pixelated; }
@media (prefers-color-scheme: dark) { .view img { border-color: #333; } }
.view pre { max-height: 320px; overflow: auto; background: #f4f4f4; padding: 0.6rem;
            border-radius: 4px; font-size: 0.75rem; }
@media (prefers-color-scheme: dark) { .view pre { background: #111318; } }
a.download { display: inline-block; font-size: 0.8rem; margin-top: 0.3rem;
             text-decoration: none; color: #2563eb; }
a.download::before { content: "\\2b07 "; }
.timing { font-size: 0.75rem; color: #999; float: right; }
.toc { margin-bottom: 1.5rem; font-size: 0.85rem; }
.toc a { margin-right: 0.8rem; }
"""


def _data_uri(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _download_link(view_name: str, filename: str, data: bytes) -> str:
    uri = _data_uri("application/octet-stream", data)
    label = f"{html.escape(view_name)} ({len(data):,} bytes)"
    return f'<a class="download" href="{uri}" download="{html.escape(filename)}">{label}</a>'


def _render_view_block(view_name: str, view: ArtifactView) -> str:
    parts = ['<div class="view">']
    if view.kind == "image" and view.preview_png is not None:
        uri = _data_uri("image/png", view.preview_png)
        parts.append(f'<img src="{uri}" alt="{html.escape(view_name)}"/>')
    elif view.text is not None:
        parts.append(f"<pre>{html.escape(view.text[:20000])}</pre>")
    parts.append(_download_link(view_name, view.download_filename, view.download_bytes))
    parts.append("</div>")
    return "".join(parts)


def render_html_report(run: DebugRun, *, title: str = "Pipeline Debug Report") -> str:
    stage_views = build_stage_views(run.ctx)

    toc = "".join(
        f'<a href="#stage-{quote(sv.label)}">{html.escape(sv.label)}</a>' for sv in stage_views
    )

    sections = []
    for sv in stage_views:
        css_class = "stage" if sv.available else "stage missing"
        timing = ""
        matching_timings = {
            k: v for k, v in run.stage_timings_s.items() if any(a in k for a in sv.artifact_names)
        }
        if matching_timings:
            timing = f'<span class="timing">{sum(matching_timings.values()) * 1000:.1f} ms</span>'
        anchor = quote(sv.label)
        body = (
            "".join(_render_view_block(name, v) for name, v in sv.views.items())
            if sv.available
            else '<div class="summary">not produced by this run</div>'
        )
        sections.append(
            f'<section class="{css_class}" id="stage-{anchor}">'
            f"<h2>{html.escape(sv.label)}{timing}"
            f'<span class="badge">{html.escape(", ".join(sv.artifact_names))}</span></h2>'
            f"{body}</section>"
        )

    validation = "PASSED" if run.validation_passed else "FAILED"
    total_ms = sum(run.stage_timings_s.values()) * 1000

    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f'<div class="meta">validation: {validation} &middot; '
        f"total stage time: {total_ms:.1f} ms &middot; "
        f"{len(stage_views)} stages, {sum(1 for s in stage_views if s.available)} produced</div>"
        f'<div class="toc">{toc}</div>'
        f"{''.join(sections)}"
        "</body></html>"
    )
