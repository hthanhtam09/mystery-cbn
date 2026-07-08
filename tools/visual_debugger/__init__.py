"""Visual pipeline debugger (Sprint 22): a developer tool, not test/benchmark
infrastructure. Runs ``convert()``'s full pipeline while keeping every
stage's artifacts, renders each to a viewable/downloadable form, and emits
one self-contained HTML report -- no UI framework, no build step, no
external assets. Open the file in a browser.

    python -m tools.visual_debugger.cli path/to/image.jpg -o report.html

Stages shown (ENGINE_SPEC.md's pipeline, in order): Original -> Working
Resolution -> Quantized -> Label Map -> Region Graph -> Arc Graph -> Curves
-> Labels -> Legend -> SVG -> Preview.
"""

from __future__ import annotations
