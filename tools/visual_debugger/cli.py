"""CLI entry point for the visual pipeline debugger (Sprint 22).

Usage:
    python -m tools.visual_debugger.cli path/to/image.jpg -o report.html
    python -m tools.visual_debugger.cli path/to/image.jpg --preset hard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.visual_debugger.html_report import render_html_report
from tools.visual_debugger.runner import run_pipeline_for_debug


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.visual_debugger.cli")
    parser.add_argument("source", type=Path, help="Path to a source image")
    parser.add_argument("-o", "--out", type=Path, default=Path("debug_report.html"))
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    run = run_pipeline_for_debug(args.source, preset=args.preset, seed=args.seed)
    html_text = render_html_report(run, title=f"Debug: {args.source.name}")
    args.out.write_text(html_text, encoding="utf-8")
    print(f"wrote {args.out} ({len(html_text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
