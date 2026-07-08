"""CLI entry point for the Sprint 24 technical quality comparison framework.

Usage:
    python -m benchmarks.comparison.cli run --scope examples --out report.json
    python -m benchmarks.comparison.cli run --scope category --category animals
    python -m benchmarks.comparison.cli run --scope full

Known issue: PYTHONHASHSEED sensitivity (see GOLDEN_TEST_STANDARDS.md §8) --
this CLI re-execs with a pinned seed for the same reason
``benchmarks/golden/cli.py`` does, so region counts (and therefore every
metric derived from them) are reproducible across invocations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HASH_SEED = "0"

if os.environ.get("PYTHONHASHSEED") != _HASH_SEED:
    os.environ["PYTHONHASHSEED"] = _HASH_SEED
    os.execv(sys.executable, [sys.executable, "-m", "benchmarks.comparison.cli", *sys.argv[1:]])

from benchmarks.comparison.report import (  # noqa: E402
    compare_category,
    compare_examples,
    compare_full_dataset,
)
from benchmarks.datasets.metadata_schema import CATEGORIES  # noqa: E402


def _cmd_run(args: argparse.Namespace) -> int:
    if args.scope == "examples":
        report = compare_examples()
    elif args.scope == "category":
        if not args.category:
            print("error: --scope category requires --category", file=sys.stderr)
            return 2
        report = compare_category(args.category)
    else:
        report = compare_full_dataset()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")

    print(f"comparison run {report.run_id}: {report.to_dict()['summary']}")
    for rec in report.cautions:
        print(f"  CAUTION {rec.fixture_id} ({rec.from_preset}->{rec.to_preset}): {rec.message}")
    print(f"report written to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.comparison.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Compare technical quality across presets")
    run_parser.add_argument("--scope", choices=["examples", "category", "full"], default="examples")
    run_parser.add_argument("--category", choices=CATEGORIES, default=None)
    run_parser.add_argument("--out", default="benchmarks/comparison_reports/latest.json")
    run_parser.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
