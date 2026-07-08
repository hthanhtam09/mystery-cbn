"""CLI entry point for the dataset golden-test framework (Sprint 21).

Usage:
    python -m benchmarks.golden.cli run --suite golden --out benchmarks/golden_reports/latest.json
    python -m benchmarks.golden.cli bless --suite golden
    python -m benchmarks.golden.cli bless --fixture D-animals-examples-01

Known issue (docs/GOLDEN_TEST_STANDARDS.md §8): building this framework
surfaced that the engine's region-merge path is sensitive to Python's
per-process hash-seed randomization for some fixtures (observed region
count varying run-to-run for byte-identical input, e.g. the ``animals``
category's overlapping-blob label maps) -- a real QM-27 determinism gap in
the engine, out of scope to fix here ("no engine modifications"). This CLI
works around it at the harness level by re-executing itself with
``PYTHONHASHSEED=0`` so golden comparisons are reproducible; this masks the
symptom for testing purposes only and does not fix the underlying bug.
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
    os.execv(sys.executable, [sys.executable, "-m", "benchmarks.golden.cli", *sys.argv[1:]])

from benchmarks.golden.report import run_full_dataset_suite, run_golden_suite  # noqa: E402
from benchmarks.golden.update import (  # noqa: E402
    bless_fixture_ids,
    bless_full_dataset,
    bless_golden_ladder,
)


def _cmd_run(args: argparse.Namespace) -> int:
    report = run_full_dataset_suite() if args.suite == "full" else run_golden_suite()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")

    print(f"golden run {report.run_id}: {'ACCEPTED' if report.accepted else 'REJECTED'}")
    print(f"{report.to_dict()['summary']}")
    for failure in report.failures:
        print(f"  FAIL {failure.fixture_id} ({failure.category}): {failure.svg_outcome.value}")
    print(f"report written to {out_path}")
    return 0 if report.accepted else 1


def _cmd_bless(args: argparse.Namespace) -> int:
    if args.fixture:
        blessed = bless_fixture_ids(args.fixture)
    elif args.suite == "full":
        blessed = bless_full_dataset()
    else:
        blessed = bless_golden_ladder()
    for fixture_id in blessed:
        print(f"blessed {fixture_id}")
    print(f"blessed {len(blessed)} fixture(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.golden.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Compare current output to stored goldens")
    run_parser.add_argument("--suite", choices=["golden", "full"], default="golden")
    run_parser.add_argument("--out", default="benchmarks/golden_reports/latest.json")
    run_parser.set_defaults(func=_cmd_run)

    bless_parser = subparsers.add_parser("bless", help="Regenerate goldens from current output")
    bless_parser.add_argument("--suite", choices=["golden", "full"], default="golden")
    bless_parser.add_argument(
        "--fixture", action="append", help="Bless a specific fixture id (repeatable)"
    )
    bless_parser.set_defaults(func=_cmd_bless)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
