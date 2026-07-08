"""CLI entry point for the Benchmark Framework.

Usage:
    python -m benchmarks.framework.cli run --suite smoke --out benchmarks/reports/latest
    python -m benchmarks.framework.cli bless --suite smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.framework.exporters import (
    write_csv_report,
    write_html_dashboard,
    write_json_report,
    write_stage_timings_csv,
)
from benchmarks.framework.fixtures import load_full_ladder, load_smoke_fixtures
from benchmarks.framework.pipeline import run_pipeline
from benchmarks.framework.regression import update_baseline
from benchmarks.framework.report import build_report
from benchmarks.framework.visual import write_golden


def _cmd_run(args: argparse.Namespace) -> int:
    report = build_report(
        suite=args.suite, repeats=args.repeats, require_golden=args.require_golden
    )
    out_dir = Path(args.out)
    write_json_report(report, out_dir / "report.json")
    write_csv_report(report, out_dir / "metrics.csv")
    write_stage_timings_csv(report, out_dir / "stages.csv")
    write_html_dashboard(report, out_dir / "dashboard.html")

    print(f"run {report.run_id}: {'ACCEPTED' if report.accepted else 'REJECTED'}")
    print(f"score {report.score_total:.2f}/100 {report.score_dimensions}")
    if not report.accepted:
        for f in report.failures:
            print(f"  FAIL {f.metric} {f.fixture}/{f.preset}: {f.value} not in {f.band}")
    print(f"reports written to {out_dir}/")
    return 0 if report.accepted else 1


def _cmd_bless(args: argparse.Namespace) -> int:
    """Regenerate goldens and seed baselines from the current engine output
    (BENCHMARK_SPEC §4.3 / §7.1: explicit, never implicit)."""
    from mysterycbn import __version__ as engine_version

    fixtures = load_smoke_fixtures() if args.suite == "smoke" else load_full_ladder()
    for fixture in fixtures:
        run = run_pipeline(fixture)
        write_golden(run, engine_version=engine_version, config_hash="0" * 64)
        from benchmarks.framework.quality import compute_quality_report

        quality = compute_quality_report(run)
        for metric_id, result in quality.metrics.items():
            update_baseline(fixture.fixture_id, metric_id, result.value, run_id="bless")
        print(f"blessed {fixture.fixture_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.framework")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a benchmark suite and emit reports")
    run_p.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    run_p.add_argument("--repeats", type=int, default=3)
    run_p.add_argument("--out", default="benchmarks/reports/latest")
    run_p.add_argument("--require-golden", action="store_true")
    run_p.set_defaults(func=_cmd_run)

    bless_p = sub.add_parser("bless", help="regenerate goldens + baselines from current output")
    bless_p.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    bless_p.set_defaults(func=_cmd_bless)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
