"""Gate-failure triage harness.

Runs the full pipeline on a directory of images (via the visual debugger's
``run_pipeline_for_debug``, which mirrors ``convert()`` including preset
sections) and calls the four canonical validators directly on the resulting
context — instead of letting ``run_validation`` raise — so every finding can
be dumped and aggregated even when the gate would have failed.

For each finding it reports the validator, invariant, face id, whether the
face is a filler cell (``filler_region_ids``), its area in mm², and the
failing value (agreement / watertight residual / intersection kind), plus
stage timings. The aggregate table at the end is the decision input for
which robustness fix to apply (fidelity filler floor vs sliver fold vs
watertight recalibration).

Usage:
    python -m tools.gate_triage IMAGES_DIR [--preset dense] [--seed 0]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from mysterycbn.foundation.errors import EngineError
from mysterycbn.foundation.units import MM_PER_INCH, PT_PER_INCH
from mysterycbn.validate.fidelity import validate_fidelity
from mysterycbn.validate.palette import validate_palette
from mysterycbn.validate.printability import validate_printability
from mysterycbn.validate.topology import validate_topology

from tools.visual_debugger.runner import run_pipeline_for_debug

_PT_TO_MM = MM_PER_INCH / PT_PER_INCH

_FACE_RE = re.compile(r"face (\d+)")
_AGREEMENT_RE = re.compile(r"agreement ([0-9.]+)")


def _face_areas_mm2(ctx) -> dict[int, float]:
    from mysterycbn.stages.vector._face_area import face_area_pt2

    curve_set = ctx.get("curve_set")
    tolerance_pt = 0.1 / _PT_TO_MM  # 0.1mm flatten, same as validators
    return {
        face.face_id: face_area_pt2(face, curve_set, tolerance_pt) * _PT_TO_MM * _PT_TO_MM
        for face in curve_set.faces
    }


def triage_one(path: Path, preset: str, seed: int) -> list[dict]:
    """Returns one record per FATAL/WARN finding (empty list = clean)."""
    run = run_pipeline_for_debug(path, preset=preset, seed=seed)
    ctx = run.ctx
    filler = ctx.get("filler_region_ids") if ctx.has("filler_region_ids") else frozenset()
    if not isinstance(filler, (set, frozenset)):
        filler = frozenset()

    reports = (
        validate_fidelity(ctx),
        validate_topology(ctx),
        validate_printability(ctx, d_min_mm=2.5, font_min_pt=6.0),
        validate_palette(ctx, merge_delta_e=3.0),
    )
    records: list[dict] = []
    areas: dict[int, float] | None = None
    for report in reports:
        for f in report.findings:
            face_id = None
            m = _FACE_RE.search(f.location or "")
            if m:
                face_id = int(m.group(1))
            if face_id is not None and areas is None:
                areas = _face_areas_mm2(ctx)
            value = None
            m = _AGREEMENT_RE.search(f.message)
            if m:
                value = float(m.group(1))
            records.append(
                {
                    "image": path.name,
                    "validator": report.validator_name,
                    "severity": f.severity.name,
                    "invariant": f.invariant,
                    "message": f.message,
                    "location": f.location,
                    "face_id": face_id,
                    "is_filler": face_id in filler if face_id is not None else None,
                    "area_mm2": areas.get(face_id) if areas and face_id is not None else None,
                    "value": value,
                }
            )
    total = sum(run.stage_timings_s.values())
    slowest = sorted(run.stage_timings_s.items(), key=lambda kv: -kv[1])[:3]
    print(
        f"[{path.name}] gate={'PASS' if not any(r['severity'] == 'FATAL' for r in records) else 'FAIL'}"
        f" findings={len(records)} total={total:.0f}s slowest={slowest}"
    )
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images_dir", type=Path)
    ap.add_argument("--preset", default="dense")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(
        p
        for p in args.images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bin")
    )
    if not paths:
        print(f"no images in {args.images_dir}", file=sys.stderr)
        return 2

    all_records: list[dict] = []
    for p in paths:
        try:
            all_records.extend(triage_one(p, args.preset, args.seed))
        except EngineError as exc:
            print(f"[{p.name}] PIPELINE ERROR before validation: {type(exc).__name__}: {exc}")
            all_records.append(
                {"image": p.name, "validator": "pipeline", "severity": "FATAL",
                 "invariant": "-", "message": str(exc), "location": None,
                 "face_id": None, "is_filler": None, "area_mm2": None, "value": None}
            )

    fatal = [r for r in all_records if r["severity"] == "FATAL"]
    print("\n=== FATAL findings detail ===")
    for r in fatal:
        area = f"{r['area_mm2']:.1f}mm²" if r["area_mm2"] is not None else "-"
        print(
            f"  {r['image']} | {r['validator']}/{r['invariant']} | filler={r['is_filler']}"
            f" | area={area} | {r['message']}"
        )
    print("\n=== Aggregate (validator, invariant, filler) ===")
    for key, n in Counter(
        (r["validator"], r["invariant"], r["is_filler"]) for r in fatal
    ).most_common():
        print(f"  {key}: {n}")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
