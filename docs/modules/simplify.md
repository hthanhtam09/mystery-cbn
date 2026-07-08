# Module Design — Simplify (`stages/vector/simplify`)

**Status:** v1.0 — implemented (Sprint 19). Governing spec: [ARCHITECTURE.md §15](../ARCHITECTURE.md) "simplify" row; [ENGINE_SPEC.md §16-17](../ENGINE_SPEC.md).

## Purpose

Wire the existing Visvalingam-Whyatt polyline simplification (`foundation/geometry/default.py::DefaultGeometryKernel.simplify_polyline`, MATH_SPEC §8.1) into the `ArcGraph → ArcGraph` pipeline slot ARCHITECTURE.md's dossier names. Before Sprint 19 the kernel method existed and was unit-tested in isolation, but no stage ever called it (confirmed: `grep -rn "simplify_polyline" src/mysterycbn/stages/` returned nothing).

## Algorithm

Each arc's polyline is simplified independently: `PolylineData(arc.points, is_closed=arc.closed)` → `kernel.simplify_polyline(polyline, tolerance_pt)` → a new `Arc` with the same `arc_id`/`left_region`/`right_region`/`closed`, only `points` replaced. Faces (walks) are carried over unchanged — arc identity and endpoint semantics are preserved, so no face-walk reference is invalidated.

Endpoint pinning is the kernel's own responsibility (`simplify_polyline` never moves index 0 or the last index of an open polyline), which is what keeps two arcs sharing a junction in exact agreement after independent simplification — no cross-arc coordination is needed or attempted.

## Rejected alternatives

Simplifying the whole `ArcGraph` as one connected structure (shared-vertex-aware VW): rejected as unnecessary — the kernel's per-arc endpoint pinning already guarantees junction agreement without needing global bookkeeping, and per-arc simplification is what the kernel's own contract already supports without modification (no engine-module redesign, per Sprint 19's brief).

## Quality requirements

Verified end-to-end (`tests/integration/test_convert_end_to_end.py`): `fit_curves` succeeds on simplified output, all four canonical validators pass, and topology remains watertight — proving simplification does not break arc pairing or face area identity. Not independently unit-tested at the module level in this sprint (orchestration-only scope); the existing `simplify_polyline` unit tests already cover the algorithm itself.

## Configuration

| Key | Type | Default | Range |
|---|---|---|---|
| `simplify.tolerance_mm` | float | 0.15 | 0–2 |
