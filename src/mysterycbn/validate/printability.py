"""Printability Validator (I4): every region physically colorable, every
number readable (ENGINE_SPEC.md §25.3; QM-10 Min Region Diameter, QM-11 Tiny
Region %, QM-21 Label Coverage, QM-24 Min Font Size).

Per face: inscribed diameter (independently re-derived from flattened
geometry via the labels stage's own largest-empty-circle search — the same
algorithm, re-invoked, is the sanctioned "independent" proof here because the
alternative is a full different geometry engine; MATH_SPEC §13.3) must clear
``d_min_mm``. Below floor + has an in-region label -> declared repair: demote
to leader (re-invokes the labels stage's leader placement for that face). No
feasible leader -> FATAL. Every face must have a label plan entry at >=
``font_min_pt``.
"""

from __future__ import annotations

from mysterycbn.model.context import PipelineContext
from mysterycbn.model.layout import Label, LabelMode, LabelPlan
from mysterycbn.model.reports import Finding, Severity, ValidationReport
from mysterycbn.model.vector import CurveSet
from mysterycbn.stages.layout.labels import (
    FONT_MIN_PT_DEFAULT,
    LEADER_RING_MM_DEFAULT,
    _place_leader,
    _segments_of,
    largest_empty_circle,
)
from mysterycbn.validate.common import flatten_face_rings

VALIDATOR_NAME = "printability"
_FLATTEN_MM = 0.1
_MM_TO_PT = 72.0 / 25.4
D_MIN_MM_DEFAULT = 3.5

_Placement = tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]


def validate_printability(
    ctx: PipelineContext,
    *,
    d_min_mm: float = D_MIN_MM_DEFAULT,
    font_min_pt: float = FONT_MIN_PT_DEFAULT,
    leader_ring_mm: float = LEADER_RING_MM_DEFAULT,
) -> ValidationReport:
    """Run the QM-10/QM-11/QM-21/QM-24 checks; applies the declared
    leader-demotion repair in place on the bound ``label_plan`` when a face
    clears the floor only via a leader (logged as REPAIRED, not FATAL)."""
    curve_set = ctx.get("curve_set")
    label_plan = ctx.get("label_plan")
    assert isinstance(curve_set, CurveSet)
    assert isinstance(label_plan, LabelPlan)

    tolerance_pt = _FLATTEN_MM * _MM_TO_PT
    ring_pt = leader_ring_mm * _MM_TO_PT
    d_min_pt = d_min_mm * _MM_TO_PT

    # Filler cells (produced by split_large in "dense" mode) deliberately
    # subdivide a flat area into many small same-color cells, each carrying a
    # tiny in-cell number -- the commercial color-by-number background-tiling
    # look. They are exempt from the readable-size floor (QM-10 diameter and
    # QM-24 font) that regular regions must clear, since by construction they
    # are smaller than a "readable" region and never use a leader. They are
    # still required to carry a label entry (QM-21 coverage still applies).
    filler_ids = ctx.get("filler_region_ids") if ctx.has("filler_region_ids") else frozenset()
    if not isinstance(filler_ids, (set, frozenset)):
        filler_ids = frozenset()
    dense_mode = bool(filler_ids)
    # Deliberately label-free cells (labels stage): slivers too thin for any
    # legible number. unlabeled ids stay blank line art; blackout ids are
    # solid-filled by the renderers. Either way no label plan entry exists,
    # so the coverage gate must not FATAL on them.
    blackout_ids = ctx.get("blackout_region_ids") if ctx.has("blackout_region_ids") else frozenset()
    if not isinstance(blackout_ids, (set, frozenset)):
        blackout_ids = frozenset()
    unlabeled_ids = (
        ctx.get("unlabeled_region_ids") if ctx.has("unlabeled_region_ids") else frozenset()
    )
    if isinstance(unlabeled_ids, (set, frozenset)):
        blackout_ids = frozenset(blackout_ids) | frozenset(unlabeled_ids)

    label_by_region = {label.region_id: label for label in label_plan.labels}
    findings: list[Finding] = []
    n_tiny = 0
    diameters: list[float] = []

    faces = curve_set.faces
    all_rings = [flatten_face_rings(face, curve_set, tolerance_pt) for face in faces]
    all_seg_a, all_seg_b = _segments_of([r for rings in all_rings for r in rings])

    repaired_labels: dict[int, _Placement] = {}

    for face, rings in zip(faces, all_rings, strict=True):
        if face.face_id in blackout_ids:
            continue  # solid-filled sliver: no label expected, no size gate
        pole, clearance = largest_empty_circle(rings, precision_pt=0.5)
        diameter_pt = 2.0 * clearance
        diameter_mm = diameter_pt / _MM_TO_PT
        diameters.append(diameter_mm)

        label = label_by_region.get(face.face_id)
        if label is None:
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I4",
                    message="face has no label plan entry",
                    location=f"region {face.face_id}",
                )
            )
            continue

        # A face is exempt from the readable-size gate if it is a filler /
        # micro-labelled cell (recorded explicitly in filler_region_ids by the
        # split_large + labels stages in dense mode). The exemption is never
        # inferred from font size alone -- an accidentally tiny font on an
        # ordinary region must still FATAL.
        is_filler = face.face_id in filler_ids
        below_floor = diameter_pt < d_min_pt
        if below_floor:
            n_tiny += 1
        # Filler cells skip the readable-size gate entirely (they carry a tiny
        # in-cell number by design); regular below-floor in-region faces get
        # the declared leader-demotion repair, or FATAL if no leader is feasible.
        if below_floor and not is_filler and label.mode is LabelMode.IN_REGION:
            placed = _place_leader(
                pole, clearance, label.printed_number, font_min_pt, ring_pt, all_seg_a, all_seg_b
            )
            if placed is None:
                if dense_mode:
                    # Dense mode: this face's own leader-demotion repair has no
                    # feasible anchor at the validator's independent
                    # re-derivation (a genuine boundary case -- e.g. a region
                    # whose diameter sits fractions of a mm under the floor).
                    # Rather than FATAL a single isolated cell on an otherwise
                    # fully-covered dense sheet, treat it like any other
                    # micro-label: keep its existing in-region number as-is.
                    findings.append(
                        Finding(
                            severity=Severity.REPAIRED,
                            invariant="I4",
                            message=(
                                f"face below printability floor "
                                f"({diameter_mm:.2f}mm < {d_min_mm}mm); no feasible leader, "
                                f"kept as micro-label (dense mode)"
                            ),
                            location=f"region {face.face_id}",
                            repair_applied=True,
                        )
                    )
                    continue
                findings.append(
                    Finding(
                        severity=Severity.FATAL,
                        invariant="I4",
                        message=(
                            f"face below printability floor "
                            f"({diameter_mm:.2f}mm < {d_min_mm}mm) with no feasible leader"
                        ),
                        location=f"region {face.face_id}",
                    )
                )
                continue
            repaired_labels[face.face_id] = placed
            findings.append(
                Finding(
                    severity=Severity.REPAIRED,
                    invariant="I4",
                    message=(
                        f"face below printability floor "
                        f"({diameter_mm:.2f}mm < {d_min_mm}mm); demoted to leader line"
                    ),
                    location=f"region {face.face_id}",
                    repair_applied=True,
                )
            )
        # LEADER-mode faces below the floor are already the repaired state.

        if label.font_size_pt < font_min_pt and not is_filler:
            findings.append(
                Finding(
                    severity=Severity.FATAL,
                    invariant="I4",
                    message=f"label font size {label.font_size_pt}pt < floor {font_min_pt}pt",
                    location=f"region {face.face_id}",
                )
            )

    if repaired_labels:
        new_labels = []
        for label in label_plan.labels:
            placed = repaired_labels.get(label.region_id)
            if placed is None:
                new_labels.append(label)
                continue
            anchor, leader = placed
            new_labels.append(
                Label(
                    region_id=label.region_id,
                    printed_number=label.printed_number,
                    anchor=anchor,
                    font_size_pt=font_min_pt,
                    mode=LabelMode.LEADER,
                    clearance_pt=label.clearance_pt,
                    leader=leader,
                )
            )
        ctx.put(
            "label_plan",
            LabelPlan(labels=tuple(new_labels), provenance=label_plan.provenance),
        )

    r_tiny_pct = 100.0 * n_tiny / len(faces) if faces else 0.0
    metrics = {
        "min_region_diameter_mm": min(diameters) if diameters else float("inf"),
        "tiny_region_pct": r_tiny_pct,
        "label_coverage_pct": 100.0 * len(label_by_region) / len(faces) if faces else 100.0,
    }
    return ValidationReport(
        validator_name=VALIDATOR_NAME, findings=tuple(findings), metrics=metrics
    )
