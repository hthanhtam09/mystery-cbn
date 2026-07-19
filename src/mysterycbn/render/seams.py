"""Shared arc classification: filler seams (thin) vs real boundaries (bold).

An arc is a *filler seam* -- drawn at the fine stroke -- only when BOTH hold:

1. Every face referencing it is a filler cell (``render_filler_region_ids``
   from ``organic_partition``/``split_large``): it is an internal boundary
   inside a subdivided pattern, not a real region border.
2. Both of its sides carry the SAME printed palette number. When the whole
   page is organic-tiled (mystery style, ``skip_background`` off) every cell
   is a filler cell, so test 1 alone would demote the subject silhouette to
   the fine stroke too; a boundary between two different palette colors is a
   real silhouette edge regardless of filler status and stays bold.

Used by the SVG, PDF, and PNG renderers so all artifacts agree on the
bold-outline/fine-pattern hierarchy.
"""

from __future__ import annotations

from mysterycbn.model.vector import CurveSet


def arc_sides_and_faces(
    curve_set: CurveSet, number_of: dict[int, int]
) -> tuple[dict[int, tuple[int, int]], dict[int, list[int]]]:
    """Per-arc (left, right) printed numbers and touching face ids."""
    sides: dict[int, tuple[int, int]] = {}
    arc_faces: dict[int, list[int]] = {}
    for face in curve_set.faces:
        for walk in face.all_walks():
            for arc_id, rev in walk:
                left, right = sides.get(arc_id, (0, 0))
                printed = number_of.get(face.face_id, 0)
                sides[arc_id] = (printed, right) if not rev else (left, printed)
                arc_faces.setdefault(arc_id, []).append(face.face_id)
    return sides, arc_faces


def thin_seam_arc_ids(
    curve_set: CurveSet,
    number_of: dict[int, int],
    filler_ids: frozenset[int],
) -> frozenset[int]:
    """Arc ids to draw at the fine stroke (see module docstring)."""
    if not filler_ids:
        return frozenset()
    sides, arc_faces = arc_sides_and_faces(curve_set, number_of)
    thin: set[int] = set()
    for arc_id, faces in arc_faces.items():
        if not all(f in filler_ids for f in faces):
            continue
        left, right = sides.get(arc_id, (0, 0))
        if left == right:
            thin.add(arc_id)
    return frozenset(thin)
