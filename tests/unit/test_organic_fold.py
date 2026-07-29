"""Unit tests for the shared region-fold mechanics (_organic_common)."""

from __future__ import annotations

import numpy as np

from mysterycbn.stages.graph._organic_common import fold_regions_where


def test_fold_absorbs_into_the_largest_neighbour_not_the_lowest_id() -> None:
    # A dark stroke (label 2) runs down the middle. Region id 0 is the tiny
    # scan-order-first blob on the left touching only its top pixel; the big
    # mass on the right (id 2) is what the stroke actually borders. Folding by
    # lowest id hands the stroke to the small blob -- the penguin-eye-outline
    # -> pink-scoop bug; folding by area gives it to the mass.
    rows = []
    for r in range(8):
        # col 0: tiny blob (label 1) only on the first row, else the mass
        left = 1 if r == 0 else 3
        rows.append([left, left, 2, 3, 3, 3, 3, 3])
    cmap_labels = np.array(rows, dtype=np.int32)
    ids = np.zeros_like(cmap_labels)
    # Build ids so the tiny blob is id 0 (scan order) and the mass is a later id.
    next_id = 0
    seen: dict[tuple[int, int], int] = {}
    for r in range(cmap_labels.shape[0]):
        for c in range(cmap_labels.shape[1]):
            lab = int(cmap_labels[r, c])
            key = (lab, 0 if (lab == 1) else 1)
            if key not in seen:
                seen[key] = next_id
                next_id += 1
            ids[r, c] = seen[key]
    labels = [0] * next_id
    for (lab, _), rid in seen.items():
        labels[rid] = lab

    stroke_id = seen[(2, 1)]
    out_map, out_labels = fold_regions_where(
        ids,
        labels,
        should_fold=lambda _a, cur, _c: np.array(cur) == 2,
    )
    assert 2 not in out_labels  # the stroke is gone as its own region
    # Every pixel that was the stroke now carries the big mass's label (3),
    # never the tiny blob's (1).
    stroke_pixels = ids == stroke_id
    new_labels_at_stroke = np.array(out_labels)[out_map[stroke_pixels]]
    assert set(new_labels_at_stroke.tolist()) == {3}


def test_fold_prefers_the_big_mass_over_a_strip_it_runs_alongside() -> None:
    # A stroke (label 2) runs the full width. Along its whole underside sits a
    # narrow strip (label 1) -- the longest shared boundary -- while the big
    # mass above (label 3) touches only part of it. Folding by longest
    # boundary hands the stroke to the strip, which is how an elephant's mouth
    # line swallowed its teeth and fused them into one unnumberable ribbon.
    rows = []
    for r in range(15):
        if r < 10:  # the mass, over 3/4 of the width; filler beyond it
            rows.append([3] * 30 + [4] * 10)
        elif r < 12:  # the stroke
            rows.append([2] * 40)
        else:  # the narrow strip, full width
            rows.append([1] * 40)
    labels_img = np.array(rows, dtype=np.int32)

    ids = np.zeros_like(labels_img)
    seen: dict[int, int] = {}
    for lab in (3, 4, 2, 1):
        seen[lab] = len(seen)
    for lab, rid in seen.items():
        ids[labels_img == lab] = rid
    labels = [0] * len(seen)
    for lab, rid in seen.items():
        labels[rid] = lab

    stroke_pixels = labels_img == 2
    out_map, out_labels = fold_regions_where(
        ids, labels, should_fold=lambda _a, cur, _c: np.array(cur) == 2
    )
    landed = set(np.array(out_labels)[out_map[stroke_pixels]].tolist())
    assert landed == {3}, f"stroke should join the mass (3), not the strip (1); got {landed}"
