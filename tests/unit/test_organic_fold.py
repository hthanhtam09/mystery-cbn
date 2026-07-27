"""Unit tests for the shared region-fold mechanics (_organic_common)."""

from __future__ import annotations

import numpy as np

from mysterycbn.stages.graph._organic_common import fold_regions_where


def test_fold_absorbs_into_the_longest_shared_boundary_not_the_lowest_id() -> None:
    # A dark stroke (label 2) runs down the middle. Region id 0 is the tiny
    # scan-order-first blob on the left touching only its top pixel; the big
    # mass on the right (id 2) is what the stroke actually borders. Folding by
    # lowest id hands the stroke to the small blob -- the penguin-eye-outline
    # -> pink-scoop bug; folding by longest boundary gives it to the mass.
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
