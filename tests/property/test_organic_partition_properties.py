"""Property tests for the Organic Region Partition stage (ADR-003)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mysterycbn.model.records import LabelMap, Palette, PaletteColor, Provenance
from mysterycbn.stages.graph.components import build_region_graph, label_components
from mysterycbn.stages.graph.organic_partition import organic_partition_regions, stage_seed

PROV = Provenance("regions", "1.0.0", "0" * 64, "1" * 64)

# L* values chosen to stay above SKIP_DARK_LAB_L_THRESHOLD (organic_partition.py)
# so these fixture colors are never mistaken for a source image's own
# near-black outline stroke -- that skip-dark behavior has its own dedicated
# test (test_organic_partition_folds_dark_outline_region below).
_LABS = [(30.0, 0.0, 0.0), (50.0, 0.0, 0.0), (70.0, 0.0, 0.0), (95.0, 0.0, 0.0)]
PAL4 = Palette(
    colors=tuple(PaletteColor.from_lab(i, lab, 100) for i, lab in enumerate(_LABS)),
    provenance=PROV,
)


def _boundary_identity(component_map: np.ndarray, edges: tuple, regions: tuple) -> None:
    """Same double-entry crack identity ``components.py``'s own build asserts,
    re-checked against the organic stage's rebuilt output."""
    h, w = component_map.shape
    internal = int((component_map[:, :-1] != component_map[:, 1:]).sum()) + int(
        (component_map[:-1, :] != component_map[1:, :]).sum()
    )
    total = sum(w_len for *_, w_len in edges)
    assert total == internal
    border = 2 * (h + w)
    assert sum(r.perimeter_px for r in regions) == border + 2 * total


def _single_component_per_region(component_map: np.ndarray) -> None:
    """Every output region id, isolated as a mask, is exactly one 4-connected
    blob -- the planar "one region == one connected face" invariant topology
    extraction relies on."""
    n = int(component_map.max()) + 1
    for rid in range(n):
        mask = (component_map == rid).astype(np.int32)
        labeled = label_components(mask)
        # background (non-rid pixels) all share id 0 in a 2-valued input, so
        # the region itself must be the *other* single component.
        region_ids = set(labeled[component_map == rid].tolist())
        assert len(region_ids) == 1


@settings(max_examples=25, deadline=None)
@given(st.integers(20, 40), st.integers(20, 40), st.integers(0, 2**31 - 1))
def test_organic_partition_invariants(h: int, w: int, seed: int) -> None:
    labels = np.random.default_rng(seed).integers(0, 2, (h, w)).astype(np.int32)
    graph = build_region_graph(LabelMap(labels=labels, provenance=PROV), PAL4)

    new_graph, filler_ids, render_filler_ids = organic_partition_regions(
        graph,
        PAL4,
        mode="streamline",
        min_area_px=20.0,
        seed_density_px=15.0,
        rim_px=1.0,
        warp_px=2.0,
        noise_scale_px=5.0,
        fold_a_min_px=5.0,
        warp_seed=stage_seed(seed),
    )

    # Total area conserved; no sub-floor region (folding is authoritative).
    assert sum(r.area_px for r in new_graph.regions) == labels.size
    assert all(r.area_px >= 5.0 for r in new_graph.regions)

    # Label inheritance: every output region's label matches some original
    # region's label (organic partitioning never invents or reassigns a
    # palette index outside the source set).
    original_labels = {r.label for r in graph.regions}
    assert all(r.label in original_labels for r in new_graph.regions)

    _boundary_identity(new_graph.component_map, new_graph.edges, new_graph.regions)
    _single_component_per_region(new_graph.component_map)

    # render_filler_ids is always a subset of filler_ids (core excludes rim).
    assert render_filler_ids <= filler_ids
    assert all(0 <= i < len(new_graph.regions) for i in filler_ids)


@settings(max_examples=15, deadline=None)
@given(st.integers(20, 35), st.integers(20, 35), st.integers(0, 2**31 - 1))
def test_organic_partition_is_deterministic(h: int, w: int, seed: int) -> None:
    labels = np.random.default_rng(seed).integers(0, 2, (h, w)).astype(np.int32)
    graph = build_region_graph(LabelMap(labels=labels, provenance=PROV), PAL4)

    kwargs = dict(
        mode="streamline",
        min_area_px=20.0,
        seed_density_px=15.0,
        rim_px=1.0,
        warp_px=2.0,
        noise_scale_px=5.0,
        fold_a_min_px=5.0,
        warp_seed=stage_seed(seed),
    )
    g1, f1, rf1 = organic_partition_regions(graph, PAL4, **kwargs)
    g2, f2, rf2 = organic_partition_regions(graph, PAL4, **kwargs)

    assert np.array_equal(g1.component_map, g2.component_map)
    assert g1.regions == g2.regions
    assert g1.edges == g2.edges
    assert f1 == f2
    assert rf1 == rf2


@settings(max_examples=15, deadline=None)
@given(st.integers(20, 35), st.integers(20, 35), st.integers(0, 2**31 - 1))
def test_organic_partition_min_area_gate_is_a_true_passthrough(h: int, w: int, seed: int) -> None:
    """A ``min_area_px`` above every region's area must leave the graph
    unchanged in shape (still re-derived/re-stamped, but identical
    component_map and region set) -- the gate is a real no-op, not merely a
    smaller effect."""
    labels = np.random.default_rng(seed).integers(0, 2, (h, w)).astype(np.int32)
    graph = build_region_graph(LabelMap(labels=labels, provenance=PROV), PAL4)

    huge_floor = float(labels.size + 1)
    new_graph, filler_ids, render_filler_ids = organic_partition_regions(
        graph,
        PAL4,
        mode="streamline",
        min_area_px=huge_floor,
        seed_density_px=15.0,
        rim_px=1.0,
        warp_px=2.0,
        noise_scale_px=5.0,
        fold_a_min_px=0.0,
        warp_seed=stage_seed(seed),
    )
    assert np.array_equal(new_graph.component_map, graph.component_map)
    assert [r.label for r in new_graph.regions] == [r.label for r in graph.regions]
    assert filler_ids == frozenset()
    assert render_filler_ids == frozenset()


@settings(max_examples=15, deadline=None)
@given(st.integers(20, 35), st.integers(20, 35), st.integers(0, 2**31 - 1))
def test_organic_partition_island_label_inheritance(h: int, w: int, seed: int) -> None:
    """With islands enabled, every region -- including any carved island --
    still carries a label present in the original graph's label set (ADR-003
    decision: islands always inherit their parent's palette label)."""
    labels = np.random.default_rng(seed).integers(0, 2, (h, w)).astype(np.int32)
    graph = build_region_graph(LabelMap(labels=labels, provenance=PROV), PAL4)
    original_labels = {r.label for r in graph.regions}

    new_graph, _, _ = organic_partition_regions(
        graph,
        PAL4,
        mode="streamline",
        min_area_px=20.0,
        seed_density_px=15.0,
        rim_px=1.0,
        warp_px=2.0,
        noise_scale_px=5.0,
        island_probability=0.9,
        island_min_area_px=5.0,
        fold_a_min_px=2.0,
        warp_seed=stage_seed(seed),
    )
    assert all(r.label in original_labels for r in new_graph.regions)
    _boundary_identity(new_graph.component_map, new_graph.edges, new_graph.regions)


def test_organic_partition_skips_background_by_default() -> None:
    """A large page-border-touching background region must stay a single,
    untouched region -- only the interior "subject" region is organic-
    partitioned (regression test: without this, a flat background was
    shattered into hundreds of small cells, reading as visual noise and
    making a subject's real silhouette look doubled where many new organic-
    cell boundaries ran close beside it)."""
    h, w = 60, 60
    labels = np.zeros((h, w), dtype=np.int32)
    labels[10:50, 10:50] = 1  # interior "subject" block; label 0 touches every page edge
    graph = build_region_graph(LabelMap(labels=labels, provenance=PROV), PAL4)
    background_id = next(
        r.region_id for r in graph.regions if r.area_px == max(x.area_px for x in graph.regions)
    )

    new_graph, filler_ids, _ = organic_partition_regions(
        graph,
        PAL4,
        mode="streamline",
        min_area_px=200.0,
        seed_density_px=100.0,
        rim_px=0.0,
        warp_px=3.0,
        noise_scale_px=8.0,
        fold_a_min_px=5.0,
        warp_seed=stage_seed(0),
    )

    # The background's palette label must still appear on exactly one region
    # (never subdivided), while the subject's label appears on many.
    background_label = graph.regions[background_id].label
    subject_label = next(r.label for r in graph.regions if r.region_id != background_id)
    by_label: dict[int, int] = {}
    for r in new_graph.regions:
        by_label[r.label] = by_label.get(r.label, 0) + 1
    assert by_label[background_label] == 1
    assert by_label[subject_label] > 1
    assert not any(
        r.label == background_label and r.region_id in filler_ids for r in new_graph.regions
    )


def test_organic_partition_folds_dark_outline_region() -> None:
    """A thin, ring-shaped near-black region (simulating a source image's own
    pre-drawn cartoon outline stroke, which has real width and so quantizes
    into its own region) must be folded into a neighbor before partitioning,
    not left as its own standalone region -- otherwise organic-partitioning
    the regions on either side of the ring still traces both of the ring's
    edges as real silhouette boundaries, reading as a doubled outline
    (regression test for the artifact found on a real cartoon photo)."""
    h, w = 40, 40
    labels = np.zeros((h, w), dtype=np.int32)  # label 0: background
    labels[5:35, 5:35] = 1  # label 1: subject fill
    labels[5:35, 5:7] = 2  # label 2: a thin near-black "outline" strip along one edge
    dark_labs = [(30.0, 0.0, 0.0), (70.0, 0.0, 0.0), (2.0, 0.0, 0.0), (95.0, 0.0, 0.0)]
    dark_pal = Palette(
        colors=tuple(PaletteColor.from_lab(i, lab, 100) for i, lab in enumerate(dark_labs)),
        provenance=PROV,
    )
    graph = build_region_graph(LabelMap(labels=labels, provenance=PROV), dark_pal)
    assert any(r.label == 2 for r in graph.regions)  # the dark strip exists pre-fold

    new_graph, _, _ = organic_partition_regions(
        graph,
        dark_pal,
        mode="streamline",
        min_area_px=50.0,
        seed_density_px=40.0,
        rim_px=0.0,
        warp_px=2.0,
        noise_scale_px=5.0,
        fold_a_min_px=5.0,
        warp_seed=stage_seed(0),
    )

    # The dark label must be gone entirely -- folded into a neighbor, not
    # left as its own (even organic-partitioned) region.
    assert not any(r.label == 2 for r in new_graph.regions)
    assert sum(r.area_px for r in new_graph.regions) == labels.size
    _boundary_identity(new_graph.component_map, new_graph.edges, new_graph.regions)
