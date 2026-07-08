"""Unit tests for the synthetic fixture generators."""

from __future__ import annotations

from benchmarks.framework.fixtures import (
    available_fixture_ids,
    fixture_manifest,
    load_fixture,
    load_full_ladder,
    load_smoke_fixtures,
)


def test_every_fixture_loads_deterministically() -> None:
    for fixture_id in available_fixture_ids():
        a = load_fixture(fixture_id)
        b = load_fixture(fixture_id)
        assert a.content_hash == b.content_hash
        assert a.labels.min() >= 0
        assert a.labels.max() < a.n_colors


def test_degenerate_fixture_is_truly_single_color() -> None:
    fx = load_fixture("F-degen-1")
    assert fx.labels.max() == 0
    assert fx.n_colors == 1


def test_unknown_fixture_raises() -> None:
    try:
        load_fixture("F-does-not-exist")
    except KeyError as exc:
        assert "unknown fixture" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_smoke_ladder_is_two_fixtures() -> None:
    assert len(load_smoke_fixtures()) == 2


def test_full_ladder_covers_every_fixture() -> None:
    assert {fx.fixture_id for fx in load_full_ladder()} == set(available_fixture_ids())


def test_manifest_hashes_match_loaded_fixtures() -> None:
    manifest = fixture_manifest()
    for fixture_id in available_fixture_ids():
        fx = load_fixture(fixture_id)
        assert manifest[fixture_id]["content_hash"] == fx.content_hash
