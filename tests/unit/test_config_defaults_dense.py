"""The "dense" preset carries the mystery-style organic tiling config."""

from mysterycbn.app.config_defaults import difficulty_preset


def test_dense_preset_enables_full_page_organic_tiling() -> None:
    overlay = difficulty_preset("dense")
    organic = overlay["organic"]
    assert organic["enabled"] is True
    assert organic["skip_background"] is False
    assert organic["mode"] == "streamline"
    assert organic["seed_density_mm2"] == 120.0
    assert organic["min_area_mm2"] == 40.0
    assert organic["warp_strength_mm"] == 4.0
    assert organic["noise_scale_mm"] == 22.0
    assert organic["ribbon_elongation"] == 0.7
    assert organic["min_inner_diameter_mm"] == 3.2
    assert overlay["split"] == {"enabled": False}
    assert overlay["preprocess"] == {"max_working_px": 2400}
    assert overlay["simplify"] == {"tolerance_mm": 0.2}
    assert overlay["bezier"] == {"fit_error_mm": 0.22, "corner_angle_deg": 70.0}
    assert overlay["validate"] == {
        "fidelity_min_agreement": 0.93,
        "fidelity_min_agreement_filler": 0.85,
    }


def test_other_presets_leave_organic_alone() -> None:
    for preset in ("easy", "medium", "hard"):
        assert "organic" not in difficulty_preset(preset)
