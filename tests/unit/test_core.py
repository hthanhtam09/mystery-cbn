"""Tests for config, pipeline framework, and context plumbing."""

import pytest

from mysterycbn.core.config import Difficulty, EngineConfig, PageConfig
from mysterycbn.core.errors import ConfigError, StageError
from mysterycbn.core.pipeline import FunctionStage, Pipeline
from mysterycbn.core.types import PipelineContext


class TestConfig:
    def test_defaults_valid(self):
        cfg = EngineConfig()
        assert cfg.quantize.n_colors == 18
        assert cfg.page.content_width_mm > 0

    def test_frozen(self):
        cfg = EngineConfig()
        with pytest.raises(Exception):
            cfg.debug = True  # type: ignore[misc]

    def test_presets(self):
        easy = EngineConfig.preset("easy")
        hard = EngineConfig.preset(Difficulty.HARD)
        assert easy.quantize.n_colors < hard.quantize.n_colors
        assert easy.regions.min_region_mm > hard.regions.min_region_mm

    def test_preset_overrides_win(self):
        cfg = EngineConfig.preset("easy", quantize={"n_colors": 7})
        assert cfg.quantize.n_colors == 7
        assert cfg.quantize.min_delta_e == 10.0  # rest of preset intact

    def test_margins_must_leave_content(self):
        with pytest.raises(ConfigError):
            PageConfig(width_mm=100, height_mm=100, margin_mm=60)

    def test_page_point_conversion(self):
        page = PageConfig(width_mm=25.4, height_mm=50.8, margin_mm=0)
        assert page.width_pt == pytest.approx(72.0)
        assert page.height_pt == pytest.approx(144.0)


class TestPipeline:
    def test_runs_in_order_and_times(self):
        order = []
        stages = [
            FunctionStage("a", lambda ctx: order.append("a"), provides=("image",)),
            FunctionStage("b", lambda ctx: order.append("b"), requires=("image",)),
        ]

        def set_image(ctx):
            order.append("a")
            import numpy as np

            ctx.image = np.zeros((4, 4, 3), dtype="float32")

        stages[0] = FunctionStage("a", set_image, provides=("image",))
        ctx = Pipeline(stages).run(PipelineContext(config=EngineConfig()))
        assert order == ["a", "b"]
        assert [t.name for t in ctx.trace.timings] == ["a", "b"]

    def test_rejects_misordered_stages_at_build_time(self):
        stages = [FunctionStage("b", lambda ctx: None, requires=("label_map",))]
        with pytest.raises(ConfigError, match="label_map"):
            Pipeline(stages)

    def test_wraps_exceptions_with_stage_name(self):
        def boom(ctx):
            raise ValueError("kapow")

        pipeline = Pipeline([FunctionStage("boom", boom)])
        with pytest.raises(StageError, match=r"\[boom\] kapow"):
            pipeline.run(PipelineContext(config=EngineConfig()))

    def test_detects_unfulfilled_provides(self):
        pipeline = Pipeline([FunctionStage("liar", lambda ctx: None, provides=("palette",))])
        with pytest.raises(StageError, match="did not provide"):
            pipeline.run(PipelineContext(config=EngineConfig()))
