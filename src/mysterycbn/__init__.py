"""Mystery color-by-number conversion engine."""

from .core.config import Difficulty, EngineConfig
from .core.errors import EngineError
from .core.pipeline import Pipeline
from .core.types import PipelineContext

__all__ = ["Difficulty", "EngineConfig", "EngineError", "Pipeline", "PipelineContext"]
__version__ = "0.1.0"
