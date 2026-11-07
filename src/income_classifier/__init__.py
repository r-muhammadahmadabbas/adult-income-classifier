"""Adult Income classifier package."""

from .predict import IncomePredictor
from .pipeline import build_pipeline

__all__ = ["IncomePredictor", "build_pipeline"]
__version__ = "1.0.0"
