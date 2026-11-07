"""Production inference for the Adult Income classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import pandas as pd

from .config import DEFAULT_THRESHOLD, MODEL_PATH, THRESHOLD_PATH
from .features import engineer_features


def _register_unpickle_shim() -> None:
    """The shipped artifact was pickled with ``engineer_features`` in ``__main__``.

    Registering the package function under ``__main__`` lets joblib restore the
    FunctionTransformer without retraining the model.
    """
    import __main__

    if not hasattr(__main__, "engineer_features"):
        __main__.engineer_features = engineer_features


class IncomePredictor:
    """Load a trained pipeline and produce business-ready predictions."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold_path: Path = THRESHOLD_PATH,
        threshold: Optional[float] = None,
    ) -> None:
        _register_unpickle_shim()
        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.threshold = threshold if threshold is not None else self._load_threshold(threshold_path)

    @staticmethod
    def _load_threshold(path: Path) -> float:
        try:
            with path.open(encoding="utf-8") as fh:
                return float(json.load(fh)["optimal_threshold"])
        except (FileNotFoundError, KeyError, ValueError):
            return DEFAULT_THRESHOLD

    def predict_proba(self, df: pd.DataFrame) -> List[float]:
        return self.model.predict_proba(df)[:, 1].tolist()

    def predict(self, df: pd.DataFrame) -> List[Dict]:
        """Return probability, binary label and human-readable income band."""
        probabilities = self.predict_proba(df)
        results = []
        for prob in probabilities:
            label = int(prob >= self.threshold)
            results.append(
                {
                    "probability": round(float(prob), 4),
                    "prediction": label,
                    "income_band": ">50K" if label else "<=50K",
                    "threshold": self.threshold,
                }
            )
        return results

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict and attach results to a copy of the input frame."""
        out = df.copy()
        results = pd.DataFrame(self.predict(df))
        return pd.concat([out.reset_index(drop=True), results], axis=1)
