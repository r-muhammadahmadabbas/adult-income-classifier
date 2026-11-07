"""Domain feature engineering for the UCI Adult dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLS, NUMERIC_COLS

# Re-exported for convenience and for pickled-pipeline reconstruction.
__all__ = [
    "engineer_features",
    "NUMERIC_COLS",
    "CATEGORICAL_COLS",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seven domain-driven features to the raw 14-column dataframe.

    The function returns a copy containing all original columns plus:

    * ``has_capital_gain`` / ``has_capital_loss`` – presence flags
    * ``log_capital_gain`` – log1p of the heavily skewed capital gain
    * ``age_group`` – life-cycle bucket
    * ``hours_category`` – work-intensity bucket
    * ``higher_ed`` – Bachelor's degree or above
    * ``edu_hours_interaction`` – education years × weekly hours
    """
    out = df.copy()

    out["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    out["log_capital_gain"] = np.log1p(df["capital_gain"])
    out["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)

    out["age_group"] = pd.cut(
        df["age"],
        bins=[0, 24, 39, 54, 64, 100],
        labels=["<25", "25-39", "40-54", "55-64", "65+"],
    )
    out["hours_category"] = pd.cut(
        df["hours_per_week"],
        bins=[0, 19, 39, 49, 100],
        labels=["<20", "20-39", "40-49", "50+"],
    )

    out["higher_ed"] = (df["education_num"] >= 14).astype(int)
    out["edu_hours_interaction"] = df["education_num"] * df["hours_per_week"]

    return out
