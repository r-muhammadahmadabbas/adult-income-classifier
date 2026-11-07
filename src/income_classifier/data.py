"""Dataset loading, cleaning and splitting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ADULT_URL, RANDOM_STATE, RAW_COLUMNS, TARGET_COLUMN, TEST_SIZE


def load_raw(source: Union[str, Path] = ADULT_URL) -> pd.DataFrame:
    """Load the UCI Adult dataset from a URL or a local CSV file.

    The raw file has no header, so column names are supplied explicitly.
    Rows containing ``?`` (missing values) are dropped, matching the
    original analysis.
    """
    read_kwargs = dict(
        names=RAW_COLUMNS,
        na_values=" ?",
        skipinitialspace=True,
    )
    if isinstance(source, (str, Path)) and Path(source).exists():
        df = pd.read_csv(source, **read_kwargs)
    else:
        df = pd.read_csv(str(source), **read_kwargs)

    return df.dropna().reset_index(drop=True)


def to_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into ``X`` and a binary ``y`` (>50K == 1)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = (df[TARGET_COLUMN] == ">50K").astype(int)
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Create a reproducible, stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def load_dataset(source: Union[str, Path] = ADULT_URL):
    """Convenience wrapper: load -> clean -> split."""
    return split_dataset(*to_features_and_target(load_raw(source)))
