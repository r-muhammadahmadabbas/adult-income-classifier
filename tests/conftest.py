"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest


RAW_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    """Small deterministic frame covering the raw 14 columns."""
    rng = np.random.default_rng(42)
    n = 240
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "workclass": rng.choice(["Private", "Self-emp-not-inc", "Local-gov"], n),
            "fnlwgt": rng.integers(10_000, 400_000, n),
            "education": rng.choice(["HS-grad", "Bachelors", "Masters"], n),
            "education_num": rng.integers(6, 17, n),
            "marital_status": rng.choice(["Married-civ-spouse", "Never-married"], n),
            "occupation": rng.choice(["Exec-managerial", "Other-service", "Sales"], n),
            "relationship": rng.choice(["Husband", "Not-in-family"], n),
            "race": rng.choice(["White", "Black", "Asian-Pac-Islander"], n),
            "sex": rng.choice(["Male", "Female"], n),
            "capital_gain": rng.choice([0, 0, 0, 5000, 12000], n),
            "capital_loss": rng.choice([0, 0, 1500], n),
            "hours_per_week": rng.integers(10, 65, n),
            "native_country": rng.choice(["United-States", "Mexico"], n),
        }
    )
