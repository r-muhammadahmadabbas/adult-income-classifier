"""Central configuration for the Adult Income classifier.

All paths are resolved relative to the project root so the project can be
cloned anywhere and run without editing source code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "income_pipeline.joblib"))
THRESHOLD_PATH = Path(os.getenv("THRESHOLD_PATH", MODELS_DIR / "threshold.json"))

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SELECT_K = 30

ADULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)

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
    "income",
]

FEATURE_COLUMNS = RAW_COLUMNS[:-1]
TARGET_COLUMN = "income"

NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "has_capital_gain",
    "log_capital_gain",
    "has_capital_loss",
    "higher_ed",
    "edu_hours_interaction",
]

CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "age_group",
    "hours_category",
]

DEFAULT_THRESHOLD = 0.832


def load_threshold() -> float:
    """Return the persisted business threshold, falling back to the default."""
    try:
        with THRESHOLD_PATH.open(encoding="utf-8") as fh:
            return float(json.load(fh)["optimal_threshold"])
    except (FileNotFoundError, KeyError, ValueError):
        return DEFAULT_THRESHOLD
