import json
from pathlib import Path

import pytest

from income_classifier.config import MODEL_PATH, THRESHOLD_PATH
from income_classifier.predict import IncomePredictor

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not THRESHOLD_PATH.exists(),
    reason="Shipped model artifacts are not present.",
)


def test_shipped_artifact_loads_and_scores():
    predictor = IncomePredictor()
    row = {
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 200000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 5000,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States",
    }
    import pandas as pd

    result = predictor.predict(pd.DataFrame([row]))[0]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["income_band"] in {">50K", "<=50K"}
    assert 0.0 <= result["threshold"] <= 1.0
