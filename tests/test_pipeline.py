import numpy as np

from income_classifier.pipeline import build_pipeline
from income_classifier.train import find_optimal_threshold


def test_pipeline_fits_and_predicts(raw_frame):
    y = (raw_frame["education_num"] >= 14).astype(int)
    pipe = build_pipeline("hgb", k=10, classifier_params={"max_iter": 20})
    pipe.fit(raw_frame, y)
    proba = pipe.predict_proba(raw_frame)[:, 1]
    assert len(proba) == len(raw_frame)
    assert np.all((proba >= 0) & (proba <= 1))


def test_logistic_pipeline_supports_all_features(raw_frame):
    y = (raw_frame["capital_gain"] > 0).astype(int)
    pipe = build_pipeline("lr", k="all")
    pipe.fit(raw_frame, y)
    assert pipe.predict(raw_frame).shape[0] == len(raw_frame)


def test_threshold_selection_respects_recall_floor():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(0.5 * y + rng.normal(0.25, 0.15, 500), 0, 1)
    info = find_optimal_threshold(y, proba, min_recall=0.2)
    assert 0 <= info["optimal_threshold"] <= 1
    assert info["recall"] >= 0.2 or info["recall"] == 0
    assert set(["precision", "f1", "brier_score", "roc_auc"]).issubset(info)
