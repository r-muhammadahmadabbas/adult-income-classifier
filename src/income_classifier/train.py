"""Training, tuning, calibration and threshold selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import (
    CV_FOLDS,
    MODEL_PATH,
    RANDOM_STATE,
    THRESHOLD_PATH,
)
from .pipeline import build_pipeline, search_space


def tune(
    X_train,
    y_train,
    classifier: str = "hgb",
    *,
    n_iter: int = 30,
    cv_folds: int = CV_FOLDS,
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV optimising precision."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        build_pipeline(classifier),
        search_space(classifier),
        n_iter=n_iter,
        scoring="precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score=np.nan,
    )
    search.fit(X_train, y_train)
    return search


def calibrate(
    estimator,
    X_train,
    y_train,
    *,
    method: str = "isotonic",
    cv_folds: int = CV_FOLDS,
) -> CalibratedClassifierCV:
    """Calibrate an already-fitted pipeline with cross-validated isotonic regression."""
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
        ),
        n_jobs=-1,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def find_optimal_threshold(
    y_true,
    y_proba,
    *,
    min_recall: float = 0.30,
) -> Dict[str, float]:
    """Pick the threshold that maximises precision subject to a recall floor."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]

    mask = (recall >= min_recall) & (precision > 0)
    if mask.any():
        idx = int(np.argmax(np.where(mask, precision, -1)))
    else:
        idx = int(np.argmax(precision))

    threshold = float(thresholds[idx])
    from sklearn.metrics import (
        brier_score_loss,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "optimal_threshold": threshold,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f1": float(f1_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "calibration_method": "isotonic",
        "calibration_cv_folds": CV_FOLDS,
    }


def save_artifacts(
    estimator,
    threshold_info: Dict,
    *,
    model_path: Path = MODEL_PATH,
    threshold_path: Path = THRESHOLD_PATH,
) -> Tuple[Path, Path]:
    """Persist the fitted pipeline and threshold metadata."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_path)
    with threshold_path.open("w", encoding="utf-8") as fh:
        json.dump(threshold_info, fh, indent=2)
    return model_path, threshold_path


def train_and_save(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    classifier: str = "hgb",
    n_iter: int = 30,
) -> Dict:
    """Full training workflow: tune -> calibrate -> threshold -> save."""
    search = tune(X_train, y_train, classifier=classifier, n_iter=n_iter)
    calibrated = calibrate(search.best_estimator_, X_train, y_train)
    proba = calibrated.predict_proba(X_val)[:, 1]
    threshold_info = find_optimal_threshold(y_val, proba)
    threshold_info["best_params"] = search.best_params_
    threshold_info["cv_precision"] = float(search.best_score_)

    save_artifacts(calibrated, threshold_info)
    return threshold_info
