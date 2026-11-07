"""Scikit-learn pipeline construction."""
from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_STATE, SELECT_K
from .features import engineer_features

CLASSIFIERS = {
    "hgb": lambda **kw: HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, **kw
    ),
    "rf": lambda **kw: RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, **kw
    ),
    "lr": lambda **kw: LogisticRegression(
        solver="liblinear", random_state=RANDOM_STATE, max_iter=1000, **kw
    ),
}


def build_preprocessor() -> ColumnTransformer:
    """Return the numeric + categorical preprocessing transformer."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(
    classifier: str = "hgb",
    *,
    k: Optional[int] = SELECT_K,
    classifier_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build a full ``engineer -> preprocess -> select -> classify`` pipeline.

    Parameters
    ----------
    classifier:
        One of ``"hgb"``, ``"rf"`` or ``"lr"``.
    k:
        Number of features kept by ``SelectKBest``. Use ``None`` or ``"all"``
        to keep every feature.
    classifier_params:
        Extra keyword arguments forwarded to the estimator constructor.
    """
    if classifier not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier {classifier!r}. Choose from {sorted(CLASSIFIERS)}"
        )

    steps = [
        ("engineer", FunctionTransformer(engineer_features)),
        ("preprocessor", build_preprocessor()),
    ]
    if k is not None and k != "all":
        steps.append(("select", SelectKBest(mutual_info_classif, k=k)))
    steps.append(("model", CLASSIFIERS[classifier](**(classifier_params or {}))))
    return Pipeline(steps)


def search_space(classifier: str) -> Dict[str, Any]:
    """Return a reasonable RandomizedSearchCV space for a classifier."""
    if classifier == "hgb":
        from scipy.stats import loguniform, randint

        return {
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_iter": randint(80, 400),
            "model__max_depth": [3, 5, 7, 10, None],
            "model__l2_regularization": loguniform(1e-3, 10),
            "model__min_samples_leaf": randint(5, 21),
        }
    if classifier == "rf":
        from scipy.stats import randint

        return {
            "model__n_estimators": randint(100, 301),
            "model__max_depth": [5, 10, 15, 20, None],
            "model__min_samples_leaf": randint(1, 9),
            "model__max_features": ["sqrt", "log2", 0.5],
        }
    if classifier == "lr":
        from scipy.stats import loguniform

        return {
            "model__penalty": ["l1", "l2"],
            "model__C": loguniform(1e-3, 1e3),
        }
    raise ValueError(f"Unknown classifier {classifier!r}")
