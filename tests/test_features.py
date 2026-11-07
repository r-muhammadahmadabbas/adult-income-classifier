import numpy as np
import pandas as pd

from income_classifier.features import engineer_features


def test_engineer_features_adds_seven_columns(raw_frame):
    out = engineer_features(raw_frame)
    assert out.shape[1] == raw_frame.shape[1] + 7
    expected = {
        "has_capital_gain",
        "log_capital_gain",
        "has_capital_loss",
        "age_group",
        "hours_category",
        "higher_ed",
        "edu_hours_interaction",
    }
    assert expected.issubset(out.columns)


def test_capital_flags_and_log_are_correct(raw_frame):
    out = engineer_features(raw_frame)
    assert (out.loc[raw_frame["capital_gain"] > 0, "has_capital_gain"] == 1).all()
    assert (out.loc[raw_frame["capital_gain"] == 0, "has_capital_gain"] == 0).all()
    np.testing.assert_allclose(
        out["log_capital_gain"], np.log1p(raw_frame["capital_gain"])
    )


def test_age_and_hours_buckets_are_ordered(raw_frame):
    out = engineer_features(raw_frame)
    assert set(out["age_group"].cat.categories) == {"<25", "25-39", "40-54", "55-64", "65+"}
    assert set(out["hours_category"].cat.categories) == {"<20", "20-39", "40-49", "50+"}


def test_higher_education_threshold(raw_frame):
    out = engineer_features(raw_frame)
    assert (out.loc[raw_frame["education_num"] >= 14, "higher_ed"] == 1).all()
    assert (out.loc[raw_frame["education_num"] < 14, "higher_ed"] == 0).all()
