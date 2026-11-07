# Adult Income Classifier

[![CI](https://github.com/your-org/adult-income-classifier/actions/workflows/ci.yml/badge.svg)](../../actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-ready machine-learning service that predicts whether a person earns **more than \$50K per year** from U.S. Census demographic and employment data. The system is deliberately tuned for **precision**, because the business use case is outbound outreach: contacting someone who is *not* a high earner wastes money.

---

## Business objective

| Goal | Metric | Result |
| --- | --- | --- |
| Minimise wasted outreach | **Precision** | **0.9695** |
| Keep a meaningful coverage floor | Recall | 0.3036 |
| Rank high earners reliably | ROC AUC | 0.9213 |
| Calibrated probabilities | Brier score | 0.0918 |
| Decision threshold | Precision-first | 0.832 |

At threshold `0.832`, **96.95 % of contacted people truly earn > \$50K**, with only **15 false alarms** on the untouched test set (versus 303 at the default `0.5` threshold).

---

## Architecture

```text
Raw Census row (14 features)
        │
        ▼
engineer_features()          # +7 domain features → 21 columns
        │
        ▼
ColumnTransformer
 ├─ numeric:   median impute → StandardScaler
 └─ categorical: mode impute → OneHotEncoder   (~122 features)
        │
        ▼
SelectKBest(mutual_info_classif, k=30)
        │
        ▼
HistGradientBoostingClassifier
        │
        ▼
predict_proba() ──► business threshold (0.832) ──► >50K / ≤50K
```

The *entire* preprocessing chain lives inside a single scikit-learn `Pipeline`, so training and inference cannot drift apart. The shipped artifact is calibrated with 5-fold isotonic regression before the threshold is selected.

---

## Project structure

```text
adult-income-classifier/
├── .github/workflows/ci.yml     # Lint + tests on Python 3.10–3.12
├── data/                        # Data instructions (raw data is not committed)
├── docs/figures/                # Evaluation plots
├── models/
│   ├── income_pipeline.joblib   # Deployable pipeline
│   └── threshold.json           # Threshold + hold-out metrics
├── scripts/
│   └── run_training.py          # Thin wrapper around the CLI
├── src/income_classifier/
│   ├── config.py                # Paths, seeds, column contracts
│   ├── data.py                  # Load / clean / split
│   ├── features.py              # Domain feature engineering
│   ├── pipeline.py              # Preprocessing + model pipelines
│   ├── train.py                 # Tuning, calibration, thresholding
│   ├── evaluate.py              # Metrics + business summary
│   ├── predict.py               # Production inference
│   └── cli.py                   # `income-classifier` commands
└── tests/                       # Fast, offline unit tests
```

---

## Quick start

```bash
git clone https://github.com/your-org/adult-income-classifier.git
cd adult-income-classifier

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
pytest
```

### Predict with the shipped model

```bash
income-classifier predict --json '{
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
  "native_country": "United-States"
}'
```

```json
{
  "probability": 0.9971,
  "prediction": 1,
  "income_band": ">50K",
  "threshold": 0.832
}
```

### Python API

```python
import pandas as pd
from income_classifier import IncomePredictor

model = IncomePredictor()
results = model.predict(pd.DataFrame([row]))
```

### Retrain from scratch

```bash
income-classifier train --classifier hgb --iterations 50
income-classifier evaluate
```

`train` performs a `RandomizedSearchCV` search over the precision metric, calibrates the winner with isotonic regression, selects the precision-maximising threshold subject to `recall >= 0.30`, and writes fresh artifacts to `models/`.

---

## Model development journey

| Stage | Model | Validation precision |
| --- | --- | --- |
| Baseline | Majority class | 0.0000 |
| Baseline | Rule: `education_num >= 13` | 0.4844 |
| Week 1 | Logistic Regression | 0.7428 |
| Week 2 | HistGradientBoosting + CV | 0.7758 |
| Week 3 | Tuned HGB + isotonic calibration | **0.9695** |

Key engineering decisions:

- **`log_capital_gain`** compresses a 0–99,999 skew into a usable range.
- **`edu_hours_interaction`** captures the “educated + hard-working” segment.
- **Mutual-information feature selection** drops the noisy `fnlwgt` sampling weight while keeping the strongest 30 signals.
- **Isotonic calibration** improves the Brier score from `0.0949` to `0.0918`.
- **Business threshold 0.832** trades recall for precision, which is correct for paid outreach.

### Feature importance (permutation, hold-out sample)

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | `marital_status` | 0.0524 |
| 2 | `capital_gain` | 0.0502 |
| 3 | `education_num` | 0.0352 |
| 4 | `age` | 0.0222 |
| 5 | `capital_loss` | 0.0132 |

See `docs/figures/feature-importance.png` for the full chart.

---

## Evaluation

| Metric | Value |
| --- | --- |
| Precision | 0.9695 |
| Recall | 0.3036 |
| F1 | 0.4624 |
| Specificity | 0.9970 |
| ROC AUC | 0.9213 |
| Brier score | 0.0918 |
| Confusion matrix | TN=4930, FP=15, FN=1092, TP=476 |

```bash
income-classifier evaluate
```

---

## Testing

```bash
pytest -q
ruff check src tests
```

The test suite is fully offline: it builds synthetic Census-shaped frames, fits small pipelines, and validates feature engineering, threshold selection and artifact loading.

---

## Docker

```bash
docker build -t adult-income-classifier .
docker run --rm adult-income-classifier evaluate
```

---

## Reproducibility

- Fixed `random_state=42` everywhere.
- Pin-compatible ranges in `pyproject.toml` / `requirements.txt`.
- The raw dataset is downloaded from the canonical UCI URL.
- `models/threshold.json` records the exact hold-out metrics behind the shipped threshold.

---

## License

MIT. The UCI Adult dataset is distributed under its own terms; see the
[UCI repository](https://archive.ics.uci.edu/ml/datasets/adult) for details.
