"""Command-line interface for training, evaluation and inference."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import MODEL_PATH, THRESHOLD_PATH
from .data import load_dataset
from .evaluate import business_summary, classification_metrics, confusion_counts
from .predict import IncomePredictor
from .train import train_and_save


def cmd_train(args: argparse.Namespace) -> None:
    X_train, X_test, y_train, y_test = load_dataset(args.data)
    info = train_and_save(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier=args.classifier,
        n_iter=args.iterations,
    )
    print(json.dumps(info, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved threshold metadata to {THRESHOLD_PATH}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    predictor = IncomePredictor(threshold=args.threshold)
    _, X_test, _, y_test = load_dataset(args.data)
    probabilities = predictor.predict_proba(X_test)
    predictions = [int(p >= predictor.threshold) for p in probabilities]
    metrics = classification_metrics(y_test, predictions, probabilities)
    counts = confusion_counts(y_test, predictions)
    print(json.dumps({"metrics": metrics, "confusion": counts, "business": business_summary(counts)}, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    predictor = IncomePredictor(threshold=args.threshold)
    df = pd.read_csv(args.input) if args.input else pd.DataFrame([json.loads(args.json)])
    print(predictor.predict_frame(df).to_json(orient="records", indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="income-classifier",
        description="Train, evaluate and run the UCI Adult income classifier.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Tune, calibrate and save a new model.")
    train.add_argument("--classifier", choices=["hgb", "rf", "lr"], default="hgb")
    train.add_argument("--iterations", type=int, default=30)
    train.add_argument("--data", default=None, help="Optional local CSV path.")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="Evaluate the saved model on a hold-out split.")
    evaluate.add_argument("--data", default=None)
    evaluate.add_argument("--threshold", type=float, default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    predict = sub.add_parser("predict", help="Predict one JSON row or a CSV file.")
    predict.add_argument("--json", default=None, help="Single-row JSON object.")
    predict.add_argument("--input", default=None, help="CSV file with raw columns.")
    predict.add_argument("--threshold", type=float, default=None)
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
