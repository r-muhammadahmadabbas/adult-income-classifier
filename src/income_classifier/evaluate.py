"""Evaluation metrics and reports."""
from __future__ import annotations

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Return the standard metric bundle used throughout the project."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        metrics["brier"] = float(brier_score_loss(y_true, y_proba))
    return metrics


def confusion_counts(y_true, y_pred) -> Dict[str, int]:
    """Return TN/FP/FN/TP as a dictionary."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def business_summary(counts: Dict[str, int]) -> Dict[str, float]:
    """Translate confusion counts into the outreach-focused business view."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    contacted = tp + fp
    return {
        "people_contacted": contacted,
        "true_high_earners": tp,
        "wasted_outreach": fp,
        "missed_high_earners": fn,
        "precision_pct": round(100 * tp / contacted, 2) if contacted else 0.0,
    }
