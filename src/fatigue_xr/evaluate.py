from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fatigue_xr.modeling import load_features
from fatigue_xr.reporting import render_markdown_table, write_markdown


def evaluate_saved_model(
    features_path: Path, model_path: Path, report_path: Path
) -> None:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    X_df, y, _, _ = load_features(features_path)
    X_eval = X_df[feature_columns]

    y_pred = pipeline.predict(X_eval)
    y_score = get_scores(pipeline, X_eval)
    metrics = compute_metrics(y, y_pred, y_score)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report_path, build_report_lines(metrics))


def get_scores(pipeline, X) -> np.ndarray | None:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        return proba[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X)
    return None


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is None:
        metrics["roc_auc"] = float("nan")
    else:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def build_report_lines(metrics: dict[str, float]) -> list[str]:
    lines = [
        "# Model Evaluation (Descriptive)",
        "",
        "This evaluation uses the full dataset and is not leak-safe.",
        "",
        "## Metrics",
        "",
    ]

    rows = [
        ["accuracy", f"{metrics['accuracy']:.4f}"],
        ["precision", f"{metrics['precision']:.4f}"],
        ["recall", f"{metrics['recall']:.4f}"],
        ["f1", f"{metrics['f1']:.4f}"],
        ["roc_auc", f"{metrics['roc_auc']:.4f}"],
    ]
    lines.extend(render_markdown_table(["Metric", "Value"], rows))
    return lines
