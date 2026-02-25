from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from fatigue_xr.modeling import build_models, load_features, make_feature_sets
from fatigue_xr.reporting import render_markdown_table, write_markdown


def train_and_select_best(
    features_path: Path,
    out_model_path: Path,
    report_path: Path,
    cm_png_path: Path,
    random_state: int = 42,
    test_size: float = 0.2,
) -> None:
    X_df, y, groups, dropped_features = load_features(features_path)
    feature_sets = make_feature_sets(X_df)
    models = build_models(random_state=random_state)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X_df, y, groups))

    X_train = X_df.iloc[train_idx]
    y_train = y.iloc[train_idx]
    g_train = groups.iloc[train_idx]
    X_test = X_df.iloc[test_idx]
    y_test = y.iloc[test_idx]
    g_test = groups.iloc[test_idx]

    cv_results: list[dict[str, Any]] = []
    best_combo = None
    best_key = (float("-inf"), float("-inf"))

    for feature_set_name, feature_cols in feature_sets.items():
        if not feature_cols:
            continue
        for model_name, pipeline in models.items():
            metrics = cross_validate_grouped(
                pipeline, X_train[feature_cols], y_train, g_train
            )
            cv_results.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    **metrics,
                }
            )

            roc_mean = metrics["roc_auc_mean"]
            f1_mean = metrics["f1_mean"]
            if (roc_mean, f1_mean) > best_key:
                best_key = (roc_mean, f1_mean)
                best_combo = (feature_set_name, model_name)

    if best_combo is None:
        raise ValueError("No valid model configurations found.")

    best_feature_set, best_model_name = best_combo
    best_cols = feature_sets[best_feature_set]
    best_pipeline = models[best_model_name]
    best_pipeline.fit(X_train[best_cols], y_train)

    y_pred = best_pipeline.predict(X_test[best_cols])
    y_score = get_scores(best_pipeline, X_test[best_cols])
    holdout_metrics = compute_metrics(y_test, y_pred, y_score)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, cm_png_path)

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_columns": best_cols,
            "feature_set": best_feature_set,
            "model_name": best_model_name,
            "random_state": random_state,
        },
        out_model_path,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(
        report_path,
        build_report_lines(
            X_df=X_df,
            y=y,
            groups=groups,
            train_groups=sorted(set(g_train.tolist())),
            test_groups=sorted(set(g_test.tolist())),
            cv_results=cv_results,
            best_combo=best_combo,
            holdout_metrics=holdout_metrics,
            cm_png_path=cm_png_path,
            dropped_features=dropped_features,
        ),
    )

    export_feature_importance(
        best_pipeline,
        best_cols,
        best_model_name,
        report_path.parent / "feature_importance.csv",
    )


def cross_validate_grouped(pipeline, X, y, groups) -> dict[str, float]:
    gkf = GroupKFold(n_splits=5)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        y_score = get_scores(pipeline, X_val)
        fold_metrics.append(compute_metrics(y_val, y_pred, y_score))

    return aggregate_metrics(fold_metrics)


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


def aggregate_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    metrics = {}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        metrics[f"{key}_mean"] = float(np.nanmean(values))
        metrics[f"{key}_std"] = float(np.nanstd(values))
    return metrics


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["single", "dual"])
    ax.set_yticklabels(["single", "dual"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report_lines(
    X_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    train_groups: list[str],
    test_groups: list[str],
    cv_results: list[dict[str, Any]],
    best_combo: tuple[str, str],
    holdout_metrics: dict[str, float],
    cm_png_path: Path,
    dropped_features: list[str],
) -> list[str]:
    lines = [
        "# Model Training Report",
        "",
        f"Dataset windows: {len(X_df)}",
        f"Feature columns: {len(X_df.columns)}",
        f"Participants: {groups.nunique()}",
        "",
        "## Train/Test Split",
        "",
        f"Train participants ({len(train_groups)}): {', '.join(train_groups)}",
        f"Test participants ({len(test_groups)}): {', '.join(test_groups)}",
        "",
        "## Cross-Validation Results (Train Only)",
        "",
    ]

    rows = []
    for result in cv_results:
        rows.append(
            [
                result["feature_set"],
                result["model"],
                format_mean_std(result, "accuracy"),
                format_mean_std(result, "precision"),
                format_mean_std(result, "recall"),
                format_mean_std(result, "f1"),
                format_mean_std(result, "roc_auc"),
            ]
        )
    lines.extend(
        render_markdown_table(
            ["Feature Set", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            rows,
        )
    )

    lines.extend(
        [
            "",
            "## Selected Model",
            "",
            f"Best combo: {best_combo[0]} + {best_combo[1]}",
            "",
            "## Dropped Features",
            "",
        ]
    )

    if dropped_features:
        lines.extend([f"- {feature}" for feature in dropped_features])
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Holdout Test Metrics",
            "",
        ]
    )

    holdout_rows = [
        ["accuracy", f"{holdout_metrics['accuracy']:.4f}"],
        ["precision", f"{holdout_metrics['precision']:.4f}"],
        ["recall", f"{holdout_metrics['recall']:.4f}"],
        ["f1", f"{holdout_metrics['f1']:.4f}"],
        ["roc_auc", f"{holdout_metrics['roc_auc']:.4f}"],
    ]
    lines.extend(render_markdown_table(["Metric", "Value"], holdout_rows))

    lines.extend(
        [
            "",
            f"Confusion matrix image: `{cm_png_path.name}`",
        ]
    )

    return lines


def format_mean_std(result: dict[str, Any], key: str) -> str:
    mean = result.get(f"{key}_mean", float("nan"))
    std = result.get(f"{key}_std", float("nan"))
    return f"{mean:.4f} ± {std:.4f}"


def export_feature_importance(
    pipeline, feature_columns: list[str], model_name: str, out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = pipeline.named_steps.get("model")
    if model is None:
        return

    if model_name == "rf" and hasattr(model, "feature_importances_"):
        importances = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False)
        importances.to_csv(out_path, index=False)
        return

    if model_name == "logreg" and hasattr(model, "coef_"):
        coef = model.coef_[0]
        df = pd.DataFrame(
            {
                "feature": feature_columns,
                "coef": coef,
                "abs_coef": np.abs(coef),
            }
        ).sort_values(by="abs_coef", ascending=False)
        df = df.drop(columns=["abs_coef"])
        df.to_csv(out_path, index=False)
