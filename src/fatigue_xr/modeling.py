from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {
    "participant_id",
    "condition",
    "session_id",
    "window_start_sec",
    "window_end_sec",
}


def load_features(
    features_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    df = pd.read_parquet(features_path)
    df = df[df["condition"].isin(["single", "dual"])].copy()

    y = (df["condition"] == "dual").astype(int)
    groups = df["participant_id"].astype(str)

    drop_cols = [col for col in META_COLUMNS if col in df.columns]
    X_df = df.drop(columns=drop_cols)
    X_df = X_df.select_dtypes(include=[np.number])

    feature_cols = list(X_df.columns)
    drop_all_nan = [col for col in feature_cols if X_df[col].isna().all()]
    if drop_all_nan:
        X_df = X_df.drop(columns=drop_all_nan)

    return X_df, y, groups, drop_all_nan


def make_feature_sets(X_df: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(X_df.columns)
    pupil = [col for col in columns if col.startswith("pupil_")]
    blink = [col for col in columns if col.startswith("blink_")]
    all_cols = columns

    return {
        "pupil_only": pupil,
        "pupil_blink": pupil + blink,
        "all": all_cols,
    }


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {"logreg": logreg, "rf": rf}
