from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fatigue_xr.adaptation import AdaptationEngine
from fatigue_xr.features import window_features


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def select_et_file(
    manifest_df: pd.DataFrame,
    participant_id: str,
    condition: str,
    session_id: str,
) -> Path:
    matches = manifest_df[
        (manifest_df["participant_id"] == participant_id)
        & (manifest_df["condition"] == condition)
        & (manifest_df["session_id"] == session_id)
    ]

    if matches.empty:
        raise ValueError("No matching ET file found in manifest.")
    if len(matches) > 1:
        raise ValueError("Multiple matching ET files found in manifest.")
    return Path(matches.iloc[0]["out_path"])


def run_replay(
    participant_id: str,
    condition: str,
    session_id: str,
    model_path: Path,
    manifest_path: Path,
    window_len_sec: float = 10.0,
    stride_sec: float = 1.0,
    speed: float = 1.0,
    max_steps: int | None = None,
    out_csv: Path | None = None,
) -> Path:
    manifest_df = pd.read_parquet(manifest_path)
    et_path = select_et_file(manifest_df, participant_id, condition, session_id)

    df = pd.read_parquet(et_path)
    if df.empty or "time_sec" not in df.columns:
        raise ValueError("ET file is empty or missing time_sec.")

    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    feature_columns = list(bundle["feature_columns"])

    t0 = float(df["time_sec"].min())
    t1 = float(df["time_sec"].max())
    engine = AdaptationEngine()

    results: list[dict[str, Any]] = []
    step = 0
    window_start = t0

    while window_start + window_len_sec <= t1:
        if max_steps is not None and step >= max_steps:
            break

        window_end = window_start + window_len_sec
        window_df = df[(df["time_sec"] >= window_start) & (df["time_sec"] < window_end)]

        feats = window_features(window_df, window_len_sec)
        features_row = build_feature_row(feats, feature_columns)
        score = predict_score(pipeline, features_row)
        adaptation = engine.update(window_end, score)

        results.append(
            {
                "participant_id": participant_id,
                "condition": condition,
                "session_id": session_id,
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                "score": score,
                "state": adaptation["state"],
                "action": adaptation["action"],
                "action_changed": adaptation["action_changed"],
                "pupil_valid_frac": feats.get("pupil_valid_frac"),
                "gaze_valid_frac": feats.get("gaze_valid_frac"),
                "blink_rate_per_min": feats.get("blink_rate_per_min"),
            }
        )

        step += 1
        window_start += stride_sec
        if speed > 0:
            time.sleep(stride_sec / speed)

    out_path = out_csv
    if out_path is None:
        out_path = Path(
            "reports"
        ) / f"replay_{participant_id}_{condition}_{session_id}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    return out_path


def build_feature_row(
    feats: dict[str, Any], feature_columns: list[str]
) -> pd.DataFrame:
    row = {col: feats.get(col, np.nan) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def predict_score(pipeline, X: pd.DataFrame) -> float:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        return float(proba[0][1])
    if hasattr(pipeline, "decision_function"):
        score = pipeline.decision_function(X)
        return float(1.0 / (1.0 + np.exp(-score[0])))
    return float("nan")
