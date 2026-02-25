from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fatigue_xr.adaptation import AdaptationEngine
from fatigue_xr.features import window_features
from fatigue_xr.logging_utils import get_logger


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def load_et_from_manifest(
    manifest_path: Path, participant_id: str, condition: str, session_id: str
) -> pd.DataFrame:
    logger = get_logger(__name__)
    participant_id = participant_id.strip()
    condition = condition.strip().lower()
    session_id = session_id.strip()

    manifest_df = pd.read_parquet(manifest_path)
    subset = manifest_df[
        (manifest_df["participant_id"] == participant_id)
        & (manifest_df["condition"].str.lower() == condition)
    ]

    candidates = [
        session_id,
        f"{condition}__{session_id}",
        f"{condition.upper()}__{session_id}",
    ]

    matches = subset[subset["session_id"].isin(candidates)]
    if matches.empty:
        suffix_matches = subset[subset["session_id"].str.endswith(session_id)]
        if not suffix_matches.empty:
            preferred = suffix_matches[
                suffix_matches["session_id"].str.startswith(f"{condition}__")
            ]
            if len(preferred) == 1:
                matches = preferred
            elif len(suffix_matches) == 1:
                matches = suffix_matches
            else:
                options = suffix_matches["session_id"].tolist()
                raise ValueError(
                    "Ambiguous session_id match. Candidates: "
                    + ", ".join(options)
                )

    if matches.empty:
        available = subset["session_id"].tolist()[:10]
        raise ValueError(
            "No matching ET file found. "
            f"participant_id={participant_id}, condition={condition}, "
            f"session_id={session_id}. "
            f"Available sessions: {', '.join(available)}"
        )

    if len(matches) > 1:
        options = matches["session_id"].tolist()
        raise ValueError("Multiple matching ET files found: " + ", ".join(options))

    et_path = Path(matches.iloc[0]["out_path"])
    logger.info("Selected ET file", extra={"path": str(et_path)})
    return pd.read_parquet(et_path)


def predict_score(bundle: dict, feature_row: dict) -> float:
    feature_columns = list(bundle["feature_columns"])
    pipeline = bundle["pipeline"]
    row = {col: feature_row.get(col, np.nan) for col in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        return float(proba[0][1])
    if hasattr(pipeline, "decision_function"):
        score = pipeline.decision_function(X)
        return float(1.0 / (1.0 + np.exp(-score[0])))
    return float("nan")


def run_replay(
    participant_id: str,
    condition: str,
    session_id: str,
    model_path: Path,
    manifest_path: Path,
    window_len_sec: float = 10.0,
    stride_sec: float = 1.0,
    speed: float = 10.0,
    max_steps: int | None = None,
    out_csv: Path | None = None,
) -> Path:
    logger = get_logger(__name__)
    logger.info("Replay start", extra={"participant_id": participant_id})

    df = load_et_from_manifest(manifest_path, participant_id, condition, session_id)
    if df.empty or "time_sec" not in df.columns:
        raise ValueError("ET file is empty or missing time_sec.")

    bundle = load_model_bundle(model_path)
    time_series = pd.to_numeric(df["time_sec"], errors="coerce")
    t0 = float(time_series.min(skipna=True))
    t1 = float(time_series.max(skipna=True))

    engine = AdaptationEngine()
    results: list[dict[str, Any]] = []
    step = 0
    window_start = t0

    while window_start + window_len_sec <= t1:
        if max_steps is not None and step >= max_steps:
            break

        window_end = window_start + window_len_sec
        mask = (time_series >= window_start) & (time_series < window_end)
        window_df = df.loc[mask].copy()
        feats = window_features(window_df, window_len_sec)
        score = predict_score(bundle, feats)
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
                "n_samples_window": feats.get("n_samples_window"),
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
    logger.info("Replay finished", extra={"out_csv": str(out_path)})
    return out_path
