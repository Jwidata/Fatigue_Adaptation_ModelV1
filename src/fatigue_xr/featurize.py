from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fatigue_xr.config import FEATURES_DIR
from fatigue_xr.features import window_features
from fatigue_xr.logging_utils import get_logger


def featurize_all(
    manifest_path: Path,
    out_path: Path,
    window_len_sec: float = 10.0,
    stride_sec: float = 1.0,
    limit_files: int | None = None,
) -> pd.DataFrame:
    logger = get_logger(__name__)
    manifest = pd.read_parquet(manifest_path)
    if limit_files is not None:
        manifest = manifest.head(limit_files)

    feature_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        in_path = Path(row["out_path"])
        participant_id = row["participant_id"]
        condition = row.get("condition", "unknown")
        session_id = row.get("session_id", "unknown")

        logger.info("Featurizing ET file", extra={"path": str(in_path)})
        df = pd.read_parquet(in_path)
        if df.empty or "time_sec" not in df.columns:
            continue

        t0 = float(df["time_sec"].min())
        t1 = float(df["time_sec"].max())
        if not np.isfinite(t0) or not np.isfinite(t1):
            continue

        window_start = t0
        while window_start + window_len_sec <= t1:
            window_end = window_start + window_len_sec
            window_df = df[(df["time_sec"] >= window_start) & (df["time_sec"] < window_end)]

            feats = window_features(window_df, window_len_sec)
            feats.update(
                {
                    "participant_id": participant_id,
                    "condition": condition,
                    "session_id": session_id,
                }
            )
            feature_rows.append(feats)
            window_start += stride_sec

    features_df = pd.DataFrame(feature_rows)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out_path, index=False)
    csv_path = FEATURES_DIR / "window_features.csv"
    features_df.to_csv(csv_path, index=False)

    log_feature_qa(logger, features_df)
    return features_df


def log_feature_qa(logger, features_df: pd.DataFrame) -> None:
    total_windows = int(len(features_df))
    if total_windows == 0:
        logger.info("Featurize QA", extra={"total_windows": 0})
        return

    pupil_frac = pd.to_numeric(features_df.get("pupil_valid_frac"), errors="coerce")
    gaze_frac = pd.to_numeric(features_df.get("gaze_valid_frac"), errors="coerce")

    pupil_low = float((pupil_frac < 0.2).mean() * 100.0) if not pupil_frac.empty else float("nan")
    gaze_low = float((gaze_frac < 0.2).mean() * 100.0) if not gaze_frac.empty else float("nan")

    logger.info(
        "Featurize QA",
        extra={
            "total_windows": total_windows,
            "pupil_low_pct": pupil_low,
            "gaze_low_pct": gaze_low,
        },
    )
