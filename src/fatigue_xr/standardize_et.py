from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fatigue_xr.logging_utils import get_logger


def standardize_et(df: pd.DataFrame, timetick_hz: int) -> pd.DataFrame:
    logger = get_logger(__name__)
    if "TIMETICK" not in df.columns:
        raise ValueError("TIMETICK column is required for standardization")

    out = df.copy()
    timetick = pd.to_numeric(out["TIMETICK"], errors="coerce")
    first_tick = timetick.dropna().iloc[0]
    time_sec = (timetick - first_tick) / float(timetick_hz)
    out["time_sec"] = time_sec
    out["sample_dt_sec"] = time_sec.diff()

    pupil_valid = build_valid_flag(out, ["LPV", "RPV"])
    gaze_valid = build_valid_flag(out, ["BPOGV", "FPOGV"])

    out["pupil_valid"] = pupil_valid
    out["gaze_valid"] = gaze_valid
    out["row_valid_for_pupil_features"] = pupil_valid
    out["row_valid_for_gaze_features"] = gaze_valid

    stats = sampling_stats(out)
    logger.info("ET sampling stats", extra=stats)
    return out


def build_valid_flag(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    series_list = []
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        series_list.append(values > 0.5)
    if not series_list:
        return pd.Series([False] * len(df), index=df.index)
    combined = series_list[0]
    for series in series_list[1:]:
        combined = combined | series
    return combined


def sampling_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "sample_dt_sec" in df.columns:
        dt = pd.to_numeric(df["sample_dt_sec"], errors="coerce")
    else:
        dt = pd.Series(dtype=float)

    finite_dt = dt[np.isfinite(dt)]
    mean_dt = float(finite_dt.mean()) if not finite_dt.empty else float("nan")
    std_dt = float(finite_dt.std()) if not finite_dt.empty else float("nan")
    approx_hz = float(1.0 / mean_dt) if mean_dt and np.isfinite(mean_dt) else float("nan")

    return {
        "mean_dt": mean_dt,
        "std_dt": std_dt,
        "approx_hz": approx_hz,
    }
