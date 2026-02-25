from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_pupil_signal(df: pd.DataFrame) -> pd.Series:
    left = pd.Series([np.nan] * len(df), index=df.index)
    right = pd.Series([np.nan] * len(df), index=df.index)

    if "LPD" in df.columns and "LPV" in df.columns:
        lpv = pd.to_numeric(df["LPV"], errors="coerce").fillna(0)
        lpd = pd.to_numeric(df["LPD"], errors="coerce")
        left = lpd.where(lpv > 0.5)

    if "RPD" in df.columns and "RPV" in df.columns:
        rpv = pd.to_numeric(df["RPV"], errors="coerce").fillna(0)
        rpd = pd.to_numeric(df["RPD"], errors="coerce")
        right = rpd.where(rpv > 0.5)

    pupil = pd.concat([left, right], axis=1)
    return pupil.mean(axis=1, skipna=True)


def choose_gaze_xy(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    if "BPOGX" in df.columns and "BPOGY" in df.columns:
        x = pd.to_numeric(df["BPOGX"], errors="coerce")
        y = pd.to_numeric(df["BPOGY"], errors="coerce")
        if "BPOGV" in df.columns:
            valid = pd.to_numeric(df["BPOGV"], errors="coerce").fillna(0) > 0.5
        else:
            valid = pd.Series([False] * len(df), index=df.index)
        return x, y, valid

    x = pd.to_numeric(df.get("FPOGX"), errors="coerce")
    y = pd.to_numeric(df.get("FPOGY"), errors="coerce")
    if "FPOGV" in df.columns:
        valid = pd.to_numeric(df["FPOGV"], errors="coerce").fillna(0) > 0.5
    else:
        valid = pd.Series([False] * len(df), index=df.index)
    return x, y, valid


def window_features(window_df: pd.DataFrame, window_len_sec: float) -> dict[str, Any]:
    features: dict[str, Any] = {}
    features["n_samples_window"] = int(len(window_df))

    if "time_sec" in window_df.columns and not window_df["time_sec"].empty:
        features["window_start_sec"] = float(window_df["time_sec"].iloc[0])
        features["window_end_sec"] = float(window_df["time_sec"].iloc[-1])
    else:
        features["window_start_sec"] = float("nan")
        features["window_end_sec"] = float("nan")

    pupil_diam = compute_pupil_signal(window_df)
    pupil_finite = pupil_diam[np.isfinite(pupil_diam)]
    total_rows = len(window_df) if len(window_df) else 0
    pupil_valid_frac = float(len(pupil_finite) / total_rows) if total_rows else float("nan")

    if not pupil_finite.empty:
        features.update(
            {
                "pupil_mean": float(pupil_finite.mean()),
                "pupil_std": float(pupil_finite.std()),
                "pupil_iqr": float(pupil_finite.quantile(0.75) - pupil_finite.quantile(0.25)),
                "pupil_p10": float(pupil_finite.quantile(0.10)),
                "pupil_p50": float(pupil_finite.quantile(0.50)),
                "pupil_p90": float(pupil_finite.quantile(0.90)),
            }
        )
        features["pupil_slope"] = linear_slope(window_df, pupil_diam)
    else:
        features.update(
            {
                "pupil_mean": float("nan"),
                "pupil_std": float("nan"),
                "pupil_iqr": float("nan"),
                "pupil_p10": float("nan"),
                "pupil_p50": float("nan"),
                "pupil_p90": float("nan"),
                "pupil_slope": float("nan"),
            }
        )
    features["pupil_valid_frac"] = pupil_valid_frac

    if "BKID" in window_df.columns:
        blink_ids = pd.to_numeric(window_df["BKID"], errors="coerce").fillna(0)
        unique_blinks = blink_ids[blink_ids > 0].unique().tolist()
        blink_count = int(len(unique_blinks))
        features["blink_count"] = blink_count
        features["blink_rate_per_min"] = (
            float(blink_count) / (window_len_sec / 60.0)
            if window_len_sec > 0
            else float("nan")
        )
        if "BKDUR" in window_df.columns:
            blink_dur = pd.to_numeric(window_df["BKDUR"], errors="coerce")
            blink_dur = blink_dur[blink_dur > 0]
            features["blink_dur_mean"] = (
                float(blink_dur.mean()) if not blink_dur.empty else float("nan")
            )
    else:
        features["blink_count"] = float("nan")
        features["blink_rate_per_min"] = float("nan")
        features["blink_dur_mean"] = float("nan")

    gaze_x, gaze_y, gaze_valid = choose_gaze_xy(window_df)
    gaze_valid = gaze_valid.fillna(False)
    valid_rows = window_df[gaze_valid]
    gaze_valid_frac = float(len(valid_rows) / total_rows) if total_rows else float("nan")

    if not valid_rows.empty:
        valid_x = gaze_x[gaze_valid]
        valid_y = gaze_y[gaze_valid]
        features["gaze_dispersion"] = float(valid_x.std() + valid_y.std())

        if "time_sec" in window_df.columns:
            time_sec = pd.to_numeric(window_df["time_sec"], errors="coerce")
            diffs = build_gaze_speed(valid_x, valid_y, time_sec[gaze_valid])
            features["gaze_speed_proxy"] = diffs
        else:
            features["gaze_speed_proxy"] = float("nan")
    else:
        features["gaze_dispersion"] = float("nan")
        features["gaze_speed_proxy"] = float("nan")

    features["gaze_valid_frac"] = gaze_valid_frac

    if "AOI" in window_df.columns:
        aoi_series = window_df["AOI"].dropna()
        if not aoi_series.empty:
            transitions = int((aoi_series != aoi_series.shift()).sum() - 1)
            transitions = max(transitions, 0)
            features["aoi_switch_rate_per_min"] = (
                float(transitions) / (window_len_sec / 60.0)
                if window_len_sec > 0
                else float("nan")
            )
            counts = aoi_series.value_counts(normalize=True)
            entropy = -float((counts * np.log2(counts)).sum()) if not counts.empty else float("nan")
            features["aoi_entropy"] = entropy
            features["aoi_top_frac"] = float(counts.max()) if not counts.empty else float("nan")
        else:
            features["aoi_switch_rate_per_min"] = float("nan")
            features["aoi_entropy"] = float("nan")
            features["aoi_top_frac"] = float("nan")
    else:
        features["aoi_switch_rate_per_min"] = float("nan")
        features["aoi_entropy"] = float("nan")
        features["aoi_top_frac"] = float("nan")

    return features


def linear_slope(df: pd.DataFrame, series: pd.Series) -> float:
    if "time_sec" not in df.columns:
        return float("nan")
    time_sec = pd.to_numeric(df["time_sec"], errors="coerce")
    valid = np.isfinite(series) & np.isfinite(time_sec)
    if valid.sum() < 2:
        return float("nan")
    x = time_sec[valid].to_numpy()
    y = series[valid].to_numpy()
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def build_gaze_speed(x: pd.Series, y: pd.Series, time_sec: pd.Series) -> float:
    if len(x) < 2:
        return float("nan")
    dx = x.diff().abs()
    dy = y.diff().abs()
    dt = time_sec.diff()
    valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) & (dt > 0)
    if valid.sum() == 0:
        return float("nan")
    speed = (dx[valid] + dy[valid]) / dt[valid]
    return float(speed.mean()) if not speed.empty else float("nan")
