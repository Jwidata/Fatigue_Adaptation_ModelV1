from __future__ import annotations

import re


CANONICAL_COLUMNS = [
    "MEDIA_ID",
    "MEDIA_NAME",
    "CNT",
    "TIME",
    "TIMETICK",
    "FPOGX",
    "FPOGY",
    "FPOGS",
    "FPOGD",
    "FPOGID",
    "FPOGV",
    "BPOGX",
    "BPOGY",
    "BPOGV",
    "CX",
    "CY",
    "CS",
    "USER",
    "LPCX",
    "LPCY",
    "LPD",
    "LPS",
    "LPV",
    "RPCX",
    "RPCY",
    "RPD",
    "RPS",
    "RPV",
    "BKID",
    "BKDUR",
    "BKPMIN",
    "AOI",
]


def normalize_col(name: str) -> str:
    upper = name.upper()
    return re.sub(r"[^A-Z0-9]+", "", upper)


def build_column_map(df_columns: list[str]) -> dict[str, str]:
    canonical_set = set(CANONICAL_COLUMNS)
    mapping: dict[str, str] = {}

    for raw_col in df_columns:
        normalized = normalize_col(str(raw_col))
        if normalized.startswith("TIMETICK"):
            mapping[raw_col] = "TIMETICK"
            continue
        if normalized.startswith("TIME"):
            mapping[raw_col] = "TIME"
            continue
        if normalized in canonical_set:
            mapping[raw_col] = normalized

    return mapping
