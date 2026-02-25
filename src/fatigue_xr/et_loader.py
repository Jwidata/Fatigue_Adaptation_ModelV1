from __future__ import annotations

from pathlib import Path

import pandas as pd

from fatigue_xr.colnorm import CANONICAL_COLUMNS, build_column_map
from fatigue_xr.logging_utils import get_logger


def load_et_xlsx(path: Path) -> pd.DataFrame:
    logger = get_logger(__name__)
    df = pd.read_excel(path, engine="openpyxl")

    mapping = build_column_map([str(col) for col in df.columns])
    df = df.rename(columns=mapping)

    present = [col for col in CANONICAL_COLUMNS if col in df.columns]
    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        logger.warning("Missing canonical ET columns", extra={"missing": missing})

    return df[present]
