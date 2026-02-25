from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from fatigue_xr.config import PROCESSED_DIR


MODALITIES = ("et", "nback", "drt", "nasatlx", "unknown")


@dataclass(frozen=True)
class DatasetFile:
    participant_id: str
    rel_path: str
    abs_path: str
    ext: str
    modality: str
    condition: str
    session_id: str
    file_size_bytes: int
    mtime_iso: str


def build_dataset_index(raw_root: Path) -> pd.DataFrame:
    rows = [row.__dict__ for row in iter_dataset_files(raw_root)]
    df = pd.DataFrame(rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / "dataset_index.parquet"
    csv_path = PROCESSED_DIR / "dataset_index.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return df


def iter_dataset_files(raw_root: Path) -> Iterable[DatasetFile]:
    participant_roots = [
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name.lower().startswith("id")
    ]
    participant_roots.sort(key=lambda path: path.name.lower())

    for participant_root in participant_roots:
        for file_path in participant_root.rglob("*"):
            if not file_path.is_file():
                continue
            yield build_file_row(raw_root, participant_root, file_path)


def build_file_row(
    raw_root: Path, participant_root: Path, file_path: Path
) -> DatasetFile:
    stat = file_path.stat()
    ext = file_path.suffix.lower()
    rel_path = file_path.relative_to(raw_root).as_posix()
    abs_path = str(file_path.resolve())

    modality = infer_modality(file_path)
    condition = infer_condition(file_path, modality)
    session_id = infer_session_id(participant_root, file_path)

    return DatasetFile(
        participant_id=participant_root.name,
        rel_path=rel_path,
        abs_path=abs_path,
        ext=ext,
        modality=modality,
        condition=condition,
        session_id=session_id,
        file_size_bytes=stat.st_size,
        mtime_iso=datetime.fromtimestamp(stat.st_mtime).isoformat(),
    )


def infer_modality(file_path: Path) -> str:
    path_lower = file_path.as_posix().lower()

    if "et" in path_lower or "eye" in path_lower:
        return "et"
    if "nback" in path_lower or "n-back" in path_lower:
        return "nback"
    if "drt" in path_lower:
        return "drt"
    if "tlx" in path_lower or "nasa" in path_lower:
        return "nasatlx"
    return "unknown"


def infer_condition(file_path: Path, modality: str) -> str:
    path_lower = file_path.as_posix().lower()

    if "single" in path_lower:
        return "single"
    if "dual" in path_lower:
        return "dual"
    if modality == "drt":
        return "dual"
    return "unknown"


def infer_session_id(participant_root: Path, file_path: Path) -> str:
    stem = file_path.stem
    session_hint = infer_session_hint(participant_root, file_path)
    if session_hint:
        return f"{session_hint}__{stem}"
    return stem


def infer_session_hint(participant_root: Path, file_path: Path) -> str:
    ignore_tokens = ("et", "eye", "nback", "n-back", "drt", "tlx", "nasa")
    for parent in file_path.parents:
        if parent == participant_root:
            break
        name_lower = parent.name.lower()
        if any(token in name_lower for token in ignore_tokens):
            continue
        if not parent.name:
            continue
        return normalize_hint(parent.name)
    return ""


def normalize_hint(value: str) -> str:
    normalized = value.strip().replace(" ", "-")
    cleaned = "".join(
        char for char in normalized if char.isalnum() or char in {"-", "_"}
    )
    return cleaned.lower()
