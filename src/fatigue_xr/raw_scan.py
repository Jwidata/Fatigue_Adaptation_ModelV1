from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fatigue_xr.config import REPORTS_DIR

MODALITIES = ("ET", "NBACK", "DRT", "NASATLX", "UNKNOWN")


@dataclass(frozen=True)
class ParticipantScan:
    participant_id: str
    root: Path
    extension_counts: dict[str, int]
    modality_counts: dict[str, int]

    @property
    def total_files(self) -> int:
        return sum(self.extension_counts.values())


def scan_raw_root(raw_root: Path) -> list[ParticipantScan]:
    participants = [
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name.lower().startswith("id")
    ]
    participants.sort(key=lambda path: path.name.lower())

    return [scan_participant(path) for path in participants]


def scan_participant(participant_root: Path) -> ParticipantScan:
    extension_counts: dict[str, int] = {}
    modality_counts = {modality: 0 for modality in MODALITIES}

    for file_path in participant_root.rglob("*"):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower() or "<none>"
        extension_counts[extension] = extension_counts.get(extension, 0) + 1

        modality = infer_modality(file_path)
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    return ParticipantScan(
        participant_id=participant_root.name,
        root=participant_root,
        extension_counts=extension_counts,
        modality_counts=modality_counts,
    )


def infer_modality(file_path: Path) -> str:
    extension = file_path.suffix.lower()
    path_lower = file_path.as_posix().lower()

    if extension == ".xlsx" and ("et" in path_lower or "eye" in path_lower):
        return "ET"
    if extension == ".csv" and ("nback" in path_lower or "n-back" in path_lower):
        return "NBACK"
    if extension in {".xlsx", ".csv"} and "drt" in path_lower:
        return "DRT"
    if extension in {".xlsx", ".csv"} and (
        "tlx" in path_lower or "nasa" in path_lower
    ):
        return "NASATLX"
    return "UNKNOWN"


def print_participant_summary(participants: Iterable[ParticipantScan]) -> None:
    participants_list = list(participants)
    if not participants_list:
        print("No participant folders found.")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Raw Data Scan")
        table.add_column("Participant", justify="left")
        table.add_column("Total", justify="right")
        table.add_column("ET", justify="right")
        table.add_column("NBACK", justify="right")
        table.add_column("DRT", justify="right")
        table.add_column("NASATLX", justify="right")
        table.add_column("UNKNOWN", justify="right")
        table.add_column("Extensions", justify="left")

        for participant in participants_list:
            table.add_row(
                participant.participant_id,
                str(participant.total_files),
                str(participant.modality_counts.get("ET", 0)),
                str(participant.modality_counts.get("NBACK", 0)),
                str(participant.modality_counts.get("DRT", 0)),
                str(participant.modality_counts.get("NASATLX", 0)),
                str(participant.modality_counts.get("UNKNOWN", 0)),
                format_extension_counts(participant.extension_counts),
            )

        console.print(table)
    except Exception:
        header = (
            "Participant | Total | ET | NBACK | DRT | NASATLX | UNKNOWN | Extensions"
        )
        print(header)
        print("-" * len(header))
        for participant in participants_list:
            row = " | ".join(
                [
                    participant.participant_id,
                    str(participant.total_files),
                    str(participant.modality_counts.get("ET", 0)),
                    str(participant.modality_counts.get("NBACK", 0)),
                    str(participant.modality_counts.get("DRT", 0)),
                    str(participant.modality_counts.get("NASATLX", 0)),
                    str(participant.modality_counts.get("UNKNOWN", 0)),
                    format_extension_counts(participant.extension_counts),
                ]
            )
            print(row)


def write_markdown_report(
    raw_root: Path, participants: Iterable[ParticipantScan]
) -> Path:
    participants_list = list(participants)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "raw_scan.md"

    modality_totals = build_modality_totals(participants_list)
    missing_map = build_missing_modality_map(participants_list)

    lines: list[str] = [
        "# Raw Scan Report",
        "",
        f"Raw root: `{raw_root}`",
        "",
        f"Total participants: {len(participants_list)}",
        "",
        "## Modality Counts (Overall)",
        "",
        "| Modality | Count |",
        "| --- | ---: |",
    ]

    for modality in MODALITIES:
        lines.append(f"| {modality} | {modality_totals.get(modality, 0)} |")

    lines.extend(
        [
            "",
            "## Participants Missing Modalities",
            "",
            "Missing does not imply error; this is purely observational.",
            "",
        ]
    )

    for modality in ("ET", "NBACK", "DRT", "NASATLX"):
        missing_list = missing_map.get(modality, [])
        if missing_list:
            lines.append(f"- {modality}: {', '.join(missing_list)}")
        else:
            lines.append(f"- {modality}: None")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_modality_totals(participants: Iterable[ParticipantScan]) -> dict[str, int]:
    totals = {modality: 0 for modality in MODALITIES}
    for participant in participants:
        for modality, count in participant.modality_counts.items():
            totals[modality] = totals.get(modality, 0) + count
    return totals


def build_missing_modality_map(
    participants: Iterable[ParticipantScan],
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {modality: [] for modality in MODALITIES}
    for participant in participants:
        for modality in ("ET", "NBACK", "DRT", "NASATLX"):
            if participant.modality_counts.get(modality, 0) == 0:
                missing[modality].append(participant.participant_id)
    return missing


def format_extension_counts(extension_counts: dict[str, int]) -> str:
    if not extension_counts:
        return ""
    parts = [
        f"{extension}={count}"
        for extension, count in sorted(extension_counts.items())
    ]
    return " ".join(parts)
