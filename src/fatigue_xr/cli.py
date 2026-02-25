from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import typer

from fatigue_xr.config import (
    FEATURES_DIR,
    PROCESSED_DIR,
    RAW_ROOT_DEFAULT,
    REPORTS_DIR,
    TIMETICK_HZ,
)
from fatigue_xr.et_loader import load_et_xlsx
from fatigue_xr.evaluate import evaluate_saved_model
from fatigue_xr.featurize import featurize_all
from fatigue_xr.ingest import MODALITIES, build_dataset_index
from fatigue_xr.logging_utils import get_logger, log_event, setup_logging
from fatigue_xr.raw_scan import (
    print_participant_summary,
    scan_raw_root,
    write_markdown_report,
)
from fatigue_xr.reporting import render_markdown_table, write_markdown
from fatigue_xr.standardize_et import sampling_stats, standardize_et
from fatigue_xr.train import train_and_select_best

app = typer.Typer(add_completion=False)


@app.command("scan-raw")
def scan_raw(
    raw_root: Path = typer.Option(
        RAW_ROOT_DEFAULT,
        "--raw-root",
        help="Path to raw input root (default: Data)",
    ),
) -> None:
    """Scan raw eye-tracking data and report dataset structure."""
    setup_logging()
    logger = get_logger(__name__)

    raw_root = raw_root.expanduser()
    if not raw_root.exists():
        raise typer.BadParameter(f"Raw root does not exist: {raw_root}")

    log_event(logger, "scan_raw_start", raw_root=raw_root)
    participants = scan_raw_root(raw_root)
    print_participant_summary(participants)

    report_path = write_markdown_report(raw_root, participants)
    log_event(logger, "scan_raw_report_written", report_path=report_path)
    typer.echo(f"Wrote report to {report_path}")


@app.command("ingest")
def ingest(
    raw_root: Path = typer.Option(
        RAW_ROOT_DEFAULT,
        "--raw-root",
        help="Path to raw input root (default: Data)",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--no-overwrite",
        help="Overwrite existing index outputs",
    ),
) -> None:
    """Build dataset index and write a dataset inventory report."""
    setup_logging()
    logger = get_logger(__name__)

    raw_root = raw_root.expanduser()
    if not raw_root.exists():
        raise typer.BadParameter(f"Raw root does not exist: {raw_root}")

    parquet_path = PROCESSED_DIR / "dataset_index.parquet"
    csv_path = PROCESSED_DIR / "dataset_index.csv"
    if not overwrite and (parquet_path.exists() or csv_path.exists()):
        typer.echo("Index outputs already exist; run with --overwrite to refresh.")
        df = load_existing_index(parquet_path)
    else:
        log_event(logger, "ingest_start", raw_root=raw_root)
        df = build_dataset_index(raw_root)
        log_event(logger, "ingest_index_written", parquet_path=parquet_path)

    report_path = write_dataset_inventory(raw_root, df)
    typer.echo(f"Wrote report to {report_path}")


@app.command("peek")
def peek(
    modality: str = typer.Option(
        ...,
        "--modality",
        help="Modality to inspect (et, nback, drt, nasatlx, unknown)",
    ),
    n: int = typer.Option(2, "--n", help="Number of paths to show"),
) -> None:
    """Print example paths and condition values for a modality."""
    parquet_path = PROCESSED_DIR / "dataset_index.parquet"
    if not parquet_path.exists():
        raise typer.BadParameter("dataset_index.parquet not found; run ingest first.")

    df: pd.DataFrame = load_existing_index(parquet_path)
    modality_lower = modality.lower()
    if modality_lower not in MODALITIES:
        raise typer.BadParameter(f"Unknown modality: {modality}")

    subset = cast(pd.DataFrame, df[df["modality"] == modality_lower])
    rel_series = cast(pd.Series, subset["rel_path"])
    cond_series = cast(pd.Series, subset["condition"])
    rel_paths = rel_series.head(n).tolist()
    conditions = sorted(set(cond_series.dropna().tolist()))

    typer.echo(f"Modality: {modality_lower}")
    typer.echo(f"Conditions: {', '.join(conditions) if conditions else 'None'}")
    for path in rel_paths:
        typer.echo(path)


@app.command("standardize-et")
def standardize_et_cmd(
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Process only the first N ET files",
    ),
) -> None:
    """Standardize ET files from the dataset index."""
    setup_logging()
    logger = get_logger(__name__)

    index_path = PROCESSED_DIR / "dataset_index.parquet"
    if not index_path.exists():
        raise typer.BadParameter("dataset_index.parquet not found; run ingest first.")

    df: pd.DataFrame = load_existing_index(index_path)
    df = cast(pd.DataFrame, df[df["modality"] == "et"].copy())
    if df.empty:
        typer.echo("No ET files found in dataset index.")
        return

    if limit is not None:
        df = df.head(limit)

    out_dir = PROCESSED_DIR / "et"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    columns = list(df.columns)
    col_index = {name: idx for idx, name in enumerate(columns)}
    for row in df.itertuples(index=False, name=None):
        in_path = Path(row[col_index["abs_path"]])
        participant_id = row[col_index["participant_id"]]
        condition = (
            row[col_index["condition"]] if "condition" in col_index else "unknown"
        )
        session_id = (
            row[col_index["session_id"]] if "session_id" in col_index else "unknown"
        )

        safe_session_id = sanitize_name(str(session_id))
        safe_condition = sanitize_name(str(condition))
        out_path = out_dir / f"{participant_id}_{safe_condition}_{safe_session_id}.parquet"

        logger.info("Standardizing ET file", extra={"path": str(in_path)})
        raw_df = load_et_xlsx(in_path)
        standardized = standardize_et(raw_df, TIMETICK_HZ)

        stats = sampling_stats(standardized)
        standardized.to_parquet(out_path, index=False)

        time_sec = standardized["time_sec"]
        start_time = float(time_sec.iloc[0]) if not time_sec.empty else float("nan")
        end_time = float(time_sec.iloc[-1]) if not time_sec.empty else float("nan")

        manifest_rows.append(
            {
                "participant_id": participant_id,
                "condition": condition,
                "session_id": session_id,
                "out_path": str(out_path),
                "n_rows": int(len(standardized)),
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "approx_hz": stats.get("approx_hz"),
            }
        )

    manifest_path = PROCESSED_DIR / "et_manifest.parquet"
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_parquet(manifest_path, index=False)
    typer.echo(f"Wrote manifest to {manifest_path}")


@app.command("featurize")
def featurize(
    manifest_path: Path = typer.Option(
        PROCESSED_DIR / "et_manifest.parquet",
        "--manifest-path",
        help="Path to ET manifest parquet",
    ),
    window_len_sec: float = typer.Option(
        10.0,
        "--window-len-sec",
        help="Window length in seconds",
    ),
    stride_sec: float = typer.Option(
        1.0,
        "--stride-sec",
        help="Stride length in seconds",
    ),
    limit_files: int | None = typer.Option(
        None,
        "--limit-files",
        help="Process only the first N files from manifest",
    ),
) -> None:
    """Extract sliding-window features from standardized ET data."""
    setup_logging()
    logger = get_logger(__name__)

    manifest_path = manifest_path.expanduser()
    if not manifest_path.exists():
        raise typer.BadParameter("ET manifest not found; run standardize-et first.")

    out_path = FEATURES_DIR / "window_features.parquet"
    df = featurize_all(
        manifest_path=manifest_path,
        out_path=out_path,
        window_len_sec=window_len_sec,
        stride_sec=stride_sec,
        limit_files=limit_files,
    )

    report_path = REPORTS_DIR / "feature_summary.md"
    write_feature_summary(df, report_path)
    logger.info("Feature summary written", extra={"path": str(report_path)})
    typer.echo(f"Wrote features to {out_path}")


@app.command("train")
def train(
    features_path: Path = typer.Option(
        FEATURES_DIR / "window_features.parquet",
        "--features-path",
        help="Path to window feature parquet",
    ),
    model_out: Path = typer.Option(
        Path("models") / "best_model.joblib",
        "--model-out",
        help="Path to save trained model",
    ),
    test_size: float = typer.Option(0.2, "--test-size", help="Holdout test size"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed"),
) -> None:
    """Train models and select the best configuration."""
    setup_logging()

    features_path = features_path.expanduser()
    model_out = model_out.expanduser()
    report_path = REPORTS_DIR / "model_report.md"
    cm_png_path = REPORTS_DIR / "confusion_matrix.png"

    train_and_select_best(
        features_path=features_path,
        out_model_path=model_out,
        report_path=report_path,
        cm_png_path=cm_png_path,
        random_state=random_state,
        test_size=test_size,
    )
    typer.echo(f"Wrote model to {model_out}")
    typer.echo(f"Wrote report to {report_path}")


@app.command("evaluate")
def evaluate(
    features_path: Path = typer.Option(
        FEATURES_DIR / "window_features.parquet",
        "--features-path",
        help="Path to window feature parquet",
    ),
    model_path: Path = typer.Option(
        Path("models") / "best_model.joblib",
        "--model-path",
        help="Path to trained model bundle",
    ),
) -> None:
    """Evaluate a saved model on the full dataset."""
    setup_logging()
    features_path = features_path.expanduser()
    model_path = model_path.expanduser()
    report_path = REPORTS_DIR / "model_eval_descriptive.md"

    evaluate_saved_model(
        features_path=features_path,
        model_path=model_path,
        report_path=report_path,
    )
    typer.echo(f"Wrote report to {report_path}")


@app.command("stats")
def stats(
    features_path: Path = typer.Option(
        FEATURES_DIR / "window_features.parquet",
        "--features-path",
        help="Path to window feature parquet",
    ),
) -> None:
    """Summarize feature dataset completeness."""
    setup_logging()
    logger = get_logger(__name__)

    features_path = features_path.expanduser()
    if not features_path.exists():
        raise typer.BadParameter("Feature file not found; run featurize first.")

    df = pd.read_parquet(features_path)
    stats_summary = compute_feature_stats(df)
    logger.info("Feature stats", extra=stats_summary)

    typer.echo(f"Windows: {stats_summary['n_windows']}")
    typer.echo(f"Participants: {stats_summary['n_participants']}")
    typer.echo(f"Condition counts: {stats_summary['condition_counts']}")
    typer.echo(
        "Missing % by group: "
        f"pupil={stats_summary['missing_pct_pupil']:.2f}, "
        f"blink={stats_summary['missing_pct_blink']:.2f}, "
        f"gaze={stats_summary['missing_pct_gaze']:.2f}, "
        f"aoi={stats_summary['missing_pct_aoi']:.2f}"
    )
    if stats_summary["all_nan_columns"]:
        typer.echo("All-NaN columns:")
        for col in stats_summary["all_nan_columns"]:
            typer.echo(f"- {col}")
    else:
        typer.echo("All-NaN columns: None")

    report_path = REPORTS_DIR / "feature_stats.md"
    write_markdown(report_path, build_feature_stats_report(stats_summary))
    typer.echo(f"Wrote report to {report_path}")


def load_existing_index(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def write_dataset_inventory(raw_root: Path, df):
    participants = sorted(df["participant_id"].unique().tolist())
    modality_counts = df["modality"].value_counts().to_dict()
    modality_condition_counts = (
        df.groupby(["modality", "condition"]).size().reset_index(name="count")
    )

    missing_map = build_missing_modalities(df, participants)
    report_path = REPORTS_DIR / "dataset_inventory.md"

    lines = [
        "# Dataset Inventory",
        "",
        f"Raw root: `{raw_root}`",
        "",
        f"Total participants: {len(participants)}",
        "",
        "## Counts by Modality",
        "",
    ]

    rows = [[modality, str(modality_counts.get(modality, 0))] for modality in MODALITIES]
    lines.extend(render_markdown_table(["Modality", "Count"], rows))

    lines.extend(
        [
            "",
            "## Counts by Modality x Condition",
            "",
        ]
    )

    mc_rows = [
        [row["modality"], row["condition"], str(row["count"])]
        for _, row in modality_condition_counts.iterrows()
    ]
    lines.extend(render_markdown_table(["Modality", "Condition", "Count"], mc_rows))

    lines.extend(
        [
            "",
            "## Participants Missing Modalities",
            "",
        ]
    )
    for modality in ("et", "nback", "drt", "nasatlx"):
        missing = missing_map.get(modality, [])
        lines.append(
            f"- {modality}: {', '.join(missing) if missing else 'None'}"
        )

    lines.extend(
        [
            "",
            "## Example Paths by Modality",
            "",
        ]
    )
    for modality in MODALITIES:
        lines.append(f"### {modality}")
        examples = df[df["modality"] == modality]["rel_path"].head(10).tolist()
        if examples:
            lines.extend([f"- {path}" for path in examples])
        else:
            lines.append("- None")
        lines.append("")

    return write_markdown(report_path, lines)


def write_feature_summary(features_df: pd.DataFrame, report_path: Path) -> None:
    total_windows = int(len(features_df))
    lines = [
        "# Feature Summary",
        "",
        f"Total windows: {total_windows}",
        "",
        "## Counts by Condition",
        "",
    ]

    if total_windows:
        counts = features_df["condition"].value_counts().reset_index()
        counts.columns = ["condition", "count"]
        rows = [[row["condition"], str(row["count"])] for _, row in counts.iterrows()]
        lines.extend(render_markdown_table(["Condition", "Count"], rows))
    else:
        lines.append("No windows produced.")

    write_markdown(report_path, lines)


def compute_feature_stats(df: pd.DataFrame) -> dict[str, Any]:
    n_windows = int(len(df))
    if "participant_id" in df.columns:
        n_participants = int(cast(int, df["participant_id"].nunique()))
    else:
        n_participants = 0

    if "condition" in df.columns:
        condition_counts = df["condition"].value_counts().to_dict()
    else:
        condition_counts = {}

    feature_df = df.drop(
        columns=[
            col
            for col in [
                "participant_id",
                "condition",
                "session_id",
                "window_start_sec",
                "window_end_sec",
            ]
            if col in df.columns
        ],
        errors="ignore",
    )

    all_nan_columns = [
        col for col in feature_df.columns if bool(feature_df[col].isna().all())
    ]

    def missing_pct(prefix: str) -> float:
        cols = [col for col in feature_df.columns if col.startswith(prefix)]
        if not cols:
            return float("nan")
        return float(feature_df[cols].isna().mean().mean() * 100.0)

    return {
        "n_windows": n_windows,
        "n_participants": n_participants,
        "condition_counts": condition_counts,
        "missing_pct_pupil": missing_pct("pupil_"),
        "missing_pct_blink": missing_pct("blink_"),
        "missing_pct_gaze": missing_pct("gaze_"),
        "missing_pct_aoi": missing_pct("aoi_"),
        "all_nan_columns": all_nan_columns,
    }


def build_feature_stats_report(stats_summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Feature Stats",
        "",
        f"Total windows: {stats_summary['n_windows']}",
        f"Participants: {stats_summary['n_participants']}",
        "",
        "## Condition Counts",
        "",
    ]

    condition_counts = cast(dict[str, int], stats_summary.get("condition_counts", {}))
    if condition_counts:
        rows = [[key, str(value)] for key, value in condition_counts.items()]
        lines.extend(render_markdown_table(["Condition", "Count"], rows))
    else:
        lines.append("No condition labels found.")

    lines.extend(
        [
            "",
            "## Missing Percentage by Feature Group",
            "",
            f"pupil: {stats_summary['missing_pct_pupil']:.2f}%",
            f"blink: {stats_summary['missing_pct_blink']:.2f}%",
            f"gaze: {stats_summary['missing_pct_gaze']:.2f}%",
            f"aoi: {stats_summary['missing_pct_aoi']:.2f}%",
            "",
            "## All-NaN Columns",
            "",
        ]
    )

    all_nan_columns = cast(list[str], stats_summary.get("all_nan_columns", []))
    if all_nan_columns:
        lines.extend([f"- {col}" for col in all_nan_columns])
    else:
        lines.append("- None")

    return lines


def build_missing_modalities(df, participants: list[str]) -> dict[str, list[str]]:
    missing = {modality: [] for modality in MODALITIES}
    for participant_id in participants:
        subset = df[df["participant_id"] == participant_id]
        for modality in ("et", "nback", "drt", "nasatlx"):
            if subset[subset["modality"] == modality].empty:
                missing[modality].append(participant_id)
    return missing


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


if __name__ == "__main__":
    app()
