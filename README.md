# Fatigue XR Demo

This project demonstrates a full, transparent pipeline for eye‑tracking fatigue/cognitive‑load analysis. It starts from raw participant folders, builds a dataset index, standardizes ET streams, extracts windowed features, trains a group‑aware classifier (no participant leakage), and replays model decisions in pseudo‑real time for XR adaptation. A WebSocket server streams those decisions to a demo client.

The goals are:

- Make every step auditable and reproducible.
- Keep raw input untouched (`Data/`).
- Use simple, well‑documented rules for labeling and adaptation.

## Project Layout

- `Data/` raw input (do not move/rename)
- `data/processed/` processed outputs (index, standardized ET, manifests)
- `data/features/` window features
- `models/` trained model bundles
- `reports/` markdown reports, replay CSVs, demo client
- `src/fatigue_xr/` source package
- `tests/`, `notebooks/`

## Requirements

- Python 3.11+
- Dependencies are managed in `pyproject.toml`

## Setup

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux (bash/zsh):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install in editable mode:

```bash
pip install -e .
```

## End‑to‑End Pipeline (What Happens and Why)

### 1) Scan raw structure
Command:

```bash
python -m fatigue_xr.cli scan-raw --raw-root Data
```

What it does:

- Discovers participant folders (ID*)
- Counts files by modality (ET/NBACK/DRT/TLX)
- Writes a summary report to `reports/raw_scan.md`

Why it exists:

- Confirms the raw dataset is consistent before processing.
- Highlights missing modalities without stopping the pipeline.

### 2) Build dataset index
Command:

```bash
python -m fatigue_xr.cli ingest --raw-root Data
```

What it does:

- Recursively indexes all files in participant folders
- Infers modality and condition from file paths
- Creates a stable per‑file index (`data/processed/dataset_index.parquet` and `.csv`)

Why it exists:

- Provides a single source of truth for downstream steps.
- Makes later operations deterministic and auditable.

### 3) Standardize ET streams
Command:

```bash
python -m fatigue_xr.cli standardize-et --limit 2
python -m fatigue_xr.cli standardize-et
```

What it does:

- Loads ET Excel files
- Normalizes column names into canonical forms
- Adds time axis, validity flags, and sampling statistics
- Writes standardized parquet files to `data/processed/et/`
- Writes `data/processed/et_manifest.parquet`

Why it exists:

- Ensures all ET files use a common schema.
- Keeps raw data untouched while producing analysis‑ready streams.

### 4) Extract sliding‑window features
Command:

```bash
python -m fatigue_xr.cli featurize --window-len-sec 10 --stride-sec 1
```

What it does:

- Generates windowed features from standardized ET
- Computes pupil, gaze, blink, AOI features where available
- Writes `data/features/window_features.parquet` and `.csv`

Why it exists:

- Converts high‑frequency time series into stable, ML‑ready features.
- Keeps windowing consistent for model training and replay.

### 5) Feature stats and missingness
Command:

```bash
python -m fatigue_xr.cli stats --features-path data/features/window_features.parquet
```

What it does:

- Reports window counts, participant counts, condition balance
- Summarizes missingness by feature group
- Lists all‑NaN features
- Writes `reports/feature_stats.md`

Why it exists:

- Makes data quality visible before model training.
- Documents which features are excluded (all‑NaN).

### 6) Train with group‑aware split
Command:

```bash
python -m fatigue_xr.cli train --features-path data/features/window_features.parquet
```

What it does:

- Defines label: `dual` = 1, `single` = 0
- Uses group‑aware split by `participant_id` to avoid leakage
- Runs CV (GroupKFold) on train only
- Selects best model by ROC‑AUC (tie‑break: F1)
- Saves model bundle to `models/best_model.joblib`
- Writes report to `reports/model_report.md`

Why it exists:

- Avoids optimistic results from participant leakage.
- Produces an auditable model selection process.

### 7) Evaluate model (descriptive)
Command:

```bash
python -m fatigue_xr.cli evaluate --features-path data/features/window_features.parquet
```

What it does:

- Evaluates the saved model on the full dataset
- Writes `reports/model_eval_descriptive.md`

Why it exists:

- Provides a quick descriptive sanity check.
- Clearly labeled as not leak‑safe.

### 8) Replay model decisions (pseudo‑real time)
Commands:

```bash
python -m fatigue_xr.cli list-et
python -m fatigue_xr.cli replay --participant-id ID01 --condition dual --session-id ID01_ET_0 --speed 20 --max-steps 60
```

What it does:

- Streams standardized ET through the window feature extractor
- Applies the trained model to each window
- Feeds scores into a hysteresis + cooldown state machine
- Writes replay CSV to `reports/replay_{participant_id}_{condition}_{session_id}.csv`

Why it exists:

- Simulates real‑time inference and adaptation actions.
- Produces a traceable output for demos or debugging.

## WebSocket Server + Demo Client

Start the server:

```bash
python -m fatigue_xr.cli serve --host 127.0.0.1 --port 8000
```

Open in browser:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/client`

WebSocket endpoint:

```
ws://127.0.0.1:8000/ws/replay?participant_id=ID01&condition=dual&session_id=ID01_ET_0&speed=10
```

The server streams JSON messages per window and writes a CSV to `reports/ws_replay_{participant_id}_{condition}_{session_id}.csv`.

## Outputs

- `data/processed/dataset_index.parquet`
- `data/processed/et/*.parquet`
- `data/processed/et_manifest.parquet`
- `data/features/window_features.parquet` and `data/features/window_features.csv`
- `models/best_model.joblib`
- `reports/*.md`, `reports/confusion_matrix.png`, replay CSVs

## Notes

- Raw data in `Data/` is never modified.
- All feature and model outputs are derived from standardized ET only.
- The adaptation engine uses fixed thresholds and cooldowns for auditability.
