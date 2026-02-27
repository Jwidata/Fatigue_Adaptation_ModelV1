# Project Overview

This repo implements an end‑to‑end eye‑tracking workload pipeline for classifying **Single vs Dual** task condition, replaying model predictions in pseudo‑real time, and mapping scores to XR adaptation actions. It keeps raw data untouched, standardizes ET streams, extracts windowed features, trains a group‑aware model (no participant leakage), and streams replay decisions via CLI or WebSocket.

# Repo Structure

```
Data/                # Raw dataset (ID01..ID28). Untouched and gitignored.
data/processed/      # Index + standardized ET + manifests
data/features/       # Window features
models/              # Trained model bundles
reports/             # Markdown reports, replay CSVs, demo client
src/                 # Python package (fatigue_xr)
```

# Setup

Python 3.11+ recommended.

Create a venv and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Notes:

- On WSL/Linux, use `python3`.
- Inside the venv, `python` should work.

# Pipeline: Step‑by‑step Commands

## 1) Scan raw

```bash
python -m fatigue_xr.cli scan-raw --raw-root Data
```

Creates a structural report at `reports/raw_scan.md`.

## 2) Ingest index

```bash
python -m fatigue_xr.cli ingest --raw-root Data
```

Creates:

- `data/processed/dataset_index.parquet`
- `data/processed/dataset_index.csv`
- `reports/dataset_inventory.md`

## 3) Standardize ET

Quick check:

```bash
python -m fatigue_xr.cli standardize-et --limit 2
```

Full run:

```bash
python -m fatigue_xr.cli standardize-et
```

Outputs:

- `data/processed/et/*.parquet`
- `data/processed/et_manifest.parquet`

## 4) Featurize (windowed)

```bash
python -m fatigue_xr.cli featurize --window-len-sec 10 --stride-sec 1
```

Optional limit:

```bash
python -m fatigue_xr.cli featurize --limit-files 2
```

Outputs:

- `data/features/window_features.parquet`
- `data/features/window_features.csv`
- `reports/feature_summary.md`

## 5) Feature stats (missingness)

```bash
python -m fatigue_xr.cli stats --features-path data/features/window_features.parquet
```

Creates `reports/feature_stats.md`.

Notes:

- AOI features may be entirely missing depending on the dataset; such columns are automatically dropped during training.

## 6) Train + Evaluate

Train with group‑aware split (participant leakage prevented):

```bash
python -m fatigue_xr.cli train --features-path data/features/window_features.parquet
```

Evaluate descriptively on full dataset:

```bash
python -m fatigue_xr.cli evaluate --features-path data/features/window_features.parquet
```

Reports and artifacts:

- `reports/model_report.md`
- `reports/confusion_matrix.png`
- `reports/feature_importance.csv`
- `reports/model_eval_descriptive.md`
- `models/best_model.joblib`

Model selection:

- Automatically selected based on **GroupKFold CV ROC‑AUC**, tie‑break by **F1**.
- The default winner is typically Logistic Regression on `pupil_only`, but selection is data‑driven.

# Replay Demo (CLI)

List available ET sessions:

```bash
python -m fatigue_xr.cli list-et
python -m fatigue_xr.cli list-et --participant-id ID01 --condition dual
```

Replay a session:

```bash
python -m fatigue_xr.cli replay --participant-id ID01 --condition dual --session-id ID01_ET_0 --speed 20 --max-steps 60
```

Notes:

- `session_id` can be **short** (`ID01_ET_0`) or **full** (`dual__ID01_ET_0`).
- `window_len_sec` and `stride_sec` control the sliding window.

Interpretation:

- `score`: model probability of **Dual** (higher ⇒ higher load)
- `state`: LOW / MEDIUM / HIGH
- `action`: NO_CHANGE / REDUCE_CLUTTER / SLOW_ANIMATIONS / SUGGEST_BREAK

# Live Streaming Demo (FastAPI + WebSocket)

Start the server:

```bash
python -m fatigue_xr.cli serve --host 127.0.0.1 --port 8000
```

Verify health and routes:

```bash
http://127.0.0.1:8000/health
http://127.0.0.1:8000/routes
```

Open the browser client:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/client`

WebSocket endpoint:

```
ws://127.0.0.1:8000/ws/replay?participant_id=ID01&condition=dual&session_id=ID01_ET_0&speed=10
```

Normal completion closes with code **1000**. Errors are sent as JSON and then closed.

# Interactive Demo Client

Start the server:

```bash
python -m fatigue_xr.cli serve --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/`

Use the Stream Source selector:

- Replay (Dataset)
- Remote WebSocket URL
- Synthetic (Demo Generator)

# Windows one-click demo

Double click `run_demo_windows.bat`.

# Troubleshooting

- **python vs python3**
  Use `python3` on WSL/Linux. Inside venv, `python` should work.

- **WebSocket connection failing**
  Install WebSocket backend for uvicorn:
  ```bash
  pip install "uvicorn[standard]" websockets wsproto
  ```

- **session_id mismatch**
  You can pass short or full IDs. The matcher supports suffixes (`dual__ID01_ET_0` vs `ID01_ET_0`).

- **AOI features missing**
  This is expected when AOI is not present. All‑NaN columns are auto‑dropped before training.

# Reproducibility Notes

- Splits are **group‑aware** by `participant_id` to avoid leakage.
- Model selection uses **CV ROC‑AUC**, tie‑break by **F1**.
- Outputs are deterministic with `random_state=42`.
