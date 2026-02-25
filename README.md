# Fatigue XR Demo

Eye-tracking fatigue/cognitive-load demo with a raw data root at `Data/`, standardized ET processing, feature extraction, model training, and pseudo‑real‑time replay (CLI + WebSocket).

## Project Layout

- `Data/` raw input (do not move/rename)
- `data/processed/` processed outputs (index, standardized ET, manifests)
- `data/features/` window features
- `models/` trained model bundles
- `reports/` markdown reports + replay CSVs + demo client
- `src/fatigue_xr/` source package
- `tests/`, `notebooks/`

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

## CLI Workflow

Scan raw structure:

```bash
python -m fatigue_xr.cli scan-raw --raw-root Data
```

Build dataset index and inventory:

```bash
python -m fatigue_xr.cli ingest --raw-root Data
```

Standardize ET files (optional limit):

```bash
python -m fatigue_xr.cli standardize-et --limit 2
python -m fatigue_xr.cli standardize-et
```

Extract sliding-window features:

```bash
python -m fatigue_xr.cli featurize --window-len-sec 10 --stride-sec 1
```

Feature stats and missingness:

```bash
python -m fatigue_xr.cli stats --features-path data/features/window_features.parquet
```

Train model with group-aware split:

```bash
python -m fatigue_xr.cli train --features-path data/features/window_features.parquet
```

Evaluate saved model (descriptive, full dataset):

```bash
python -m fatigue_xr.cli evaluate --features-path data/features/window_features.parquet
```

List available ET sessions:

```bash
python -m fatigue_xr.cli list-et
python -m fatigue_xr.cli list-et --participant-id ID01 --condition dual
```

Replay ET session (short or full session id):

```bash
python -m fatigue_xr.cli replay --participant-id ID01 --condition dual --session-id ID01_ET_0 --speed 20 --max-steps 60
python -m fatigue_xr.cli replay --participant-id ID01 --condition dual --session-id dual__ID01_ET_0
```

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
