from __future__ import annotations

import asyncio
import csv
import re
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.websockets import WebSocketDisconnect

from fatigue_xr.adaptation import AdaptationEngine
from fatigue_xr.features import window_features
from fatigue_xr.logging_utils import get_logger, setup_logging
from fatigue_xr.replay import load_et_from_manifest, load_model_bundle, predict_score

app = FastAPI()


@app.get("/")
def home() -> HTMLResponse:
    return _serve_demo_client()


@app.get("/client")
def client() -> HTMLResponse:
    return _serve_demo_client()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/routes")
def list_routes() -> list[dict[str, Any]]:
    routes = []
    for route in app.router.routes:
        routes.append(
            {
                "path": getattr(route, "path", ""),
                "name": getattr(route, "name", ""),
                "type": type(route).__name__,
                "methods": list(getattr(route, "methods", [])),
            }
        )
    return routes


@app.get("/api/sessions")
def list_sessions(
    participant_id: str | None = None, condition: str | None = None
) -> list[dict[str, Any]]:
    manifest_path = Path("data/processed/et_manifest.parquet")
    if not manifest_path.exists():
        return []

    df = cast(pd.DataFrame, pd.read_parquet(manifest_path))
    if participant_id:
        df = cast(pd.DataFrame, df[df["participant_id"] == participant_id])
    if condition:
        df = cast(pd.DataFrame, df[df["condition"] == condition])

    sessions = []
    for _, row in df.iterrows():
        session_full = str(row["session_id"])
        session_short = (
            session_full.split("__", 1)[1] if "__" in session_full else session_full
        )
        start = row.get("start_time_sec")
        end = row.get("end_time_sec")
        start_ok = bool(pd.notna(start))
        end_ok = bool(pd.notna(end))
        if start_ok and end_ok:
            duration = float(cast(float, end) - cast(float, start))
        else:
            duration = float("nan")
        sessions.append(
            {
                "participant_id": row.get("participant_id"),
                "condition": row.get("condition"),
                "session_id": session_full,
                "session_short": session_short,
                "approx_hz": row.get("approx_hz"),
                "duration_sec": duration,
            }
        )
    return sessions


@app.get("/api/summary")
def api_summary() -> dict[str, Any]:
    model_path = Path("models/best_model.joblib")
    model_name = "unknown"
    feature_set = "unknown"
    feature_columns: list[str] | None = None

    if model_path.exists():
        try:
            bundle = load_model_bundle(model_path)
            model_name = str(bundle.get("model_name", model_name))
            feature_set = str(bundle.get("feature_set", feature_set))
            if "feature_columns" in bundle:
                feature_columns = [str(c) for c in bundle["feature_columns"]]
        except Exception:
            model_name = "unknown"
            feature_set = "unknown"

    report_path = Path("reports/model_report.md")
    metrics: dict[str, float] | str = "see report"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        patterns = {
            "roc_auc": r"ROC[- ]AUC\s*[:=]\s*([0-9.]+)",
            "f1": r"F1\s*[:=]\s*([0-9.]+)",
            "accuracy": r"Accuracy\s*[:=]\s*([0-9.]+)",
            "precision": r"Precision\s*[:=]\s*([0-9.]+)",
            "recall": r"Recall\s*[:=]\s*([0-9.]+)",
        }
        found: dict[str, float] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    found[key] = float(match.group(1))
                except ValueError:
                    continue
        if found:
            metrics = found

    def artifact_path(name: str) -> str | None:
        path = Path("reports") / name
        return str(path) if path.exists() else None

    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "confusion_matrix_png": artifact_path("confusion_matrix.png"),
        "feature_importance_csv": artifact_path("feature_importance.csv"),
        "model_report_md": artifact_path("model_report.md"),
    }


@app.get("/api/feature-importance")
def api_feature_importance() -> list[dict[str, Any]]:
    path = Path("reports/feature_importance.csv")
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feature = row.get("feature") or row.get("name") or ""
            value_raw = row.get("value") or row.get("importance") or row.get("coef")
            try:
                value = float(value_raw) if value_raw is not None else 0.0
            except ValueError:
                value = 0.0
            rows.append({"feature": feature, "value": value})
    return rows


@app.get("/api/artifact/{name}")
def api_artifact(name: str) -> FileResponse:
    allowed = {
        "confusion_matrix.png": "image/png",
        "model_report.md": "text/markdown",
        "feature_importance.csv": "text/csv",
        "feature_stats.md": "text/markdown",
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = Path("reports") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type=allowed[name])


@app.websocket("/ws/replay")
async def ws_replay(websocket: WebSocket) -> None:
    await _handle_ws_replay(websocket)


@app.websocket("/ws/replay/")
async def ws_replay_slash(websocket: WebSocket) -> None:
    await _handle_ws_replay(websocket)


async def _handle_ws_replay(websocket: WebSocket) -> None:
    setup_logging()
    logger = get_logger(__name__)
    await websocket.accept()
    await websocket.send_json(
        {"event": "connected", "server_time_ms": int(time.time() * 1000)}
    )

    params = websocket.query_params
    participant_id = params.get("participant_id")
    condition = params.get("condition")
    session_id = params.get("session_id")

    expected = ["participant_id", "condition", "session_id"]
    if not participant_id or not condition or not session_id:
        await websocket.send_json(
            {
                "error": "bad_request",
                "detail": "participant_id, condition, session_id required",
                "expected": expected,
            }
        )
        await websocket.close(code=1008)
        return

    window_len_sec = float(params.get("window_len_sec", 10))
    stride_sec = float(params.get("stride_sec", 1))
    speed = float(params.get("speed", 10))
    max_steps_param = params.get("max_steps")
    max_steps = int(max_steps_param) if max_steps_param else None
    model_path = Path(params.get("model_path", "models/best_model.joblib"))
    manifest_path = Path(
        params.get("manifest_path", "data/processed/et_manifest.parquet")
    )

    logger.info(
        "WebSocket replay connected",
        extra={
            "url": str(websocket.url),
            "participant_id": participant_id,
            "condition": condition,
            "session_id": session_id,
            "window_len_sec": window_len_sec,
            "stride_sec": stride_sec,
            "speed": speed,
            "max_steps": max_steps,
        },
    )

    results: list[dict[str, Any]] = []
    out_path = Path(
        "reports"
    ) / f"ws_replay_{participant_id}_{condition}_{session_id}.csv"

    try:
        await websocket.send_json(
            {
                "event": "loading",
                "stage": "load_model",
                "server_time_ms": int(time.time() * 1000),
            }
        )
        bundle = load_model_bundle(model_path)

        await websocket.send_json(
            {
                "event": "loading",
                "stage": "load_manifest",
                "server_time_ms": int(time.time() * 1000),
            }
        )
        await websocket.send_json(
            {
                "event": "loading",
                "stage": "load_et",
                "server_time_ms": int(time.time() * 1000),
            }
        )
        df = load_et_from_manifest(manifest_path, participant_id, condition, session_id)
        if df.empty or "time_sec" not in df.columns:
            await websocket.send_json(
                {"error": "bad_request", "detail": "ET file missing time_sec"}
            )
            await websocket.close(code=1008)
            return

        time_series = cast(pd.Series, pd.to_numeric(df["time_sec"], errors="coerce"))
        t0 = float(time_series.min(skipna=True))
        t1 = float(time_series.max(skipna=True))

        await websocket.send_json(
            {"event": "streaming_started", "server_time_ms": int(time.time() * 1000)}
        )
        engine = AdaptationEngine()
        step = 0
        window_start = t0

        while window_start + window_len_sec <= t1:
            if max_steps is not None and step >= max_steps:
                break

            window_end = window_start + window_len_sec
            mask = (time_series >= window_start) & (time_series < window_end)
            window_df = df.loc[mask].copy()
            feats = window_features(window_df, window_len_sec)
            score = predict_score(bundle, feats)
            adaptation = engine.update(window_end, score)

            payload = {
                "participant_id": participant_id,
                "condition": condition,
                "session_id": session_id,
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                "server_time_ms": int(time.time() * 1000),
                "score": score,
                "state": adaptation["state"],
                "action": adaptation["action"],
                "action_changed": adaptation["action_changed"],
                "qa": {
                    "n_samples_window": feats.get("n_samples_window"),
                    "pupil_valid_frac": feats.get("pupil_valid_frac"),
                },
            }

            await websocket.send_json(payload)
            results.append(
                {
                    **payload,
                    "pupil_valid_frac": feats.get("pupil_valid_frac"),
                    "gaze_valid_frac": feats.get("gaze_valid_frac"),
                    "blink_rate_per_min": feats.get("blink_rate_per_min"),
                    "n_samples_window": feats.get("n_samples_window"),
                }
            )

            step += 1
            window_start += stride_sec
            if speed > 0:
                await asyncio.sleep(stride_sec / speed)

        await websocket.send_json(
            {
                "event": "completed",
                "detail": "Replay finished normally",
                "server_time_ms": int(time.time() * 1000),
            }
        )
        await websocket.close(code=1000)

    except WebSocketDisconnect:
        logger.info("WebSocket replay disconnected")
    except Exception as exc:
        logger.info("WebSocket replay error", extra={"error": str(exc)})
        try:
            await websocket.send_json(
                {"error": "server_exception", "detail": str(exc)}
            )
        finally:
            await websocket.close(code=1011)
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info("WebSocket replay finished", extra={"out_csv": str(out_path)})


def _serve_demo_client() -> HTMLResponse:
    client_path = Path("reports") / "demo_client.html"
    if client_path.exists():
        html = client_path.read_text(encoding="utf-8")
        response = HTMLResponse(html)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Fatigue XR Demo</title>
  </head>
  <body>
    <h1>Fatigue XR Demo</h1>
    <p>Demo client not found.</p>
    <p>Health: <a href="/health">/health</a></p>
    <p>Routes: <a href="/routes">/routes</a></p>
  </body>
</html>
"""
    response = HTMLResponse(html)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response
