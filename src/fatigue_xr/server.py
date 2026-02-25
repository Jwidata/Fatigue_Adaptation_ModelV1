from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
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


@app.websocket("/ws/replay")
async def ws_replay(websocket: WebSocket) -> None:
    setup_logging()
    logger = get_logger(__name__)
    await websocket.accept()

    params = websocket.query_params
    participant_id = params.get("participant_id")
    condition = params.get("condition")
    session_id = params.get("session_id")

    if not participant_id or not condition or not session_id:
        await websocket.send_json({"error": "participant_id, condition, session_id required"})
        await websocket.close()
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
            "participant_id": participant_id,
            "condition": condition,
            "session_id": session_id,
        },
    )

    results: list[dict[str, Any]] = []
    out_path = Path(
        "reports"
    ) / f"ws_replay_{participant_id}_{condition}_{session_id}.csv"

    try:
        df = load_et_from_manifest(manifest_path, participant_id, condition, session_id)
        if df.empty or "time_sec" not in df.columns:
            await websocket.send_json({"error": "ET file is empty or missing time_sec"})
            await websocket.close()
            return

        bundle = load_model_bundle(model_path)
        time_series = pd.to_numeric(df["time_sec"], errors="coerce")
        time_array = time_series.to_numpy(dtype=float)
        t0 = float(pd.Series(time_array).min(skipna=True))
        t1 = float(pd.Series(time_array).max(skipna=True))

        engine = AdaptationEngine()
        step = 0
        window_start = t0

        while window_start + window_len_sec <= t1:
            if max_steps is not None and step >= max_steps:
                break

            window_end = window_start + window_len_sec
            mask = (time_array >= window_start) & (time_array < window_end)
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

    except WebSocketDisconnect:
        logger.info("WebSocket replay disconnected")
    except Exception as exc:
        logger.info("WebSocket replay error", extra={"error": str(exc)})
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info("WebSocket replay finished", extra={"out_csv": str(out_path)})


def _serve_demo_client() -> HTMLResponse:
    client_path = Path("reports") / "demo_client.html"
    if client_path.exists():
        return HTMLResponse(client_path.read_text(encoding="utf-8"))

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
    <p>Run the CLI to generate the client:</p>
    <pre>reports/demo_client.html</pre>
  </body>
</html>
"""
    return HTMLResponse(html)
