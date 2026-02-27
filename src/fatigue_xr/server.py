from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

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
        df = load_et_from_manifest(manifest_path, participant_id, condition, session_id)
        if df.empty or "time_sec" not in df.columns:
            await websocket.send_json(
                {"error": "bad_request", "detail": "ET file missing time_sec"}
            )
            await websocket.close(code=1008)
            return

        bundle = load_model_bundle(model_path)
        time_series = cast(pd.Series, pd.to_numeric(df["time_sec"], errors="coerce"))
        t0 = float(time_series.min(skipna=True))
        t1 = float(time_series.max(skipna=True))

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
            {"event": "completed", "detail": "Replay finished normally"}
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
    else:
        html = """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Fatigue XR Demo</title>
  </head>
  <body>
    <h1>Fatigue XR Demo</h1>
    <p>Demo client not found.</p>
    <p>Expected: <code>reports/demo_client.html</code></p>
  </body>
</html>
"""

    banner = """
<div style=\"background:#0f172a;color:#fff;padding:12px 16px;border-radius:8px;margin-bottom:16px;\">
  <div style=\"font-size:18px;font-weight:600;\">Server running</div>
  <div style=\"opacity:0.85;\">WebSocket endpoint: <code style=\"color:#fff;\">/ws/replay</code></div>
  <div style=\"opacity:0.85;\">Example: <code style=\"color:#fff;\">ws://127.0.0.1:8000/ws/replay?participant_id=ID01&amp;condition=dual&amp;session_id=ID01_ET_0&amp;speed=10</code></div>
</div>
"""

    if "<body>" in html:
        html = html.replace("<body>", "<body>" + banner, 1)
    else:
        html = banner + html

    return HTMLResponse(html)
