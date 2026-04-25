from __future__ import annotations

import asyncio
import json
import os
import random
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import paho.mqtt.client as mqtt
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from analyzer.baseline import BaselineCalibrator
from analyzer.model import StressModel
from analyzer.session_manager import SessionManager
from analyzer.smoother import PredictionSmoother
from analyzer.synthetic import generate_dataset
from analyzer.window import SlidingWindow, Reading
from schemas import Assessment, FeatureContrib, HealthSnapshot, ReadingResponse

BASE_DIR = Path(__file__).parent

# Config

MQTT_BROKER   = os.getenv("MQTT_BROKER",   "localhost")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC    = "sensor/health/data"
MOCK_INTERVAL = float(os.getenv("MOCK_INTERVAL", "1.0"))

_MOCK_PARAMS: dict[str, dict] = {
    "calm":           {"bpm_mu": 68.0, "bpm_sigma": 4.0,  "temp_mu": 36.75, "temp_sigma": 0.12, "fsr_mu":  85.0, "fsr_sigma":  35.0},
    "cognitive_load": {"bpm_mu": 83.0, "bpm_sigma": 5.5,  "temp_mu": 36.45, "temp_sigma": 0.16, "fsr_mu": 270.0, "fsr_sigma":  85.0},
    "stressed":       {"bpm_mu": 98.0, "bpm_sigma": 8.0,  "temp_mu": 36.10, "temp_sigma": 0.20, "fsr_mu": 560.0, "fsr_sigma": 140.0},
}

# Application state (module-level singletons shared across requests)

model       = StressModel()
window      = SlidingWindow()
baseline    = BaselineCalibrator()
smoother    = PredictionSmoother()
session_mgr = SessionManager()

_sse_clients:   list[asyncio.Queue] = []
_last_event:    Optional[dict] = None
_mock_task:     Optional[asyncio.Task] = None
_mock_scenario: str = "calm"

def _boot_model() -> None:
    if model.load():
        print(f"[model] Loaded  engine={model.active_model}  cv_acc={model.cv_accuracy:.3f}")
        return
    print("[model] Training on synthetic data…")
    X, y = generate_dataset()
    acc = model.train(X, y)
    print(f"[model] Done  engine={model.active_model}  cv_acc={acc:.3f}  samples={len(y)}")


# MQTT client — runs in a daemon thread so it doesn't block the event loop


def _start_mqtt(loop: asyncio.AbstractEventLoop) -> None:
    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(MQTT_TOPIC)
            print(f"[mqtt] Connected  broker={MQTT_BROKER}:{MQTT_PORT}  topic={MQTT_TOPIC}")
        else:
            print(f"[mqtt] Connect failed  rc={reason_code}")

    def on_message(client, userdata, msg):
        try:
            snapshot = HealthSnapshot.model_validate_json(msg.payload)
            asyncio.run_coroutine_threadsafe(_process_reading(snapshot), loop)
        except Exception as exc:
            print(f"[mqtt] parse error: {exc}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_forever()
    except Exception as exc:
        print(f"[mqtt] Not available: {exc}")


# Built-in mock generator — simulates sensor data without real hardware


async def _mock_loop() -> None:
    p    = _MOCK_PARAMS[_mock_scenario]
    bpm  = p["bpm_mu"]
    temp = p["temp_mu"]
    try:
        while True:
            p = _MOCK_PARAMS[_mock_scenario]
            bpm  += (p["bpm_mu"]  - bpm)  * 0.1 + random.gauss(0, p["bpm_sigma"]  * 0.25)
            temp += (p["temp_mu"] - temp) * 0.05 + random.gauss(0, p["temp_sigma"] * 0.20)
            fsr   = max(0, min(1023, int(random.gauss(p["fsr_mu"], p["fsr_sigma"]))))
            bpm   = max(40.0, min(180.0, bpm))
            temp  = max(34.0, min(40.0,  temp))

            snapshot = HealthSnapshot(
                device="mock",
                state="measuring",
                max30102_ready=True,
                finger=True,
                bpm_valid=True,
                bpm_current=round(bpm, 1),
                bpm_avg=int(bpm),
                bpm_samples=5,
                temp_valid=True,
                temp_c=round(temp, 3),
                fsr_raw=fsr,
                fsr_pressed=fsr > 200,
            )
            await _process_reading(snapshot)
            await asyncio.sleep(MOCK_INTERVAL)
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(_boot_model)
    loop = asyncio.get_event_loop()
    threading.Thread(target=_start_mqtt, args=(loop,), daemon=True).start()
    yield


app = FastAPI(title="Affective Computing API", lifespan=lifespan)

async def _broadcast(payload: dict) -> None:
    for q in _sse_clients:
        await q.put(payload)


def _compute_assessment(snapshot: HealthSnapshot) -> Assessment:
    if snapshot.state != "measuring" or not snapshot.bpm_valid:
        smoother.reset()
        return Assessment(stress_level="no_signal", engine="none", window_size=window.size())

    if not window.is_ready():
        smoother.reset()
        return Assessment(stress_level="warming_up", engine="none", window_size=window.size())

    label, score, conf = model.predict(window.as_list(), baseline=baseline.stats)
    label, score, conf = smoother.update(label, score, conf)

    explanation = [
        FeatureContrib(**f)
        for f in model.explain(window.as_list(), baseline=baseline.stats, top_n=3)
    ]

    return Assessment(
        stress_level=label,
        stress_score=score,
        confidence=conf,
        engine="ml",
        window_size=window.size(),
        explanation=explanation,
    )


# Single entry point for a sensor reading — called from HTTP, MQTT and mock


async def _process_reading(snapshot: HealthSnapshot) -> Assessment:
    global _last_event

    reading = Reading(
        bpm=snapshot.bpm_current,
        temp_c=snapshot.temp_c,
        fsr_raw=snapshot.fsr_raw,
        bpm_valid=snapshot.bpm_valid,
        temp_valid=snapshot.temp_valid,
    )

    if baseline.is_recording:
        baseline.add(reading)

    if snapshot.state == "measuring" and snapshot.bpm_valid:
        window.push(reading)

    assessment = _compute_assessment(snapshot)

    if session_mgr.is_recording:
        session_mgr.add_reading(
            bpm=snapshot.bpm_current,
            temp_c=snapshot.temp_c,
            fsr_raw=snapshot.fsr_raw,
            stress_score=assessment.stress_score,
        )

    event: dict = {
        "ts": datetime.now().isoformat(),
        "bpm": snapshot.bpm_current,
        "bpm_avg": snapshot.bpm_avg,
        "temp_c": snapshot.temp_c,
        "fsr_raw": snapshot.fsr_raw,
        "state": snapshot.state,
        "baseline_recording": baseline.is_recording,
        "baseline_calibrated": baseline.is_calibrated,
        "baseline_buffer": baseline.buffer_size,
        "session_recording": session_mgr.is_recording,
        "session_label": session_mgr.current_label,
        "session_count": session_mgr.current_count,
        **assessment.model_dump(),
    }
    _last_event = event
    asyncio.create_task(_broadcast(event))
    return assessment


# Endpoints


@app.post("/api/readings", response_model=ReadingResponse)
async def receive_reading(snapshot: HealthSnapshot) -> ReadingResponse:
    assessment = await _process_reading(snapshot)
    return ReadingResponse(status="ok", state=snapshot.state, assessment=assessment)


@app.get("/api/stream")
async def sse_stream(request: Request) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    _sse_clients.append(queue)

    async def _gen():
        if _last_event:
            yield f"data: {json.dumps(_last_event)}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sse_clients.remove(queue)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/baseline/start")
async def baseline_start() -> dict:
    baseline.start()
    return {"status": "recording", "min_readings": 20}


@app.post("/api/baseline/stop")
async def baseline_stop() -> dict:
    ok = baseline.stop()
    if not ok:
        raise HTTPException(400, f"Need ≥20 readings, got {baseline.buffer_size}")
    s = baseline.stats
    return {
        "status": "calibrated",
        "bpm_mean": round(s.bpm_mean, 1),
        "bpm_std": round(s.bpm_std, 2),
        "temp_mean": round(s.temp_mean, 2),
        "fsr_mean": round(s.fsr_mean, 1),
        "n_readings": s.n_readings,
    }


@app.get("/api/baseline/status")
async def baseline_status() -> dict:
    return {
        "recording": baseline.is_recording,
        "calibrated": baseline.is_calibrated,
        "buffer_size": baseline.buffer_size,
        "stats": {
            "bpm_mean": round(baseline.stats.bpm_mean, 1),
            "temp_mean": round(baseline.stats.temp_mean, 2),
            "fsr_mean": round(baseline.stats.fsr_mean, 1),
        } if baseline.is_calibrated else None,
    }


@app.post("/api/session/start")
async def session_start(body: dict) -> dict:
    label = body.get("label", "calm")
    if label not in ("calm", "cognitive_load", "stressed"):
        raise HTTPException(400, "label must be calm | cognitive_load | stressed")
    session_id = session_mgr.start(label)
    return {"status": "recording", "session_id": session_id, "label": label}


@app.post("/api/session/stop")
async def session_stop() -> dict:
    session_id = session_mgr.stop()
    if not session_id:
        raise HTTPException(400, "No active session")
    return {"status": "saved", "session_id": session_id}


@app.get("/api/session/status")
async def session_status() -> dict:
    return {
        "recording": session_mgr.is_recording,
        "label": session_mgr.current_label,
        "readings_count": session_mgr.current_count,
    }


@app.get("/api/sessions")
async def list_sessions() -> dict:
    return {"sessions": session_mgr.list_all()}


@app.get("/api/mock/status")
async def mock_status() -> dict:
    return {
        "running": _mock_task is not None and not _mock_task.done(),
        "scenario": _mock_scenario,
    }


@app.post("/api/mock/start")
async def mock_start(body: dict) -> dict:
    global _mock_task, _mock_scenario
    scenario = body.get("scenario", "calm")
    if scenario not in _MOCK_PARAMS:
        raise HTTPException(400, "scenario must be calm | cognitive_load | stressed")
    if _mock_task and not _mock_task.done():
        _mock_task.cancel()
        await asyncio.sleep(0)
    _mock_scenario = scenario
    _mock_task = asyncio.create_task(_mock_loop())
    return {"status": "running", "scenario": scenario}


@app.post("/api/mock/stop")
async def mock_stop() -> dict:
    global _mock_task
    if _mock_task and not _mock_task.done():
        _mock_task.cancel()
        _mock_task = None
    return {"status": "stopped"}


@app.get("/api/model/status")
async def model_status() -> dict:
    return {
        "trained": model.is_ready,
        "active_model": model.active_model,
        "cv_accuracy": model.cv_accuracy,
        "all_scores": model.all_scores,
        "window_size": window.size(),
        "window_ready": window.is_ready(),
        "baseline_calibrated": baseline.is_calibrated,
        "session_recording": session_mgr.is_recording,
    }


@app.post("/api/model/retrain")
async def retrain() -> dict:
    syn_X, syn_y = await asyncio.to_thread(generate_dataset)
    real_X, real_y = await asyncio.to_thread(session_mgr.build_training_data)

    if real_X is not None and len(real_X) > 0:
        X = np.vstack([syn_X, np.repeat(real_X, 5, axis=0)])
        y = np.concatenate([syn_y, np.tile(real_y, 5)])
        real_count = len(real_X)
    else:
        X, y = syn_X, syn_y
        real_count = 0

    acc = await asyncio.to_thread(model.train, X, y)
    smoother.reset()

    return {
        "cv_accuracy": round(float(acc), 4),
        "active_model": model.active_model,
        "all_scores": model.all_scores,
        "total_samples": len(y),
        "real_windows": real_count,
    }


@app.get("/api/model/explain")
async def model_explain() -> dict:
    if not model.is_ready or not model.feature_importances:
        return {"features": []}
    from analyzer.features import FEATURE_LABELS, FEATURE_NAMES
    items = [
        {"name": n, "label": FEATURE_LABELS.get(n, n), "importance": round(imp, 4)}
        for n, imp in zip(FEATURE_NAMES, model.feature_importances)
    ]
    return {"features": sorted(items, key=lambda x: x["importance"], reverse=True)}


@app.post("/api/window/reset")
async def reset_window() -> dict:
    window.clear()
    smoother.reset()
    return {"status": "ok"}


@app.get("/")
async def dashboard() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
