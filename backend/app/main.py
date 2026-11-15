import os
import time
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import calibration, store
from .agent import CONFIDENCE_FLOOR, feature_table, run_agent
from .dataset import SOURCE_LABELS, SOURCES, get_recording, load_evaluation, load_model
from .faults import FAULT_TYPES, inject_fault
from .features import FEATURE_NAMES, SAMPLE_RATE, WINDOW, extract_features
from .model import GRASP_DESCRIPTIONS, GRASP_LABELS, GRASPS
from .models import (CalibrateRequest, CreateUserRequest, FeedbackRequest, NextSignalRequest,
                     NextSignalResponse, PredictRequest, PredictResponse, ProfileResponse,
                     SessionReportResponse, UserListResponse)

app = FastAPI(
    title="NeuroGrip AI",
    description=(
        "Explainable grip-intent classification from real two-channel surface EMG, "
        "with per-wearer calibration and a safety gate that abstains on bad signals."
    ),
    version="1.0.0",
)

# Cross-origin access is only needed when the dashboard is served from a different origin
# (local development: Vite on :5173, API on :8123). In the hosted build the API serves the
# dashboard itself, so ALLOWED_ORIGINS is left empty and no other site may call the API.
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def _to_array(ch1: list, ch2: list) -> np.ndarray:
    if len(ch1) != len(ch2):
        raise HTTPException(status_code=400, detail="ch1 and ch2 must have the same length")
    if len(ch1) < WINDOW:
        raise HTTPException(status_code=400, detail=f"signal needs at least {WINDOW} samples per channel")
    signal = np.column_stack([ch1, ch2]).astype(float)
    if not np.isfinite(signal).all():
        raise HTTPException(status_code=400, detail="signal contains non-finite values")
    return signal


def _check_grasp(grasp: str) -> None:
    if grasp not in GRASPS:
        raise HTTPException(status_code=400, detail=f"grasp must be one of {GRASPS}")


def _predict_response(result: dict, latency_ms: float) -> PredictResponse:
    p = result.get("prediction")
    return PredictResponse(
        decision=result["decision"],
        abstain_reason=result.get("abstain_reason"),
        predicted_grasp=p.grasp if p else None,
        runner_up=p.runner_up if p else None,
        confidence=p.confidence if p else None,
        confidence_floor=CONFIDENCE_FLOOR,
        confidence_by_grasp=p.confidence_by_grasp if p else {},
        window_votes=p.window_votes if p else [],
        contributions=feature_table(p) if p else [],
        reasoning=result["reasoning"],
        reasoning_source=result["reasoning_source"],
        anomalies=result.get("anomalies", []),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version, "profile_store": store.backend()}


@app.get("/api/meta")
def meta():
    """Static description of the model and data, for the dashboard."""
    return {
        "grasps": [{"key": g, "label": GRASP_LABELS[g], "description": GRASP_DESCRIPTIONS[g]} for g in GRASPS],
        "sources": [{"key": s, "label": SOURCE_LABELS[s]} for s in SOURCES],
        "fault_types": FAULT_TYPES,
        "feature_names": FEATURE_NAMES,
        "sample_rate": SAMPLE_RATE,
        "confidence_floor": CONFIDENCE_FLOOR,
        "window_samples": load_model()["window"],
    }


@app.get("/api/evaluation")
def evaluation():
    """The offline evaluation produced by training/train.py."""
    return load_evaluation()


@app.get("/api/users", response_model=UserListResponse)
def users():
    return UserListResponse(users=calibration.list_profiles())


@app.post("/api/users", response_model=ProfileResponse)
def create_user(req: CreateUserRequest):
    if req.source is not None and req.source not in SOURCES:
        raise HTTPException(status_code=400, detail=f"source must be one of {SOURCES}")
    return calibration.summarise(calibration.create_profile(req.user_id, req.source))


@app.get("/api/profile/{user_id}", response_model=ProfileResponse)
def profile(user_id: str):
    return calibration.summarise(calibration.get_profile(user_id))


@app.post("/api/profile/{user_id}/reset", response_model=ProfileResponse)
def reset(user_id: str):
    return calibration.summarise(calibration.reset_profile(user_id))


@app.post("/api/signal/next", response_model=NextSignalResponse)
def next_signal(req: NextSignalRequest):
    """Serve the wearer's next unseen real recording, optionally corrupted with a sensor fault."""
    if req.grasp is not None:
        _check_grasp(req.grasp)
    if req.inject_fault is not None and req.inject_fault not in FAULT_TYPES:
        raise HTTPException(status_code=400, detail=f"inject_fault must be one of {FAULT_TYPES}")
    prof = calibration.get_profile(req.user_id)
    grasp, index = calibration.next_recording(prof, req.grasp)
    signal = get_recording(prof["source"], grasp, index)
    if req.inject_fault:
        signal = inject_fault(signal, req.inject_fault)
    return NextSignalResponse(
        user_id=prof["user_id"], source=prof["source"], true_grasp=grasp, recording_index=index,
        sample_rate=SAMPLE_RATE, injected_fault=req.inject_fault,
        ch1=np.round(signal[:, 0], 4).tolist(), ch2=np.round(signal[:, 1], 4).tolist(),
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run the decision graph on a two-channel recording."""
    signal = _to_array(req.ch1, req.ch2)
    prof = calibration.get_profile(req.user_id)
    start = time.perf_counter()
    result = run_agent(signal, prof)
    return _predict_response(result, 1000 * (time.perf_counter() - start))


@app.post("/api/feedback", response_model=ProfileResponse)
def feedback(req: FeedbackRequest):
    """Confirm what the grasp really was. The server re-runs the prediction itself (so the logged
    outcome cannot be spoofed by the client), logs it, then learns from the recording - unless the
    recording is corrupted, in which case nothing is learned."""
    _check_grasp(req.actual_grasp)
    signal = _to_array(req.ch1, req.ch2)
    prof = calibration.get_profile(req.user_id)
    result = run_agent(signal, prof)
    if result.get("anomalies"):
        summary = calibration.summarise(prof)
        return ProfileResponse(**summary, learned=False,
                               note="Recording has a sensor fault; it was not used for calibration.")
    p = result["prediction"]
    prof = calibration.learn(prof, extract_features(signal), req.actual_grasp, p.grasp,
                             abstained=result["decision"] == "abstain", confidence=p.confidence)
    return ProfileResponse(**calibration.summarise(prof), learned=True)


@app.post("/api/calibrate", response_model=ProfileResponse)
def calibrate(req: CalibrateRequest):
    """Guided calibration: the wearer performs every grasp `reps` times. Each recording is
    predicted first (and scored), then learned from."""
    prof = calibration.get_profile(req.user_id)
    for _ in range(req.reps):
        for grasp in GRASPS:
            g, index = calibration.next_recording(prof, grasp)
            signal = get_recording(prof["source"], g, index)
            result = run_agent(signal, prof)
            if result.get("anomalies"):
                continue
            p = result["prediction"]
            prof = calibration.learn(prof, extract_features(signal), g, p.grasp,
                                     abstained=result["decision"] == "abstain", confidence=p.confidence)
    return ProfileResponse(**calibration.summarise(prof), learned=True)


@app.get("/api/report/{user_id}", response_model=SessionReportResponse)
def report(user_id: str):
    """Session summary for a clinician reviewing a fitting session, next to the accuracy the
    offline evaluation says to expect for that amount of calibration."""
    summary = calibration.summarise(calibration.get_profile(user_id))
    curve = load_evaluation()["calibration_curve"]["reps_per_grasp"]
    return SessionReportResponse(
        **summary, generated_at=datetime.now(timezone.utc).isoformat(),
        expected_accuracy_by_reps={reps: v["recording"] for reps, v in curve.items()},
    )


# Hosted build: serve the compiled dashboard from the same origin as the API. Registered last,
# so every /api route above takes precedence. Absent in local development (Vite serves it).
FRONTEND_DIST = os.environ.get(
    "FRONTEND_DIST", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend", "dist")
)
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="dashboard")
