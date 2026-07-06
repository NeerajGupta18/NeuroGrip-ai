from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import run_agent
from .calibration import apply_feedback, get_profile, list_profiles
from .features import extract_features
from .models import (
    FeedbackRequest,
    GenerateSignalRequest,
    GenerateSignalResponse,
    PredictRequest,
    PredictResponse,
    ProfileResponse,
    SessionReportResponse,
    UserListResponse,
)
from .signal_simulator import ANOMALY_TYPES, INTENTS, generate_signal

app = FastAPI(
    title="NeuroGrip AI",
    description=(
        "Explainable AI pipeline that interprets simulated myoelectric "
        "(muscle) signals into bionic hand grip intent, with per-user "
        "adaptive calibration and signal-quality safety checks."
    ),
    version="0.2.0",
)

# Wide open CORS since this is a local demo project, not a production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _profile_to_response(p: dict) -> ProfileResponse:
    return ProfileResponse(
        user_id=p["user_id"],
        session_count=p["session_count"],
        total_count=p["total_count"],
        correct_count=p["correct_count"],
        accuracy_history=p["accuracy_history"],
        intent_distribution=p["intent_distribution"],
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "intents": INTENTS, "anomaly_types": ANOMALY_TYPES}


@app.post("/api/signal/generate", response_model=GenerateSignalResponse)
def generate(req: GenerateSignalRequest):
    if req.intent is not None and req.intent not in INTENTS:
        raise HTTPException(status_code=400, detail=f"intent must be one of {INTENTS}")
    if req.inject_anomaly is not None and req.inject_anomaly not in ANOMALY_TYPES:
        raise HTTPException(status_code=400, detail=f"inject_anomaly must be one of {ANOMALY_TYPES}")
    result = generate_signal(intent=req.intent, seed=req.seed, inject_anomaly=req.inject_anomaly)
    return GenerateSignalResponse(
        signal=result["signal"],
        true_intent=result["intent"],
        window_size=result["window_size"],
        injected_anomaly=result["injected_anomaly"],
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.signal) < 10:
        raise HTTPException(status_code=400, detail="signal window too short")
    result = run_agent(signal=req.signal, user_id=req.user_id)
    return PredictResponse(
        predicted_intent=result["predicted_intent"],
        confidence=result["confidence"],
        confidence_by_intent=result["confidence_by_intent"],
        features=result["features"],
        reasoning=result["reasoning"],
        anomalies=result.get("anomalies", []),
    )


@app.post("/api/feedback", response_model=ProfileResponse)
def feedback(req: FeedbackRequest):
    if req.actual_intent not in INTENTS:
        raise HTTPException(status_code=400, detail=f"actual_intent must be one of {INTENTS}")
    features = extract_features(req.signal)
    profile = apply_feedback(
        user_id=req.user_id,
        features=features,
        predicted_intent=req.predicted_intent,
        actual_intent=req.actual_intent,
    )
    return _profile_to_response(profile)


@app.get("/api/profile/{user_id}", response_model=ProfileResponse)
def profile(user_id: str):
    p = get_profile(user_id)
    return _profile_to_response(p)


@app.get("/api/users", response_model=UserListResponse)
def users():
    """List every user_id with a saved calibration profile, for the
    multi-user profile switcher in the UI."""
    return UserListResponse(users=list_profiles())


@app.get("/api/report/{user_id}", response_model=SessionReportResponse)
def report(user_id: str):
    """Return a summarized session report for a user — accuracy trend and
    grip-intent distribution — the data behind the frontend's downloadable
    report, aimed at the same audience a clinician reviewing a remote
    rehabilitation session would be."""
    p = get_profile(user_id)
    overall_accuracy = (
        round(100 * p["correct_count"] / p["total_count"], 1) if p["total_count"] else 0.0
    )
    return SessionReportResponse(
        user_id=p["user_id"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        session_count=p["session_count"],
        total_count=p["total_count"],
        correct_count=p["correct_count"],
        overall_accuracy=overall_accuracy,
        accuracy_history=p["accuracy_history"],
        intent_distribution=p["intent_distribution"],
    )
