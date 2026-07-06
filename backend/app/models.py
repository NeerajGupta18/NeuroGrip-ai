from typing import Optional

from pydantic import BaseModel, Field


class GenerateSignalRequest(BaseModel):
    intent: Optional[str] = Field(
        default=None,
        description="Force a specific grip intent, or omit for a random one.",
    )
    seed: Optional[int] = None
    inject_anomaly: Optional[str] = Field(
        default=None,
        description="Optionally corrupt the signal ('dropout', 'spike', 'noisy') to demo the safety layer.",
    )


class GenerateSignalResponse(BaseModel):
    signal: list[float]
    true_intent: str
    window_size: int
    injected_anomaly: Optional[str] = None


class PredictRequest(BaseModel):
    signal: list[float]
    user_id: str = "demo_user"


class AnomalyItem(BaseModel):
    type: str
    severity: str
    message: str


class PredictResponse(BaseModel):
    predicted_intent: str
    confidence: float
    confidence_by_intent: dict
    features: dict
    reasoning: str
    anomalies: list[AnomalyItem] = []


class FeedbackRequest(BaseModel):
    user_id: str = "demo_user"
    signal: list[float]
    predicted_intent: str
    actual_intent: str


class ProfileResponse(BaseModel):
    user_id: str
    session_count: int
    total_count: int
    correct_count: int
    accuracy_history: list[float]
    intent_distribution: dict


class UserListResponse(BaseModel):
    users: list[str]


class SessionReportResponse(BaseModel):
    user_id: str
    generated_at: str
    session_count: int
    total_count: int
    correct_count: int
    overall_accuracy: float
    accuracy_history: list[float]
    intent_distribution: dict

