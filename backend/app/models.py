from typing import Optional

from pydantic import BaseModel, Field

MAX_SAMPLES = 10_000     # 20 s at 500 Hz; guards the API against oversized payloads


class Signal(BaseModel):
    ch1: list[float] = Field(max_length=MAX_SAMPLES)
    ch2: list[float] = Field(max_length=MAX_SAMPLES)


class CreateUserRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=40)
    source: Optional[str] = Field(default=None, description="Which recorded wearer feeds this profile.")


class NextSignalRequest(BaseModel):
    user_id: str = "demo_user"
    grasp: Optional[str] = Field(default=None, description="Force a grasp, or omit for the next in the queue.")
    inject_fault: Optional[str] = Field(default=None, description="'dropout', 'spike' or 'hum'.")


class NextSignalResponse(Signal):
    user_id: str
    source: str
    true_grasp: str
    recording_index: int
    sample_rate: int
    injected_fault: Optional[str] = None


class PredictRequest(Signal):
    user_id: str = "demo_user"


class AnomalyItem(BaseModel):
    type: str
    channel: str
    severity: str
    message: str


class PredictResponse(BaseModel):
    decision: str                               # "act" | "abstain"
    abstain_reason: Optional[str] = None        # "sensor_fault" | "not_calibrated" | "low_confidence"
    predicted_grasp: Optional[str] = None       # None when a fault blocked classification
    runner_up: Optional[str] = None
    confidence: Optional[float] = None
    confidence_floor: float
    confidence_by_grasp: dict = {}
    window_votes: list[str] = []
    contributions: list[dict] = []
    reasoning: str
    reasoning_source: str
    anomalies: list[AnomalyItem] = []
    latency_ms: float


class FeedbackRequest(Signal):
    user_id: str = "demo_user"
    actual_grasp: str


class CalibrateRequest(BaseModel):
    user_id: str = "demo_user"
    reps: int = Field(default=1, ge=1, le=5, description="Recordings per grasp to calibrate on.")


class ProfileResponse(BaseModel):
    user_id: str
    source: str
    recordings_confirmed: int
    recordings_learned: dict
    calibrated_grasps: int
    overall_accuracy: Optional[float]
    recent_accuracy: Optional[float]
    rolling_accuracy: list[float]
    rolling_window: int
    abstained: int
    accuracy_when_acting: Optional[float]
    grasp_distribution: dict
    unseen_recordings_left: int
    learned: Optional[bool] = None              # set by /api/feedback
    note: Optional[str] = None


class UserListResponse(BaseModel):
    users: list[dict]


class SessionReportResponse(ProfileResponse):
    generated_at: str
    expected_accuracy_by_reps: dict
