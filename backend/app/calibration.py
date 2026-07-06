"""
calibration.py

Implements adaptive, per-user calibration — the "AI learns your rhythm"
idea behind real bionic hand systems (e.g. Kawatek's INTELIHAND concept).

Each user has a profile holding a feature-space centroid per grip intent.
Classification measures distance from incoming features to each centroid.
When the user confirms/corrects a prediction (POST /api/feedback), the
centroid for the *actual* intent is nudged toward the observed features
using an exponential moving average. Over repeated sessions, the profile
increasingly reflects that specific user's signal patterns, and accuracy
on held-out predictions should trend upward — visualized as a learning
curve on the frontend.

Profiles are persisted as small JSON files on disk (profiles/<user_id>.json).
This is intentionally simple (no database) since the point of the project
is the AI pipeline and UX, not storage infrastructure.
"""

import json
import os
from statistics import mean

from .features import extract_features
from .signal_simulator import INTENTS, generate_signal

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

FEATURE_KEYS = ["mav", "var", "rms", "wl", "zc"]
LEARNING_RATE = 0.2
SAMPLES_PER_BASE_CENTROID = 25


def _compute_base_centroids() -> dict:
    """Bootstrap starting centroids by averaging features over several
    simulated signals per intent, using fixed seeds so results are
    deterministic across server restarts."""
    centroids = {}
    for intent in INTENTS:
        feats_list = []
        for seed in range(SAMPLES_PER_BASE_CENTROID):
            sample = generate_signal(intent=intent, seed=seed * 7 + 1)
            feats_list.append(extract_features(sample["signal"]))
        centroids[intent] = {
            key: round(mean(f[key] for f in feats_list), 5) for key in FEATURE_KEYS
        }
    return centroids


BASE_CENTROIDS = _compute_base_centroids()


def _profile_path(user_id: str) -> str:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_")) or "default"
    return os.path.join(PROFILES_DIR, f"{safe_id}.json")


def load_profile(user_id: str) -> dict:
    path = _profile_path(user_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            profile = json.load(f)
        # Backward-compatible defaults for profiles saved before a field existed.
        profile.setdefault("intent_distribution", {intent: 0 for intent in INTENTS})
        return profile
    profile = {
        "user_id": user_id,
        "centroids": {k: dict(v) for k, v in BASE_CENTROIDS.items()},
        "session_count": 0,
        "correct_count": 0,
        "total_count": 0,
        "accuracy_history": [],
        "intent_distribution": {intent: 0 for intent in INTENTS},
    }
    save_profile(profile)
    return profile


def save_profile(profile: dict) -> None:
    with open(_profile_path(profile["user_id"]), "w") as f:
        json.dump(profile, f, indent=2)


def list_profiles() -> list[str]:
    """Return user_ids for all users with a saved calibration profile."""
    user_ids = []
    for filename in sorted(os.listdir(PROFILES_DIR)):
        if filename.endswith(".json"):
            with open(os.path.join(PROFILES_DIR, filename), "r") as f:
                try:
                    data = json.load(f)
                    user_ids.append(data.get("user_id", filename[:-5]))
                except json.JSONDecodeError:
                    continue
    return user_ids


def apply_feedback(user_id: str, features: dict, predicted_intent: str, actual_intent: str) -> dict:
    profile = load_profile(user_id)

    # Nudge the centroid of the true intent toward the observed features.
    centroid = profile["centroids"][actual_intent]
    for key in FEATURE_KEYS:
        centroid[key] = round(
            (1 - LEARNING_RATE) * centroid[key] + LEARNING_RATE * features[key], 5
        )

    profile["session_count"] += 1
    profile["total_count"] += 1
    profile["intent_distribution"][actual_intent] = (
        profile["intent_distribution"].get(actual_intent, 0) + 1
    )
    if predicted_intent == actual_intent:
        profile["correct_count"] += 1

    running_accuracy = round(100 * profile["correct_count"] / profile["total_count"], 1)
    profile["accuracy_history"].append(running_accuracy)

    save_profile(profile)
    return profile


def get_profile(user_id: str) -> dict:
    return load_profile(user_id)
