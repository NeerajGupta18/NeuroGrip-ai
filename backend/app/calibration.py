"""
calibration.py

Per-wearer personalisation.

A profile stores, for each grasp, the running *sum* of the feature vectors of
every window the wearer has confirmed, plus how many windows that is. From
those two numbers model.personalised_means() rebuilds the wearer's class means
at any time - nothing else about the model changes per person.

Accuracy is tracked "test-then-train": each recording is predicted first, the
result is logged, and only then is it used to update the profile. A profile is
never scored on a recording it has already learned from, because the dashboard
serves each wearer's recordings from a shuffled queue without repeats.
"""

import hashlib
import random
from datetime import datetime, timezone

import numpy as np

from . import store
from .dataset import SOURCES, recordings_per_grasp, source_model
from .features import FEATURE_NAMES
from .model import GRASPS, PRIOR_WINDOWS, personalised_means

ROLLING_WINDOW = 12   # recordings, for the rolling-accuracy curve


def _default_source(user_id: str) -> str:
    digest = hashlib.sha256(user_id.encode()).digest()
    return SOURCES[digest[0] % len(SOURCES)]


def _fresh_queue(source: str, seed: str) -> list:
    queue = [[g, i] for g in GRASPS for i in range(recordings_per_grasp(source))]
    random.Random(seed).shuffle(queue)
    return queue


def create_profile(user_id: str, source: str | None = None) -> dict:
    user_id = store.safe_id(user_id)
    source = source if source in SOURCES else _default_source(user_id)
    if store.get(user_id) is None:
        store.make_room()
    profile = {
        "user_id": user_id,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_sums": [[0.0] * len(FEATURE_NAMES) for _ in GRASPS],
        "window_counts": [0] * len(GRASPS),
        "recordings_learned": {g: 0 for g in GRASPS},
        "outcomes": [],                    # one entry per confirmed recording
        "queue": _fresh_queue(source, user_id),
        "queue_cycles": 0,
    }
    store.save(profile)
    return profile


def get_profile(user_id: str) -> dict:
    return store.get(user_id) or create_profile(user_id)


def reset_profile(user_id: str) -> dict:
    existing = store.get(user_id)
    return create_profile(user_id, existing["source"] if existing else None)


def list_profiles() -> list[dict]:
    profiles = [store.get(uid) for uid in store.list_ids()]
    return [{"user_id": p["user_id"], "source": p["source"]} for p in profiles if p]


def next_recording(profile: dict, grasp: str | None = None) -> tuple[str, int]:
    """Pop the next unseen recording (optionally of a given grasp) from the queue."""
    queue = profile["queue"]
    position = next((i for i, (g, _) in enumerate(queue) if grasp is None or g == grasp), None)
    if position is None:                   # exhausted: start a new cycle (recordings now repeat)
        profile["queue_cycles"] += 1
        queue[:] = _fresh_queue(profile["source"], f"{profile['user_id']}:{profile['queue_cycles']}") + queue
        position = next(i for i, (g, _) in enumerate(queue) if grasp is None or g == grasp)
    g, index = queue.pop(position)
    store.save(profile)
    return g, index


def wearer_means(profile: dict) -> np.ndarray:
    model = source_model(profile["source"])
    return personalised_means(
        model["population_means"], np.array(profile["feature_sums"]),
        np.array(profile["window_counts"]), PRIOR_WINDOWS,
    )


def learn(profile: dict, features: np.ndarray, actual_grasp: str, predicted_grasp: str,
          abstained: bool, confidence: float) -> dict:
    """Log the outcome of a prediction, then fold the recording into the profile."""
    c = GRASPS.index(actual_grasp)
    profile["outcomes"].append({
        "actual": actual_grasp, "predicted": predicted_grasp,
        "correct": predicted_grasp == actual_grasp, "abstained": bool(abstained),
        "confidence": round(float(confidence), 4),
    })
    sums = np.array(profile["feature_sums"])
    sums[c] += features.sum(axis=0)
    profile["feature_sums"] = sums.tolist()
    profile["window_counts"][c] += int(len(features))
    profile["recordings_learned"][actual_grasp] += 1
    store.save(profile)
    return profile


def summarise(profile: dict) -> dict:
    """Everything the dashboard and the report need about a profile."""
    outcomes = profile["outcomes"]
    correct = [o["correct"] for o in outcomes]
    rolling = [
        round(100 * float(np.mean(correct[max(0, i - ROLLING_WINDOW + 1): i + 1])), 1)
        for i in range(len(correct))
    ]
    acted = [o for o in outcomes if not o["abstained"]]
    distribution = {g: 0 for g in GRASPS}
    for o in outcomes:
        distribution[o["actual"]] += 1
    return {
        "user_id": profile["user_id"],
        "source": profile["source"],
        "recordings_confirmed": len(outcomes),
        "recordings_learned": profile["recordings_learned"],
        "calibrated_grasps": sum(1 for n in profile["recordings_learned"].values() if n > 0),
        "overall_accuracy": round(100 * float(np.mean(correct)), 1) if correct else None,
        "recent_accuracy": rolling[-1] if rolling else None,
        "rolling_accuracy": rolling,
        "rolling_window": ROLLING_WINDOW,
        "abstained": len(outcomes) - len(acted),
        "accuracy_when_acting": (
            round(100 * float(np.mean([o["correct"] for o in acted])), 1) if acted else None
        ),
        "grasp_distribution": distribution,
        "unseen_recordings_left": len(profile["queue"]) if profile["queue_cycles"] == 0 else 0,
    }
