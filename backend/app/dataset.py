"""
dataset.py

Loads the model artifacts and the real EMG recordings the dashboard replays.

There is no hardware attached to this demo, so a "wearer" is one of the five
people in the dataset. Each wearer is served by a population model that was
trained on the *other four* people (see training/train.py), so from the
model's point of view every wearer is someone it has never seen before.
"""

import json
import os
from functools import lru_cache

import numpy as np

from .model import GRASPS

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
RECORDINGS_PATH = os.path.join(os.path.dirname(HERE), "data", "recordings.npz")

SOURCE_LABELS = {
    "wearer_a": "Wearer A", "wearer_b": "Wearer B", "wearer_c": "Wearer C",
    "wearer_d": "Wearer D", "wearer_e": "Wearer E",
}
SOURCES = list(SOURCE_LABELS)


@lru_cache(maxsize=1)
def load_model() -> dict:
    with open(os.path.join(ARTIFACTS, "model.json")) as f:
        raw = json.load(f)
    for source in raw["sources"].values():
        source["population_means"] = np.array(source["population_means"])
        source["precision"] = np.array(source["precision"])
    return raw


@lru_cache(maxsize=1)
def load_evaluation() -> dict:
    with open(os.path.join(ARTIFACTS, "evaluation.json")) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _recordings() -> dict:
    with np.load(RECORDINGS_PATH) as z:
        scale = float(z["scale"])
        return {sid: (z[sid], scale) for sid in SOURCES}


def recordings_per_grasp(source: str) -> int:
    return _recordings()[source][0].shape[1]


def get_recording(source: str, grasp: str, index: int) -> np.ndarray:
    """-> (T, 2) float array, one real recording."""
    stored, scale = _recordings()[source]
    return stored[GRASPS.index(grasp), index].astype(float) * scale


def source_model(source: str) -> dict:
    return load_model()["sources"][source]
