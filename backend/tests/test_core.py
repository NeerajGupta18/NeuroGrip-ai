"""Unit tests for the maths: features, classifier, explanation, safety gate."""

import numpy as np
import pytest

from app.anomaly import detect_anomalies
from app.dataset import SOURCES, get_recording, load_evaluation, source_model
from app.faults import FAULT_TYPES, inject_fault
from app.features import FEATURE_NAMES, STEP, WINDOW, extract_features, slice_windows
from app.model import GRASPS, explain, personalised_means, predict, softmax, window_scores


def test_windowing_shape():
    signal = np.random.default_rng(0).normal(size=(3000, 2))
    windows = slice_windows(signal)
    assert windows.shape == ((3000 - WINDOW) // STEP + 1, WINDOW, 2)
    assert extract_features(signal).shape == (len(windows), len(FEATURE_NAMES))


def test_too_short_signal_rejected():
    with pytest.raises(ValueError):
        slice_windows(np.zeros((WINDOW - 1, 2)))


def test_amplitude_feature_tracks_amplitude_not_offset():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(1000, 2))
    quiet, loud, shifted = extract_features(base), extract_features(4 * base), extract_features(base + 5)
    rms = FEATURE_NAMES.index("log_rms_ch1")
    assert np.allclose(loud[:, rms] - quiet[:, rms], np.log(4), atol=1e-3)   # log scale
    assert np.allclose(shifted[:, rms], quiet[:, rms], atol=1e-6)            # DC offset removed


def test_softmax_is_a_distribution():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
    assert np.allclose(p.sum(axis=1), 1) and np.all(p >= 0)


def test_personalised_means_move_from_population_to_wearer():
    population = np.zeros((len(GRASPS), 3))
    wearer_value = 10.0
    no_data = personalised_means(population, np.zeros_like(population), np.zeros(len(GRASPS)), 5.0)
    assert np.allclose(no_data, population)
    n = 500
    lots = personalised_means(population, np.full_like(population, wearer_value * n), np.full(len(GRASPS), n), 5.0)
    assert np.allclose(lots, wearer_value, atol=0.15)


def test_explanation_adds_up_exactly_to_the_margin():
    model = source_model("wearer_a")
    X = extract_features(get_recording("wearer_a", "cyl", 0))
    means, P = model["population_means"], model["precision"]
    scores = window_scores(X, means, P).mean(axis=0)
    order = np.argsort(scores)[::-1]
    contributions, margin = explain(X.mean(axis=0), means, P, int(order[0]), int(order[1]))
    assert margin == pytest.approx(scores[order[0]] - scores[order[1]], abs=1e-6)
    assert sum(c["contribution"] for c in contributions) == pytest.approx(margin, abs=1e-3)


def test_prediction_fields():
    model = source_model("wearer_b")
    p = predict(extract_features(get_recording("wearer_b", "hook", 3)), model["population_means"],
                model["precision"], model["evidence_scale"])
    assert p.grasp in GRASPS and p.runner_up in GRASPS and p.grasp != p.runner_up
    assert sum(p.confidence_by_grasp.values()) == pytest.approx(1, abs=1e-2)
    assert len(p.window_votes) == 23


@pytest.mark.parametrize("source", SOURCES)
def test_calibration_recovers_accuracy_on_an_unseen_wearer(source):
    """The headline claim, re-checked from the shipped artifacts: a population model is poor on
    a new wearer, and two recordings per grasp make it good."""
    model = source_model(source)
    means, P = model["population_means"], model["precision"]
    feats = {g: [extract_features(get_recording(source, g, i)) for i in range(30)] for g in GRASPS}

    def accuracy(class_means, start):
        hits = total = 0
        for c, g in enumerate(GRASPS):
            for X in feats[g][start:]:
                hits += window_scores(X, class_means, P).mean(axis=0).argmax() == c
                total += 1
        return hits / total

    sums = np.stack([np.concatenate(feats[g][:2]).sum(axis=0) for g in GRASPS])
    counts = np.array([sum(len(X) for X in feats[g][:2]) for g in GRASPS])
    before, after = accuracy(means, 2), accuracy(personalised_means(means, sums, counts), 2)
    assert before < 0.55
    assert after > 0.75 and after > before + 0.25


def test_clean_recordings_pass_the_safety_gate():
    thresholds = source_model("wearer_c")["thresholds"]
    flagged = sum(bool(detect_anomalies(get_recording("wearer_c", g, i), thresholds))
                  for g in GRASPS for i in range(30))
    assert flagged <= 2          # evaluation: 1 false alarm in 900 across all wearers


@pytest.mark.parametrize("fault", FAULT_TYPES)
def test_injected_faults_are_caught(fault):
    rng = np.random.default_rng(3)
    thresholds = source_model("wearer_a")["thresholds"]
    for i in range(10):
        corrupted = inject_fault(get_recording("wearer_a", GRASPS[i % 6], i), fault, rng)
        assert any(a["type"] == fault for a in detect_anomalies(corrupted, thresholds))


def test_evaluation_report_is_internally_consistent():
    ev = load_evaluation()
    curve = ev["calibration_curve"]["reps_per_grasp"]
    assert curve["0"]["recording"] == ev["unseen_wearer_no_calibration"]["recording"]
    assert curve["1"]["recording"] > curve["0"]["recording"] + 30
    assert ev["within_wearer"]["recording"] > 90
