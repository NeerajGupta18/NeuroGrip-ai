"""
signal_simulator.py

Generates synthetic myoelectric (EMG-like) signal windows for five grip
intents. Real EMG signals are noisy, roughly zero-mean electrical readings
whose *envelope* (amplitude over time) and *frequency content* change with
muscle contraction type. We approximate that behaviour with parametrized
noise so the rest of the pipeline (feature extraction, classification,
calibration) can be built and demoed without physical hardware.

This is explicitly a simulation of signal *characteristics*, not a
biophysical model — see README for the real-world framing.
"""

import numpy as np
from typing import Optional

INTENTS = ["grasp", "release", "hold_firm", "gentle_hold", "pinch"]

# Each signature describes the *shape* of muscle activation for that intent:
#   base_amp    -> overall contraction strength (envelope height)
#   noise_std   -> baseline signal noise
#   freq        -> dominant oscillation frequency (Hz-equivalent, arbitrary units)
#   envelope    -> "burst" (rises then decays), "sustain" (flat plateau), or
#                  "pulse" (short repeated bursts)
SIGNATURES = {
    "grasp":       {"base_amp": 0.85, "noise_std": 0.12, "freq": 6,  "envelope": "burst"},
    "release":     {"base_amp": 0.35, "noise_std": 0.10, "freq": 5,  "envelope": "decay"},
    "hold_firm":   {"base_amp": 0.90, "noise_std": 0.06, "freq": 3,  "envelope": "sustain"},
    "gentle_hold": {"base_amp": 0.40, "noise_std": 0.05, "freq": 3,  "envelope": "sustain"},
    "pinch":       {"base_amp": 0.55, "noise_std": 0.15, "freq": 14, "envelope": "pulse"},
}

WINDOW_SIZE = 200  # samples per signal window

ANOMALY_TYPES = ["dropout", "spike", "noisy"]


def _inject_anomaly(signal: np.ndarray, anomaly: str, rng: np.random.Generator) -> np.ndarray:
    """Corrupt an otherwise-clean signal to simulate a real sensor fault.

    - dropout: a contiguous stretch goes to ~zero, as if the electrode lost
      skin contact.
    - spike: a few samples jump to an extreme amplitude, as if from motion
      artifact or electrical interference.
    - noisy: the whole window gets heavy additional noise, as if from poor
      electrode contact or a loose connection.
    """
    signal = signal.copy()
    n = len(signal)

    if anomaly == "dropout":
        start = rng.integers(0, int(n * 0.4))
        length = rng.integers(int(n * 0.35), int(n * 0.55))
        end = min(n, start + length)
        signal[start:end] = rng.normal(0, 0.01, end - start)

    elif anomaly == "spike":
        num_spikes = rng.integers(2, 5)
        idxs = rng.choice(n, size=num_spikes, replace=False)
        spike_signs = rng.choice([-1, 1], size=num_spikes)
        signal[idxs] = spike_signs * rng.uniform(2.2, 3.2, size=num_spikes)

    elif anomaly == "noisy":
        signal = signal + rng.normal(0, 0.55, n)

    return signal


def _envelope_curve(shape: str, n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    if shape == "burst":
        # rises quickly, decays slowly
        return np.exp(-((t - 0.25) ** 2) / 0.04)
    if shape == "decay":
        return np.exp(-4 * t)
    if shape == "sustain":
        # ramps up then holds flat
        ramp = np.clip(t / 0.15, 0, 1)
        return ramp
    if shape == "pulse":
        return 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3 * t))
    return np.ones(n)


def generate_signal(
    intent: Optional[str] = None,
    seed: Optional[int] = None,
    inject_anomaly: Optional[str] = None,
) -> dict:
    """Generate one simulated signal window.

    If `intent` is None, a random intent is chosen (useful for demo/training
    mode). Returns the raw signal plus the ground-truth intent so the
    frontend can later ask the user to confirm/correct the prediction.

    `inject_anomaly` optionally corrupts the clean signal (see
    `ANOMALY_TYPES`) to exercise the safety/anomaly-detection layer on demand
    for demo purposes — real sensor faults are rare, so this makes them
    reproducible.
    """
    rng = np.random.default_rng(seed)
    if intent is None:
        intent = rng.choice(INTENTS)
    if intent not in SIGNATURES:
        raise ValueError(f"Unknown intent '{intent}'. Must be one of {INTENTS}")
    if inject_anomaly is not None and inject_anomaly not in ANOMALY_TYPES:
        raise ValueError(f"Unknown anomaly '{inject_anomaly}'. Must be one of {ANOMALY_TYPES}")

    sig = SIGNATURES[intent]
    n = WINDOW_SIZE
    envelope = _envelope_curve(sig["envelope"], n) * sig["base_amp"]

    t = np.linspace(0, 1, n)
    carrier = np.sin(2 * np.pi * sig["freq"] * t)
    noise = rng.normal(0, sig["noise_std"], n)

    signal = envelope * carrier + noise

    if inject_anomaly is not None:
        signal = _inject_anomaly(signal, inject_anomaly, rng)

    return {
        "intent": intent,
        "signal": signal.round(5).tolist(),
        "window_size": n,
        "injected_anomaly": inject_anomaly,
    }
