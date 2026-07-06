"""
anomaly.py

Safety layer for NeuroGrip AI. Real bionic hands have to know when their own
sensor input can't be trusted — a dropped electrode or an electrical spike
should never be silently classified as if it were clean data. This mirrors
the health-monitoring angle of real systems like Kawatek's RYO (e.g.
flagging abnormal glucose readings) applied to the signal-quality domain.

Three checks run against every incoming signal window:

- dropout:   a long contiguous stretch of near-zero samples, consistent
             with an electrode losing skin contact.
- spike:     one or more samples at an extreme amplitude, consistent with
             motion artifact or electrical interference.
- noisy:     overall signal variance far above anything a real contraction
             produces, consistent with a loose or poorly-placed sensor.

Thresholds are calibrated against this project's own signal generator so
they don't fire on any of the five legitimate grip signatures, only on
genuinely corrupted windows.
"""

import numpy as np

DROPOUT_RUN_FRACTION = 0.15     # 15%+ of the window near-zero in one run
DROPOUT_ZERO_THRESHOLD = 0.03
SPIKE_AMPLITUDE_THRESHOLD = 1.8  # no legitimate signature exceeds ~1.2
ROUGHNESS_THRESHOLD = 0.55       # point-to-point jaggedness; clean signals top out ~0.35


def _longest_zero_run(signal: np.ndarray, threshold: float) -> int:
    is_near_zero = np.abs(signal) < threshold
    longest = current = 0
    for flag in is_near_zero:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def detect_anomalies(signal: list[float]) -> list[dict]:
    x = np.asarray(signal, dtype=float)
    n = len(x)
    anomalies = []

    longest_run = _longest_zero_run(x, DROPOUT_ZERO_THRESHOLD)
    if longest_run / n >= DROPOUT_RUN_FRACTION:
        anomalies.append({
            "type": "dropout",
            "severity": "high",
            "message": (
                f"Sensor dropout suspected: {longest_run} consecutive near-zero "
                f"samples ({round(100 * longest_run / n)}% of the window). "
                "Check electrode contact before trusting this prediction."
            ),
        })

    max_abs = float(np.max(np.abs(x)))
    if max_abs >= SPIKE_AMPLITUDE_THRESHOLD:
        anomalies.append({
            "type": "spike",
            "severity": "high",
            "message": (
                f"Signal spike detected: peak amplitude reached {round(max_abs, 2)}, "
                "well outside the range of a normal contraction. Likely motion "
                "artifact or electrical interference."
            ),
        })

    # "Roughness" (mean absolute second difference) separates independent
    # sample-to-sample noise from smooth carrier oscillation much more
    # reliably than raw variance, which a strong-but-clean contraction can
    # also produce.
    roughness = float(np.mean(np.abs(np.diff(x, n=2))))
    if roughness >= ROUGHNESS_THRESHOLD:
        anomalies.append({
            "type": "noisy",
            "severity": "medium",
            "message": (
                f"Unusually jagged, high-frequency noise detected (roughness "
                f"{round(roughness, 3)}) — consistent with a loose or poorly-placed "
                "sensor rather than a genuine muscle contraction."
            ),
        })

    return anomalies
