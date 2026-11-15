"""
faults.py

Injects synthetic sensor faults into a *real* recording so the safety gate
can be demonstrated on demand (real faults are rare and not labelled in the
dataset). One randomly chosen channel is corrupted, as a single bad
electrode would be.
"""

from typing import Optional

import numpy as np

from .features import SAMPLE_RATE

FAULT_TYPES = ["dropout", "spike", "hum"]
RAIL = 22.0   # largest amplitude present anywhere in the dataset; spikes go beyond it


def inject_fault(signal: np.ndarray, fault: str, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if fault not in FAULT_TYPES:
        raise ValueError(f"fault must be one of {FAULT_TYPES}")
    rng = rng or np.random.default_rng()
    out = np.array(signal, dtype=float, copy=True)
    n = len(out)
    ch = int(rng.integers(0, out.shape[1]))

    if fault == "dropout":
        # Electrode lifts off: a long stretch collapses to a near-flat line.
        length = int(rng.integers(int(0.35 * n), int(0.55 * n)))
        start = int(rng.integers(0, n - length))
        out[start:start + length, ch] = np.median(out[:, ch]) + rng.normal(0, 0.004, length)

    elif fault == "spike":
        # A few samples slam well past anything a muscle produces.
        idxs = rng.choice(n, size=int(rng.integers(2, 6)), replace=False)
        out[idxs, ch] = rng.choice([-1, 1], size=len(idxs)) * rng.uniform(1.5, 3.0, len(idxs)) * RAIL

    elif fault == "hum":
        # 50 Hz mains pickup several times stronger than the muscle signal.
        t = np.arange(n) / SAMPLE_RATE
        amplitude = rng.uniform(6, 10) * out[:, ch].std()
        out[:, ch] += amplitude * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 2 * np.pi))

    return out
