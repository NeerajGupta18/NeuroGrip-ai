"""
features.py

Extracts classic EMG time-domain features from a raw signal window.
These are the same feature family (MAV, VAR, RMS, ZC, WL) used in real
myoelectric control research, which is why the classifier can work on
purely numerical grounds without needing a trained deep model.
"""

import numpy as np


def extract_features(signal: list[float]) -> dict:
    x = np.asarray(signal, dtype=float)

    mav = float(np.mean(np.abs(x)))                       # Mean Absolute Value
    var = float(np.var(x))                                 # Variance
    rms = float(np.sqrt(np.mean(x ** 2)))                   # Root Mean Square
    wl = float(np.sum(np.abs(np.diff(x))))                  # Waveform Length
    zc = int(np.sum(np.diff(np.sign(x)) != 0))               # Zero-Crossing count

    return {
        "mav": round(mav, 5),
        "var": round(var, 5),
        "rms": round(rms, 5),
        "wl": round(wl, 5),
        "zc": zc,
    }
