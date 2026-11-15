"""
features.py

Turns raw two-channel surface-EMG into the feature vectors the classifier sees.

A recording is cut into overlapping windows (0.5 s long, 0.25 s apart at
500 Hz). For every window and every channel we compute six classic
time-domain EMG features, so one window becomes a 12-number vector:

    log_rms     log of root-mean-square amplitude  -> how hard the muscle fires
    log_wl      log of mean waveform length        -> amplitude x frequency
    zc_rate     zero-crossing rate                 -> dominant frequency
    ssc_rate    slope-sign-change rate             -> high-frequency content
    mobility    Hjorth mobility                    -> mean frequency
    complexity  Hjorth complexity                  -> how "non-sinusoidal" it is

Amplitude features are logged because EMG amplitude is roughly log-normal:
after the log the classes look far more Gaussian, which is exactly what the
classifier in model.py assumes.

Everything here is plain NumPy so the deployed API has no heavy dependencies.
"""

import numpy as np

SAMPLE_RATE = 500          # Hz, fixed by the dataset
WINDOW = 250               # samples = 0.5 s
STEP = 125                 # samples = 0.25 s (50% overlap)
NOISE_FLOOR = 0.02         # amplitude below which a sign change is ignored

CHANNELS = ["ch1", "ch2"]
FEATURE_KINDS = ["log_rms", "log_wl", "zc_rate", "ssc_rate", "mobility", "complexity"]
# Order matches the columns produced by window_features().
FEATURE_NAMES = [f"{kind}_{ch}" for kind in FEATURE_KINDS for ch in CHANNELS]

FEATURE_LABELS = {
    "log_rms": "signal amplitude",
    "log_wl": "waveform length",
    "zc_rate": "zero-crossing rate",
    "ssc_rate": "slope-change rate",
    "mobility": "mean frequency (mobility)",
    "complexity": "waveform complexity",
}
CHANNEL_LABELS = {"ch1": "channel 1", "ch2": "channel 2"}


def slice_windows(signal: np.ndarray, window: int = WINDOW, step: int = STEP) -> np.ndarray:
    """(T, 2) recording -> (n_windows, window, 2) stack of overlapping windows."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or signal.shape[1] != len(CHANNELS):
        raise ValueError("signal must have shape (T, 2)")
    if len(signal) < window:
        raise ValueError(f"signal needs at least {window} samples, got {len(signal)}")
    starts = range(0, len(signal) - window + 1, step)
    return np.stack([signal[s:s + window] for s in starts])


def window_features(windows: np.ndarray) -> np.ndarray:
    """(n, L, 2) windows -> (n, 12) feature matrix, columns as FEATURE_NAMES."""
    w = windows - windows.mean(axis=1, keepdims=True)      # remove DC offset
    d1 = np.diff(w, axis=1)                                 # first difference
    d2 = np.diff(d1, axis=1)                                # second difference

    log_rms = np.log(np.sqrt((w ** 2).mean(axis=1)) + 1e-6)
    log_wl = np.log(np.abs(d1).mean(axis=1) + 1e-6)

    crosses_zero = (w[:, :-1] * w[:, 1:] < 0) & (np.abs(d1) > NOISE_FLOOR)
    zc_rate = crosses_zero.mean(axis=1)

    slope_flips = (d1[:, :-1] * d1[:, 1:] < 0) & (
        (np.abs(d1[:, :-1]) > NOISE_FLOOR) | (np.abs(d1[:, 1:]) > NOISE_FLOOR)
    )
    ssc_rate = slope_flips.mean(axis=1)

    var0 = w.var(axis=1) + 1e-9
    var1 = d1.var(axis=1) + 1e-9
    var2 = d2.var(axis=1) + 1e-9
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility

    return np.concatenate([log_rms, log_wl, zc_rate, ssc_rate, mobility, complexity], axis=1)


def extract_features(signal: np.ndarray) -> np.ndarray:
    """(T, 2) recording -> (n_windows, 12) feature matrix."""
    return window_features(slice_windows(signal))
