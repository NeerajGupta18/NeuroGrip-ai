"""
anomaly.py

Signal-quality gate. A prosthetic hand must know when its own sensor input
cannot be trusted, and must refuse to move on it. Three checks run on every
channel of every recording *before* classification:

  dropout  electrode lost skin contact: the trace goes flat. Real EMG never
           gets quieter than the amplifier noise floor, so a stretch whose
           0.1 s standard deviation falls below half the quietest block seen
           in training data is not muscle - it is a disconnected sensor.

  spike    amplitude beyond anything a muscle produced in the training data:
           motion artifact or an electrical glitch hitting the amplifier rail.

  hum      50 Hz mains interference dominating the spectrum (bad grounding or
           a loose reference electrode): the share of signal power sitting in
           the 48-52 Hz band exceeds anything seen in clean recordings.

The three thresholds are *measured* from clean recordings by
training/train.py, not hand-tuned; docs/EVALUATION.md reports the false-alarm
rate on a person whose data was not used to set them.

Every fault is "blocking": the agent abstains instead of guessing (agent.py).
"""

import numpy as np

from .features import CHANNELS, SAMPLE_RATE

BLOCK = 50                    # 0.1 s blocks for the flatline check
MIN_DROPOUT_SECONDS = 0.3
HUM_BAND = (48.0, 52.0)       # Hz


def channel_stats(x: np.ndarray) -> dict:
    """Quality statistics for one channel (1-D array)."""
    x = np.asarray(x, dtype=float)
    x = x - np.median(x)

    n_blocks = len(x) // BLOCK
    block_std = x[: n_blocks * BLOCK].reshape(n_blocks, BLOCK).std(axis=1)

    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    in_band = (freqs >= HUM_BAND[0]) & (freqs <= HUM_BAND[1])
    total = spectrum[freqs >= 5].sum()
    hum_fraction = float(spectrum[in_band].sum() / total) if total > 0 else 0.0

    return {"block_std": block_std, "peak": float(np.abs(x).max()), "hum_fraction": hum_fraction}


def _longest_run(flags: np.ndarray) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def detect_anomalies(signal: np.ndarray, thresholds: dict) -> list[dict]:
    """signal: (T, 2). thresholds: {"flat_std", "peak", "hum_fraction"}."""
    signal = np.asarray(signal, dtype=float)
    anomalies = []
    for idx, channel in enumerate(CHANNELS):
        stats = channel_stats(signal[:, idx])

        flat_seconds = _longest_run(stats["block_std"] < thresholds["flat_std"]) * BLOCK / SAMPLE_RATE
        if flat_seconds >= MIN_DROPOUT_SECONDS:
            anomalies.append({
                "type": "dropout", "channel": channel, "severity": "high",
                "message": (
                    f"Sensor dropout on {channel}: the signal sat below the amplifier noise floor "
                    f"for {flat_seconds:.1f} s. Check electrode contact."
                ),
            })

        if stats["peak"] > thresholds["peak"]:
            anomalies.append({
                "type": "spike", "channel": channel, "severity": "high",
                "message": (
                    f"Amplitude spike on {channel}: peak {stats['peak']:.1f} exceeds the largest "
                    f"muscle activation seen in training ({thresholds['peak']:.1f}). "
                    "Likely motion artifact or electrical glitch."
                ),
            })

        if stats["hum_fraction"] > thresholds["hum_fraction"]:
            anomalies.append({
                "type": "hum", "channel": channel, "severity": "high",
                "message": (
                    f"Mains interference on {channel}: {100 * stats['hum_fraction']:.0f}% of signal "
                    "power sits at 50 Hz. Check grounding / reference electrode."
                ),
            })
    return anomalies
