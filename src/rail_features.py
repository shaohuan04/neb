"""Feature extraction for the Rail Corrugation subsystem.

Two feature families are built from each 1-second, 10 kHz recording:

- Per-channel statistics (time + frequency domain) for all 128 raw
  vibration/shock channels, so the classifier keeps access to the full
  raw signal detail.
- Per-side aggregate statistics computed on the mean *vibration-only*
  waveform for each side (Side I / Side II), matching the engineering
  convention used for the app's signal-evidence plots, plus a small set
  of side-vs-side asymmetry ratios that directly encode the
  Side I vs. Side II localisation signal.

Assumption (stated per the challenge's "state your assumptions" guidance):
side aggregates use vibration channels only, not shock channels, since
that is the physically meaningful "axle-box vibration" signal for
corrugation; per-channel features still cover every raw channel
(vibration and shock) so no information is discarded from the model.
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

from .rail_channels import N_CHANNELS, side_dataframe_columns

SAMPLE_RATE_HZ = 10000
_FREQ_BANDS = {
    "band_low_energy": (0, 500),
    "band_mid_energy": (500, 1500),
    "band_high_energy": (1500, SAMPLE_RATE_HZ / 2),
}
_BASE_STAT_NAMES = [
    "rms", "std", "peak", "ptp", "kurtosis", "skew",
    "dominant_freq", "crest_factor", "spectral_centroid",
]


def _spectrum(x):
    mag = np.abs(rfft(x))
    freqs = rfftfreq(len(x), 1 / SAMPLE_RATE_HZ)
    if len(mag):
        mag[0] = 0.0  # drop DC component
    return freqs, mag


def _signal_stats(x, include_bands=False):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        out = {name: 0.0 for name in _BASE_STAT_NAMES}
        if include_bands:
            out.update({name: 0.0 for name in _FREQ_BANDS})
        return out

    y = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    freqs, mag = _spectrum(y)
    mag_sum = float(np.sum(mag))
    centroid = float(np.sum(freqs * mag) / mag_sum) if mag_sum > 0 else 0.0
    dominant = float(freqs[np.argmax(mag)]) if len(mag) else 0.0

    out = {
        "rms": rms,
        "std": float(np.std(x)),
        "peak": peak,
        "ptp": float(np.ptp(x)),
        "kurtosis": float(kurtosis(x, fisher=False)) if len(x) > 3 else 0.0,
        "skew": float(skew(x)) if len(x) > 2 else 0.0,
        "dominant_freq": dominant,
        "crest_factor": peak / max(rms, 1e-12),
        "spectral_centroid": centroid,
    }
    if include_bands:
        energy_total = float(np.sum(mag ** 2)) or 1e-12
        for name, (lo, hi) in _FREQ_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            out[name] = float(np.sum(mag[mask] ** 2) / energy_total)
    return out


WHEEL_DIAMETER_M = 0.85
WHEEL_TEETH = 90


def _speed_features(raw_speed):
    """Column 0 per the Info Kit: a toothed-wheel speed sensor (90 teeth,
    0.85 m wheel) that toggles 0/1 as teeth pass. If the column looks like
    that raw toggle signal (near-binary), derive speed in m/s by counting
    transitions; otherwise treat the column as an already-computed speed
    value and just summarise it directly. Handling both keeps this robust
    to either raw form the real files turn out to use.
    """
    s = pd.to_numeric(raw_speed, errors="coerce").fillna(0).to_numpy()
    if len(s) == 0:
        return {"speed_mean": 0.0, "speed_std": 0.0}
    if np.unique(s).size <= 3:
        transitions = float(np.sum(np.abs(np.diff((s > 0.5).astype(int)))))
        teeth_per_sec = transitions / 2.0  # each tooth causes one rise + one fall
        rev_per_sec = teeth_per_sec / WHEEL_TEETH
        speed_mps = rev_per_sec * np.pi * WHEEL_DIAMETER_M
        return {"speed_mean": float(speed_mps), "speed_std": 0.0}
    return {"speed_mean": float(np.mean(s)), "speed_std": float(np.std(s))}


def side_mean_signal(df, side, vibration_only=True):
    """Mean waveform across a side's channels (used for plots and features)."""
    cols = [c - 1 for c in side_dataframe_columns(side, vibration_only=vibration_only)]
    signals = df.iloc[:, [c + 1 for c in cols]].apply(pd.to_numeric, errors="coerce")
    return signals.mean(axis=1).fillna(0).to_numpy()


def extract_rail_features(df):
    if df.shape[1] < N_CHANNELS + 1:
        raise ValueError(
            f"Expected rotational speed plus {N_CHANNELS} vibration/shock channels "
            f"({N_CHANNELS + 1} columns total), got {df.shape[1]}."
        )

    signals = df.iloc[:, 1:N_CHANNELS + 1].apply(pd.to_numeric, errors="coerce")
    out = {}

    for ch in range(N_CHANNELS):
        stats = _signal_stats(signals.iloc[:, ch].to_numpy())
        for name, value in stats.items():
            out[f"ch{ch + 1}_{name}"] = value

    side_stats = {}
    for side in (1, 2):
        wave = side_mean_signal(df, side, vibration_only=True)
        stats = _signal_stats(wave, include_bands=True)
        side_stats[side] = stats
        for name, value in stats.items():
            out[f"side{side}_{name}"] = value

    out["side_rms_ratio"] = side_stats[1]["rms"] / max(side_stats[2]["rms"], 1e-12)
    out["side_std_ratio"] = side_stats[1]["std"] / max(side_stats[2]["std"], 1e-12)
    out["side_peak_ratio"] = side_stats[1]["peak"] / max(side_stats[2]["peak"], 1e-12)
    out["side_crest_ratio"] = side_stats[1]["crest_factor"] / max(side_stats[2]["crest_factor"], 1e-12)
    out["side_dominant_freq_diff"] = side_stats[1]["dominant_freq"] - side_stats[2]["dominant_freq"]

    speed_stats = _speed_features(df.iloc[:, 0])
    out["speed_signal_mean"] = speed_stats["speed_mean"]
    out["speed_signal_std"] = speed_stats["speed_std"]

    # Corrugation wavelength is roughly speed-invariant while the raw temporal
    # frequency isn't, so normalising dominant frequency by speed (distance
    # covered per vibration cycle) gives a more physically stable indicator
    # than the raw frequency alone.
    speed = max(speed_stats["speed_mean"], 1e-6)
    out["side1_wavelength_m"] = speed / max(side_stats[1]["dominant_freq"], 1e-6)
    out["side2_wavelength_m"] = speed / max(side_stats[2]["dominant_freq"], 1e-6)

    return pd.DataFrame([out])
