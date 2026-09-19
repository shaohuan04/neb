"""Translate model output into language an operator can act on.

The classifier answers "which class"; an operator needs "how bad, how
urgent, what do I do, and why should I believe it". Everything in this
module exists to bridge that gap.

The severity bands and maintenance timeframes below are an operating
policy for this prototype, not certified engineering thresholds. They
are defined here in one place so a rail operator can tune them to their
own maintenance standards.
"""

SIDE_AXLES = {"Side I": "1, 3, 5, 7", "Side II": "2, 4, 6, 8"}

SEVERITY_BANDS = [
    # (key, label, action, timeframe, tone)
    ("severe", "Severe", "Raise a maintenance order and inspect the {side} rail", "Within 72 hours", "bad"),
    ("high", "High", "Schedule rail grinding for the {side} rail", "Within 2 weeks", "bad"),
    ("moderate", "Moderate", "Add the {side} rail to the watch list and re-measure on the next pass", "Within 4 weeks", "warn"),
    ("low", "Low", "Log the {side} reading and re-measure on the next scheduled pass", "Next routine pass", "warn"),
]

NORMAL_OUTCOME = {
    "key": "normal",
    "label": "Normal",
    "action": "No action required — continue routine monitoring",
    "timeframe": "Next scheduled inspection",
    "tone": "ok",
    "headline": "Both rails healthy",
    "percentile_text": "",
}

FEATURE_GLOSSARY = {
    "rms": "overall vibration energy",
    "std": "how much the vibration varies",
    "peak": "the single strongest jolt",
    "ptp": "the full swing from lowest to highest",
    "kurtosis": "how 'spiky' the signal is — high values mean sharp impacts",
    "skew": "whether jolts lean positive or negative",
    "dominant_freq": "the frequency the rail is ringing at",
    "crest_factor": "how extreme the peaks are versus the average",
    "spectral_centroid": "whether energy sits in high or low frequencies",
    "band_low_energy": "share of energy below 500 Hz",
    "band_mid_energy": "share of energy from 500-1500 Hz",
    "band_high_energy": "share of energy above 1500 Hz",
    "wavelength_m": "spacing of the wear pattern along the rail",
}


def describe_feature(name):
    """Turn a feature name like 'ch74_dominant_freq' into plain English."""
    if name.startswith("side_"):
        stat = name.replace("side_", "").replace("_ratio", "").replace("_diff", "")
        readable = FEATURE_GLOSSARY.get(stat, stat.replace("_", " "))
        return f"Side I vs Side II — {readable}"
    if name.startswith("side1_") or name.startswith("side2_"):
        side = "Side I" if name.startswith("side1_") else "Side II"
        stat = name.split("_", 1)[1]
        return f"{side} — {FEATURE_GLOSSARY.get(stat, stat.replace('_', ' '))}"
    if name.startswith("ch"):
        channel, _, stat = name.partition("_")
        num = channel[2:]
        car = (int(num) - 1) // 16 + 1 if num.isdigit() else "?"
        pos = ((int(num) - 1) // 2) % 8 + 1 if num.isdigit() else "?"
        kind = "vibration" if num.isdigit() and int(num) % 2 == 1 else "shock"
        return f"Car {car}, position {pos} {kind} — {FEATURE_GLOSSARY.get(stat, stat.replace('_', ' '))}"
    if name.startswith("speed"):
        return "Train speed during the recording"
    return name.replace("_", " ")


def confidence_wording(confidence):
    """Plain-language reliability, not a bare percentage."""
    if confidence >= 0.85:
        return "High confidence", "The model is clear about this result."
    if confidence >= 0.65:
        return "Moderate confidence", "Reasonably clear, but worth a second look if it drives major work."
    return "Borderline", "The model is unsure. Verify manually before acting on this."


def _band_from_percentile(value, stats):
    """Which healthy-population band does this reading fall into?"""
    if value > stats["p99"]:
        return "severe", "higher than 99% of healthy recordings"
    if value > stats["p97"]:
        return "high", "higher than 97% of healthy recordings"
    if value > stats["p90"]:
        return "moderate", "higher than 90% of healthy recordings"
    return "low", "within the normal range for healthy recordings"


def assess(prediction, confidence, features, baselines=None):
    """Full operator-facing assessment: severity, action, urgency, wording.

    baselines comes from the trained model's metadata (healthy_baselines).
    Without it, severity grading is unavailable and we say so rather than
    inventing a number.
    """
    conf_label, conf_note = confidence_wording(confidence)

    if prediction == "Normal":
        outcome = dict(NORMAL_OUTCOME)
        outcome.update({"confidence_label": conf_label, "confidence_note": conf_note, "side": None})
        return outcome

    side = prediction
    rms_feature = "side1_rms" if side == "Side I" else "side2_rms"
    value = float(features[rms_feature])

    stats = (baselines or {}).get(rms_feature)
    if stats:
        band_key, percentile_text = _band_from_percentile(value, stats)
    else:
        # No baselines in metadata (model trained before severity support).
        band_key, percentile_text = "moderate", ""

    # A borderline call shouldn't trigger the most urgent response on its own.
    if band_key == "severe" and confidence < 0.65:
        band_key = "high"

    band = next(b for b in SEVERITY_BANDS if b[0] == band_key)
    _, label, action, timeframe, tone = band

    return {
        "key": band_key,
        "label": label,
        "action": action.format(side=side),
        "timeframe": timeframe,
        "tone": tone,
        "headline": f"Corrugation detected on the {side} rail",
        "percentile_text": percentile_text,
        "confidence_label": conf_label,
        "confidence_note": conf_note,
        "side": side,
        "axles": SIDE_AXLES.get(side, ""),
        "has_baselines": bool(stats),
    }


def explain(prediction, features):
    """Why the model reached this conclusion, in plain English."""
    s1 = float(features["side1_rms"])
    s2 = float(features["side2_rms"])
    lines = []

    if prediction == "Normal":
        ratio = max(s1, s2) / max(min(s1, s2), 1e-12)
        lines.append(
            f"Both rails are vibrating at a similar level (Side I {s1:.3f} vs Side II {s2:.3f} m/s², "
            f"a difference of just {abs(ratio - 1) * 100:.0f}%). Corrugation on one rail shows up as a "
            "clear imbalance between the two sides, and there isn't one here."
        )
    else:
        high, low = (s1, s2) if prediction == "Side I" else (s2, s1)
        other = "Side II" if prediction == "Side I" else "Side I"
        factor = high / max(low, 1e-12)
        lines.append(
            f"{prediction} is vibrating {factor:.1f}× harder than {other} "
            f"({high:.3f} vs {low:.3f} m/s²). That side-to-side imbalance is the signature of "
            f"corrugation — a repeating wear pattern on the {prediction} rail."
        )

    speed = float(features.get("speed_signal_mean", 0) or 0)
    if speed > 0.5:
        lines.append(f"Recorded at roughly {speed:.0f} m/s ({speed * 3.6:.0f} km/h).")
        wl_feature = "side1_wavelength_m" if prediction == "Side I" else "side2_wavelength_m"
        wavelength = float(features.get(wl_feature, 0) or 0)
        if prediction != "Normal" and 0.01 < wavelength < 2.0:
            lines.append(
                f"The dominant wear spacing works out to about {wavelength * 100:.0f} cm along the rail, "
                "which indicates the grinding profile needed."
            )
    return lines
