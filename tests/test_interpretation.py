import pandas as pd
import pytest

from src.rail_interpretation import assess, confidence_wording, describe_feature, explain

BASELINES = {
    "side1_rms": {"p50": 0.09, "p90": 0.13, "p97": 0.17, "p99": 0.22, "max": 0.30},
    "side2_rms": {"p50": 0.09, "p90": 0.13, "p97": 0.17, "p99": 0.22, "max": 0.30},
}


def features(s1, s2, speed=13.0):
    return pd.Series({
        "side1_rms": s1, "side2_rms": s2,
        "side_rms_ratio": s1 / s2, "speed_signal_mean": speed,
        "side1_wavelength_m": 0.06, "side2_wavelength_m": 0.06,
    })


def test_normal_gives_no_action():
    v = assess("Normal", 0.95, features(0.10, 0.10), BASELINES)
    assert v["key"] == "normal"
    assert v["side"] is None
    assert "No action" in v["action"]


@pytest.mark.parametrize("value,expected", [
    (0.12, "low"), (0.15, "moderate"), (0.20, "high"), (0.28, "severe"),
])
def test_severity_scales_with_reading(value, expected):
    v = assess("Side I", 0.95, features(value, 0.09), BASELINES)
    assert v["key"] == expected
    assert "Side I" in v["action"]


def test_low_confidence_downgrades_severe():
    """A borderline call shouldn't on its own trigger the most urgent response."""
    confident = assess("Side I", 0.95, features(0.28, 0.09), BASELINES)
    unsure = assess("Side I", 0.50, features(0.28, 0.09), BASELINES)
    assert confident["key"] == "severe"
    assert unsure["key"] == "high"
    assert unsure["confidence_label"] == "Borderline"


def test_missing_baselines_degrades_gracefully():
    v = assess("Side I", 0.9, features(0.28, 0.09), baselines=None)
    assert v["key"] == "moderate"
    assert v["percentile_text"] == ""
    assert v["has_baselines"] is False


def test_confidence_wording_bands():
    assert confidence_wording(0.95)[0] == "High confidence"
    assert confidence_wording(0.70)[0] == "Moderate confidence"
    assert confidence_wording(0.40)[0] == "Borderline"


def test_explanation_is_plain_language():
    lines = explain("Side I", features(0.30, 0.10))
    text = " ".join(lines)
    assert "3.0×" in text
    assert "Side I" in text
    assert "km/h" in text
    # no raw feature names leaking into operator-facing copy
    assert "side1_rms" not in text
    assert "side_rms_ratio" not in text


def test_describe_feature_maps_channel_to_physical_location():
    # ch74 -> 0-indexed 73 -> car 5, position 5, shock channel
    assert describe_feature("ch74_dominant_freq").startswith("Car 5, position 5 shock")
    assert describe_feature("side_std_ratio").startswith("Side I vs Side II")
    assert "Side I" in describe_feature("side1_band_low_energy")
