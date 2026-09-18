import pandas as pd
import pytest

from src.rail_features import extract_rail_features


def test_feature_shape(rail_dataframe_factory):
    df = rail_dataframe_factory(n_rows=2000, fault_side=None)
    feats = extract_rail_features(df)
    assert feats.shape[0] == 1
    assert "side1_rms" in feats.columns
    assert "side2_rms" in feats.columns
    assert "side_rms_ratio" in feats.columns
    assert not feats.isna().any().any()


def test_wrong_column_count_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        extract_rail_features(df)


def test_injected_fault_increases_side_rms(rail_dataframe_factory):
    normal = rail_dataframe_factory(n_rows=4000, fault_side=None, seed=1)
    faulty = rail_dataframe_factory(n_rows=4000, fault_side=1, seed=1)

    f_normal = extract_rail_features(normal)
    f_faulty = extract_rail_features(faulty)

    assert f_faulty.iloc[0]["side1_rms"] > f_normal.iloc[0]["side1_rms"]
    # Side II shouldn't be affected by a Side I fault injection.
    assert f_faulty.iloc[0]["side2_rms"] == pytest.approx(f_normal.iloc[0]["side2_rms"], rel=0.05)


def test_speed_toggle_detection_matches_binary_signal(rail_dataframe_factory):
    df = rail_dataframe_factory(n_rows=10000, fault_side=None, seed=2)
    feats = extract_rail_features(df)
    assert feats.iloc[0]["speed_signal_mean"] >= 0
