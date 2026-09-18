import numpy as np
import pandas as pd
import pytest

from src.rail_channels import N_CHANNELS, side_dataframe_columns

RAIL_HEADER = ["Rotating speed"]
for car in range(1, 9):
    for pos in range(1, 9):
        RAIL_HEADER.append(f"Vibration of bearing in position {pos} of car {car}")
        RAIL_HEADER.append(f"Shock of bearing in position {pos} of car {car}")


def make_rail_dataframe(n_rows=10000, fault_side=None, fault_amplitude=3.0, seed=0):
    """A synthetic recording matching the real 129-column schema, with an
    optional injected sinusoidal fault on one side's vibration channels.
    Used only for local testing — never the real competition dataset.
    """
    rng = np.random.default_rng(seed)
    speed = rng.integers(0, 2, size=n_rows)
    data = {"Rotating speed": speed}
    t = np.arange(n_rows) / 10000.0
    fault_wave = fault_amplitude * np.sin(2 * np.pi * 800 * t) if fault_side else 0.0

    for ch in range(N_CHANNELS):
        col_name = RAIL_HEADER[ch + 1]
        base = rng.normal(0, 0.3, size=n_rows)
        if fault_side and (ch + 1) in side_dataframe_columns(fault_side, vibration_only=True):
            base = base + fault_wave
        data[col_name] = base

    return pd.DataFrame(data, columns=RAIL_HEADER)


@pytest.fixture
def rail_dataframe_factory():
    return make_rail_dataframe
