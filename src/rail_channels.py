"""Canonical sensor layout for the Rail Corrugation dataset.

Per Rail_Corrugation_Info_Kit.md section 2.1: column 0 is rotational
speed; columns 1..128 are vibration/shock readings from 8 cars x 8
axle-box positions, ordered Car1-Pos1-vibration, Car1-Pos1-shock,
Car1-Pos2-vibration, Car1-Pos2-shock, ..., Car1-Pos8-*, ..., Car8-Pos8-*.
Axle positions 1,3,5,7 are Side I; positions 2,4,6,8 are Side II.

This module is the single source of truth for that layout so the
plotting code (app.py) and the feature extraction code (rail_features.py)
never drift apart.
"""

import numpy as np

N_CHANNELS = 128
AXLES_PER_GROUP = 8
CHANNELS_PER_AXLE = 2  # vibration, shock


def axle_position(channel_index0):
    """1-based axle position (1..8) for a 0-indexed channel in [0, 128)."""
    return (channel_index0 // CHANNELS_PER_AXLE) % AXLES_PER_GROUP + 1


def is_vibration_channel(channel_index0):
    """True if this 0-indexed channel is the vibration reading (vs. shock)."""
    return channel_index0 % CHANNELS_PER_AXLE == 0


def side_of_axle(axle_pos):
    """1 for Side I (odd axle positions), 2 for Side II (even)."""
    return 1 if axle_pos % 2 == 1 else 2


def side_channel_indices(side, vibration_only=True):
    """0-indexed channel positions (0..127) belonging to a side.

    side=1 -> Side I, side=2 -> Side II. When vibration_only is True,
    only the vibration channel of each axle is included (used for
    signal plots); when False, both vibration and shock channels are
    included (used for aggregate RMS features).
    """
    indices = []
    for ch in range(N_CHANNELS):
        axle = axle_position(ch)
        if side_of_axle(axle) != side:
            continue
        if vibration_only and not is_vibration_channel(ch):
            continue
        indices.append(ch)
    return indices


def side_dataframe_columns(side, vibration_only=True):
    """Column indices into a raw dataframe (speed in col 0, channels in 1..128)."""
    return [i + 1 for i in side_channel_indices(side, vibration_only=vibration_only)]
