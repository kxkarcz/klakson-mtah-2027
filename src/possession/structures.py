"""Typed containers shared across the pipeline.

Kept free of behaviour on purpose: every stage takes and returns plain immutable arrays, so a
stage can be tested with a handful of synthetic frames and cached without pickling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
LabelArray: TypeAlias = NDArray[np.int8]

# Decoder states double as label codes. The order is the alphabet of the submission file.
HOME = 0
AWAY = 1
DEAD = 2
ABSTAIN = 3
LABEL_NAMES: tuple[str, ...] = ("H", "A", "D", "X")
N_STATES = 3


@dataclass(frozen=True)
class Tracking:
    """One recording's tracking, indexed by video frame (row i is submission row i).

    Positions are in metres with the origin at a pitch corner; missing observations are NaN.
    A slot is a tracklet rather than a person, so nothing downstream may follow a slot index
    through time as if it were a player.
    """

    player_xy: FloatArray  # (n_frames, n_slots, 2), metres
    player_speed: FloatArray  # (n_frames, n_slots), m/s as reported by the provider
    slot_team: LabelArray  # (n_slots,), HOME or AWAY
    ball_xy: FloatArray  # (n_frames, 2), metres
    fps: float
    pitch_length: float  # metres, along x
    pitch_width: float  # metres, along y

    @property
    def n_frames(self) -> int:
        """Number of video frames covered."""
        return int(self.player_xy.shape[0])


@dataclass(frozen=True)
class TaggedPhase:
    """One Smart Tagging instance on the video-frame timeline.

    ``team`` is HOME, AWAY or None: the provider tags some phases (every throw-in among them)
    with no team, and that absence has to stay visible rather than be coerced to a side.
    """

    code: str  # upper-cased phase code, e.g. "SET PIECES"
    start_frame: int  # inclusive
    end_frame: int  # exclusive
    team: int | None
    set_piece_type: str | None  # lower-cased "Type" tag, e.g. "throw-in"
