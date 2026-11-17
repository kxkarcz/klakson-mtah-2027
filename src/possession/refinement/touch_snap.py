"""Snap changes of possession to the touch that made them.

The decoder recognises a change of possession from control - a calm ball next to a player -
which shows only after the ball has been won. The winning touch itself is a kink in the ball
trajectory next to a player of the team taking over. On a public match of another game
(Metrica Sample Game 3) the nearest kink lay within 0.2 s of the hand-timed recovery for
65-88% of recoveries, depending on how much the track was smoothed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.config import TouchSnapConfig
from possession.features.geometry import BallProximity
from possession.refinement.dead_ball_edges import nearest_in_window
from possession.runs import label_runs
from possession.structures import DEAD, HOME, IntArray, LabelArray


@dataclass(frozen=True)
class TouchSnap:
    """Refined labels and the shifts applied."""

    labels: LabelArray
    shifts: IntArray  # frames each moved change was shifted by (negative = earlier)


def touches_by_team(kinks: IntArray, proximity: BallProximity, radius_m: float) -> list[IntArray]:
    """Split trajectory kinks into touches by HOME and by AWAY (index = team code).

    A kink belongs to a team when its nearest player is within playing distance of the ball
    and nearer than any opponent; kinks next to nobody (bounces, the far end of a smoothed
    pass) and contested ones belong to neither.
    """
    home = np.nan_to_num(proximity.home[kinks], nan=np.inf)
    away = np.nan_to_num(proximity.away[kinks], nan=np.inf)
    return [kinks[(home <= radius_m) & (home < away)], kinks[(away <= radius_m) & (away < home)]]


def snap_switches_to_touches(
    labels: LabelArray, kinks: IntArray, proximity: BallProximity, config: TouchSnapConfig
) -> TouchSnap:
    """Move each direct H<->A change to the nearest touch by the team taking over.

    A change without such a touch in the window stays where the decoder put it, and no change
    may shorten either possession below ``min_remaining_frames``.
    """
    out = labels.copy()
    touches = touches_by_team(kinks, proximity, config.touch_radius_m)
    starts, stops, values = label_runs(labels)
    shifts: list[int] = []
    for run in range(1, values.size):
        if values[run] == DEAD or values[run - 1] == DEAD:
            continue
        frame = int(starts[run])
        target = nearest_in_window(
            touches[0 if values[run] == HOME else 1],
            frame,
            config.window_before_frames,
            config.window_after_frames,
        )
        earliest = int(starts[run - 1]) + config.min_remaining_frames
        latest = int(stops[run]) - config.min_remaining_frames
        if target is None or target == frame or not earliest <= target <= latest:
            continue
        out[min(frame, target) : max(frame, target)] = (
            values[run] if target < frame else values[run - 1]
        )
        shifts.append(target - frame)
    return TouchSnap(labels=out, shifts=np.array(shifts, dtype=np.int64))
