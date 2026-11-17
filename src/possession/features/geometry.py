"""Ball-to-player geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.preprocessing.teams import TeamMasks
from possession.structures import BoolArray, FloatArray


@dataclass(frozen=True)
class BallProximity:
    """Distance from the ball to the nearest visible player of each side, in metres.

    ``inf`` when the side has nobody tracked, NaN when the ball position is unknown.
    """

    home: FloatArray
    away: FloatArray


def _nearest(distance: FloatArray, mask: BoolArray, ball_known: BoolArray) -> FloatArray:
    nearest = np.where(mask, distance, np.inf).min(axis=1)
    return np.asarray(np.where(ball_known, nearest, np.nan), dtype=np.float64)


def ball_proximity(player_xy: FloatArray, masks: TeamMasks, ball_xy: FloatArray) -> BallProximity:
    """Nearest-player distance per team.

    Only the team-level minimum is kept: slots are tracklets, so "which player" is not a
    question this data can answer, while "how close is the closest player of each side" is.
    """
    ball_known = ~np.isnan(ball_xy[:, 0])
    delta = player_xy - ball_xy[:, None, :]
    distance = np.sqrt(np.einsum("fsk,fsk->fs", delta, delta))
    return BallProximity(
        home=_nearest(distance, masks.home, ball_known),
        away=_nearest(distance, masks.away, ball_known),
    )
