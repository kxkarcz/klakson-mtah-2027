"""Per-frame team membership of the tracked objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.structures import AWAY, HOME, BoolArray, Tracking


@dataclass(frozen=True)
class TeamMasks:
    """Which slots hold a visible player of each side on each frame."""

    home: BoolArray  # (n_frames, n_slots)
    away: BoolArray  # (n_frames, n_slots)


def assign_slots_to_teams(tracking: Tracking) -> TeamMasks:
    """Split the visible slots by team.

    The provider assigns every tracklet slot to a team for the whole match; an identity can
    change inside a slot but its side cannot, which is why the model is built on team-level
    quantities only and never on a slot's history.
    """
    visible = ~np.isnan(tracking.player_xy[:, :, 0])
    return TeamMasks(
        home=visible & (tracking.slot_team == HOME)[None, :],
        away=visible & (tracking.slot_team == AWAY)[None, :],
    )
