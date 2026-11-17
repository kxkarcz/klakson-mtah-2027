"""Team-level shape and movement features, usable when the ball is not tracked."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.config import ActivityConfig
from possession.preprocessing.teams import TeamMasks
from possession.runs import centred_mean
from possession.structures import BoolArray, FloatArray, Tracking


def collective_activity(tracking: Tracking, masks: TeamMasks, config: ActivityConfig) -> FloatArray:
    """Smoothed mean speed of all tracked players, in m/s; NaN when too few are tracked.

    When the referee stops play everybody slows to a walk within a second or two, and they do
    so whether or not the ball is tracked. That makes this the one live/dead cue available on
    every frame.
    """
    # A tracked player can still lack a speed (the provider omits it on a tracklet's first
    # frames), so the average runs over the players that have one.
    visible = (masks.home | masks.away) & np.isfinite(tracking.player_speed)
    n_visible = visible.sum(axis=1)
    total = np.where(visible, tracking.player_speed, 0.0).sum(axis=1)
    mean = np.full(tracking.n_frames, np.nan)
    np.divide(total, n_visible, out=mean, where=n_visible >= config.min_players)
    smoothed = centred_mean(mean, config.smoothing_frames)
    if not config.lookahead_enabled:
        return smoothed
    return anticipate_restart(smoothed, config.lookahead_frames)


def anticipate_restart(activity: FloatArray, lookahead_frames: int) -> FloatArray:
    """Replace each value by the larger of itself and the value ``lookahead_frames`` later.

    Players react to a stoppage and to a restart very differently. They slow down within a
    second of the whistle, but after a restart it takes the block several seconds to get back
    up to playing speed (measured on a public match of another game: activity is still below
    the live/dead midpoint 3.6 s after the median restart, against 0.7 s after the median
    stop). Looking a few seconds ahead removes that lag at the restart, and changes nothing
    at the stop, where the present value is the larger one.
    """
    ahead = np.full(activity.size, np.nan)
    if 0 < lookahead_frames < activity.size:
        ahead[:-lookahead_frames] = activity[lookahead_frames:]
    both_missing = np.isnan(activity) & np.isnan(ahead)
    larger = np.fmax(activity, ahead)
    larger[both_missing] = np.nan
    return np.asarray(larger, dtype=np.float64)


@dataclass(frozen=True)
class TeamShape:
    """Centroid and spread of one team per frame (NaN when too few players are tracked)."""

    centroid: FloatArray  # (n_frames, 2), metres
    spread: FloatArray  # (n_frames, 2), per-axis standard deviation in metres


def team_shape(player_xy: FloatArray, mask: BoolArray, min_players: int) -> TeamShape:
    """Centroid and per-axis spread of the visible players of one team."""
    count = mask.sum(axis=1)
    enough = count >= min_players
    safe = np.where(enough, count, 1)[:, None]
    weights = mask[:, :, None]
    xy = np.where(weights, player_xy, 0.0)
    centroid = xy.sum(axis=1) / safe
    variance = np.where(weights, (player_xy - centroid[:, None, :]) ** 2, 0.0).sum(axis=1) / safe
    centroid[~enough] = np.nan
    spread = np.sqrt(variance)
    spread[~enough] = np.nan
    return TeamShape(centroid=centroid, spread=spread)
