"""Ownership emissions: which team the ball-player geometry says is in control."""

from __future__ import annotations

import numpy as np

from possession.config import OwnershipConfig, TouchEvidenceConfig
from possession.features.geometry import BallProximity
from possession.structures import FloatArray, IntArray


def control_log_odds(
    proximity: BallProximity, ball_speed: FloatArray, config: OwnershipConfig
) -> FloatArray:
    """Per-frame log-odds that HOME rather than AWAY controls the ball (positive = home).

    Three factors, each a football statement:

    * *side* - the team whose nearest player is clearly closer than the opponent's. Inside the
      duel margin both can play the ball and the frame says nothing.
    * *reach* - full weight while somebody is within playing distance, fading to zero by the
      reach radius: a ball nobody can touch is a pass or a loose ball, and possession stays
      where it was.
    * *ball speed* - a ball travelling at passing speed past a player is not under control, so
      fast frames are down-weighted; control shows as a ball moving at running pace.

    Frames without a ball position contribute exactly zero.
    """
    home = np.nan_to_num(proximity.home, nan=np.inf)
    away = np.nan_to_num(proximity.away, nan=np.inf)
    nearest = np.minimum(home, away)
    with np.errstate(invalid="ignore"):
        gap = np.nan_to_num(away - home, nan=0.0, posinf=np.inf, neginf=-np.inf)
    beyond_duel = np.sign(gap) * np.maximum(np.abs(gap) - config.duel_margin_m, 0.0)
    side = np.tanh(beyond_duel / config.distance_scale_m)
    reach = np.clip(
        (config.reach_radius_m - nearest) / (config.reach_radius_m - config.control_radius_m),
        0.0,
        1.0,
    )
    speed = np.nan_to_num(ball_speed, nan=config.ball_speed_scale_mps)
    calm = 1.0 / (1.0 + (speed / config.ball_speed_scale_mps) ** 2)
    return np.asarray(config.control_llr_per_frame * side * reach * calm, dtype=np.float64)


def touch_log_odds(
    touches: IntArray, proximity: BallProximity, config: TouchEvidenceConfig
) -> FloatArray:
    """Log-odds pulses (positive = home) on the frames where the ball was touched.

    A touch is credited to the team whose nearest player is within the touch radius and
    closer than the opponent's by more than the duel margin. The pulse sits on the touch frame
    itself, so the decoder dates a change of possession at the touch that caused it.
    """
    log_odds = np.zeros(proximity.home.shape[0])
    home = np.nan_to_num(proximity.home[touches], nan=np.inf)
    away = np.nan_to_num(proximity.away[touches], nan=np.inf)
    by_home = (home <= config.touch_radius_m) & (away - home > config.duel_margin_m)
    by_away = (away <= config.touch_radius_m) & (home - away > config.duel_margin_m)
    log_odds[touches[by_home]] = config.llr_per_touch
    log_odds[touches[by_away]] = -config.llr_per_touch
    return log_odds


def ownership_log_emissions(log_odds_home: FloatArray) -> FloatArray:
    """Turn home-vs-away log-odds into per-team log-emissions, shape ``(n_frames, 2)``.

    The favoured team scores 0 and the other ``-|log-odds|``. Anchoring the favoured side at
    zero keeps ownership evidence from leaking into the live/dead decision: a player standing
    over a dead ball must not make the frame look more "live" than an empty pitch would.
    """
    return np.stack((np.minimum(log_odds_home, 0.0), np.minimum(-log_odds_home, 0.0)), axis=1)
