"""Live/dead evidence from a small supervised model.

A logistic regression over features of ball visibility, ball movement, player activity before
and after each frame, team layout and ball events, trained on other matches whose tracking
provider says frame by frame whether the ball is in play (TECHNICAL.md, section 4.2). Its
weights are part of the configuration. When enabled it replaces the per-frame emissions of the
unsupervised live/dead model; the decoder, its costs and everything downstream stay.
"""

from __future__ import annotations

import numpy as np

from possession.config import LearnedLivenessConfig, liveness_feature_count
from possession.dead_ball import N_BALL_CLASSES
from possession.features.ball_events import BallEvents
from possession.features.collective import TeamShape
from possession.learned_ownership import window_mean
from possession.preprocessing.ball import BallTrack
from possession.structures import FloatArray, IntArray, Tracking


def frames_since(mask: np.ndarray) -> IntArray:
    """Frames since ``mask`` was last true (0 where it is true; large before the first)."""
    index = np.where(mask, np.arange(mask.size), -(10**9))
    np.maximum.accumulate(index, out=index)
    return np.asarray(np.arange(mask.size) - index, dtype=np.int64)


def event_recency(
    frames: IntArray, n_frames: int, decay_frames: float, *, latest: bool
) -> FloatArray:
    """``exp(-distance / decay)`` to the latest (or next) event; zero where there is none."""
    out = np.zeros(n_frames)
    if frames.size == 0:
        return out
    index = np.arange(n_frames)
    if latest:
        position = np.searchsorted(frames, index, side="right") - 1
        exists = position >= 0
        distance = index - frames[np.maximum(position, 0)]
    else:
        position = np.searchsorted(frames, index, side="left")
        exists = position < frames.size
        distance = frames[np.minimum(position, frames.size - 1)] - index
    out[exists] = np.exp(-np.maximum(distance[exists], 0) / decay_frames)
    return out


def _ball_columns(
    tracking: Tracking, ball: BallTrack, ball_class: IntArray, config: LearnedLivenessConfig
) -> list[FloatArray]:
    """Ball class, visibility around the frame, gap lengths, speed, edge and nearest player."""
    known = ~np.isnan(ball.xy[:, 0])
    seen = known.astype(np.float64)
    columns = [(ball_class == k).astype(np.float64) for k in range(N_BALL_CLASSES)]
    columns += [window_mean(seen, start, stop) for start, stop in config.visibility_windows_frames]
    cap = config.gap_cap_frames
    since = np.minimum(frames_since(known), cap) / tracking.fps
    until = np.minimum(frames_since(known[::-1])[::-1], cap) / tracking.fps
    columns += [np.log1p(since), np.log1p(until), np.log1p(np.minimum(since, until))]
    speed = np.nan_to_num(ball.speed, nan=0.0)
    columns += [
        np.clip(speed, 0.0, config.ball_speed_cap_mps) / config.ball_speed_scale_mps * seen,
        (speed < config.resting_speed_mps) * seen,
    ]
    edge = np.nan_to_num(ball.edge_distance, nan=0.0)
    columns += [
        np.clip(edge, -config.edge_cap_m / 4.0, config.edge_cap_m) / config.edge_scale_m * seen,
        (edge < config.near_line_m) * seen,
    ]
    delta = tracking.player_xy - ball.xy[:, None, :]
    distance = np.sqrt(np.einsum("fsk,fsk->fs", delta, delta))
    nearest = np.min(np.where(np.isnan(distance), np.inf, distance), axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, config.nearest_cap_m)
    nearest = np.clip(nearest, 0.0, config.nearest_cap_m) / config.nearest_scale_m * seen
    half = config.nearest_window_frames
    columns += [nearest, window_mean(nearest, -half, half + 1)]
    return columns


def _axis_moments(values: FloatArray) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Mean and spread over the tracked players of each frame, and whether any is tracked."""
    present = np.isfinite(values)
    count = np.maximum(present.sum(axis=1), 1)
    filled = np.where(present, values, 0.0)
    mean = filled.sum(axis=1) / count
    spread = np.sqrt(np.maximum((filled**2).sum(axis=1) / count - mean**2, 0.0))
    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(spread, dtype=np.float64),
        np.asarray(present.any(axis=1), dtype=bool),
    )


def _player_columns(
    tracking: Tracking,
    activity_now: FloatArray,
    shapes: tuple[TeamShape, TeamShape],
    config: LearnedLivenessConfig,
) -> list[FloatArray]:
    """Activity around the frame, sprinting share and speed spread, team layout."""
    fallback = float(np.nanmedian(activity_now)) if np.isfinite(activity_now).any() else 0.0
    activity = np.clip(np.nan_to_num(activity_now, nan=fallback), 0.0, config.activity_cap_mps)
    activity = activity / config.activity_scale_mps
    columns = [activity, activity**2]
    columns += [
        window_mean(activity, start, stop) for start, stop in config.activity_windows_frames
    ]
    speeds = tracking.player_speed
    tracked = np.isfinite(speeds)
    count = np.maximum(tracked.sum(axis=1), 1)
    sprinting = np.where(tracked, speeds > config.sprint_speed_mps, False).sum(axis=1) / count
    _, speed_spread, _ = _axis_moments(speeds)
    columns += [sprinting, np.clip(speed_spread, 0.0, config.speed_spread_cap_mps) / 2.0]
    home, away = shapes
    gap = np.linalg.norm(home.centroid - away.centroid, axis=1)
    gap = np.clip(np.nan_to_num(gap, nan=config.team_gap_default_m), 0.0, config.team_gap_cap_m)
    columns.append(gap / config.team_gap_scale_m)
    centre_x, spread_x, any_player = _axis_moments(tracking.player_xy[:, :, 0])
    _, spread_y, _ = _axis_moments(tracking.player_xy[:, :, 1])
    halfway = tracking.pitch_length / 2.0
    centre_x = np.where(any_player, centre_x, halfway)
    columns += [
        spread_x / config.layout_scale_m,
        spread_y / config.layout_scale_m,
        np.abs(centre_x - halfway) / halfway,
    ]
    return columns


def _event_columns(
    events: BallEvents, n_frames: int, config: LearnedLivenessConfig
) -> list[FloatArray]:
    """Memory of the latest, and anticipation of the next, exit, restart kick, reappearance."""
    columns: list[FloatArray] = []
    for frames in (events.out_of_play, events.restart_kick, events.reappearance):
        ordered = np.sort(np.asarray(frames, dtype=np.int64))
        columns += [
            event_recency(ordered, n_frames, decay, latest=True)
            for decay in config.event_latest_decays_frames
        ]
        columns += [
            event_recency(ordered, n_frames, decay, latest=False)
            for decay in config.event_next_decays_frames
        ]
    return columns


def liveness_features(
    tracking: Tracking,
    *,
    ball: BallTrack,
    ball_class: IntArray,
    activity_now: FloatArray,
    shapes: tuple[TeamShape, TeamShape],
    events: BallEvents,
    config: LearnedLivenessConfig,
) -> FloatArray:
    """Feature matrix ``(n_frames, liveness_feature_count(config))``, intercept first; no NaN.

    ``activity_now`` is the collective activity without look-ahead: the windows around each
    frame do the looking ahead and behind.
    """
    ball_part = _ball_columns(tracking, ball, ball_class, config)
    columns = [np.ones(tracking.n_frames), *ball_part[:N_BALL_CLASSES]]
    player_part = _player_columns(tracking, activity_now, shapes, config)
    n_activity = 2 + len(config.activity_windows_frames)
    columns += player_part[:n_activity] + ball_part[N_BALL_CLASSES:]
    columns += player_part[n_activity:] + _event_columns(events, tracking.n_frames, config)
    features = np.nan_to_num(np.stack(columns, axis=1))
    if features.shape[1] != liveness_feature_count(config):
        raise ValueError("liveness features out of step with liveness_feature_count")
    return features


def learned_liveness_emissions(features: FloatArray, config: LearnedLivenessConfig) -> FloatArray:
    """Per-frame log-emissions ``(n_frames, 2)`` over (live, dead), tempered by ``llr_scale``."""
    log_odds_live = features @ np.asarray(config.weights, dtype=np.float64)
    live = -np.logaddexp(0.0, -log_odds_live)
    dead = -np.logaddexp(0.0, log_odds_live)
    return np.asarray(config.llr_scale * np.stack((live, dead), axis=1), dtype=np.float64)
