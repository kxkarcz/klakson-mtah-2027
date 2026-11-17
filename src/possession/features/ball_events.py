"""Discrete ball events: the instants at which the state of play actually changes.

The decoder reasons about smoothed evidence, which is right for deciding *whether* a stoppage
or a change of possession happened and blurry about *when*. The instants themselves are
visible in the ball track: it crosses a line, it is struck after resting, it comes back into
view, its velocity breaks when somebody touches it. These detectors only locate such instants;
what to do with them is decided elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from possession.config import BallConfig, EventsConfig
from possession.preprocessing.ball import BallTrack
from possession.runs import mask_runs
from possession.structures import FloatArray, IntArray


@dataclass(frozen=True)
class BallEvents:
    """Sorted frame indices of each kind of event."""

    out_of_play: IntArray  # first frame on which the ball is off the pitch, or lost at a line
    restart_kick: IntArray  # first frame on which a resting ball moves again
    reappearance: IntArray  # first tracked frame after a long disappearance
    kink: IntArray  # frames where the ball velocity changes abruptly (a touch)


def line_crossings(ball: BallTrack, margin_m: float) -> IntArray:
    """Frames on which a tracked ball goes from inside the pitch to beyond the line."""
    outside = ball.edge_distance < -margin_m
    inside = ball.edge_distance >= -margin_m
    crossed = np.zeros(outside.size, dtype=bool)
    crossed[1:] = inside[:-1] & outside[1:]
    return np.flatnonzero(crossed)


def lost_at_line(ball: BallTrack, config: EventsConfig, min_gap_frames: int) -> IntArray:
    """First missing frame of a long disappearance that starts at a line, heading out.

    A ball kicked into touch is usually lost by the tracker just as it leaves. Both conditions
    are needed: close to a line, and closer than it was a moment before; a ball lost in the
    middle of the pitch or rolling back infield is an occlusion, not an exit.
    """
    known = ~np.isnan(ball.xy[:, 0])
    starts, stops = mask_runs(~known)
    keep = (starts > config.heading_lag_frames) & (stops - starts >= min_gap_frames)
    starts = starts[keep]
    last = starts - 1
    earlier = last - config.heading_lag_frames
    near_line = ball.edge_distance[last] < config.line_proximity_m
    with np.errstate(invalid="ignore"):
        heading_out = ball.edge_distance[last] < ball.edge_distance[earlier]
    return np.asarray(starts[near_line & heading_out], dtype=np.int64)


def restart_kicks(ball: BallTrack, config: EventsConfig, stationary_speed_mps: float) -> IntArray:
    """First moving frame after the ball has rested, provided it is then really struck.

    Free kicks, corners, goal kicks and kick-offs are all taken from a ball at rest. The speed
    test afterwards separates a kick from a ball that is merely nudged into place.
    """
    resting = ~np.isnan(ball.xy[:, 0]) & (ball.speed < stationary_speed_mps)
    starts, stops = mask_runs(resting)
    keep = (stops - starts >= config.min_rest_frames) & (stops < resting.size)
    stops = stops[keep]
    speed = np.nan_to_num(ball.speed, nan=0.0)
    padded = np.concatenate((speed, np.zeros(config.kick_window_frames)))
    peak = sliding_window_view(padded, config.kick_window_frames).max(axis=1)[: speed.size]
    return np.asarray(stops[peak[stops] >= config.kick_speed_mps], dtype=np.int64)


def reappearances(ball: BallTrack, min_gap_frames: int) -> IntArray:
    """First tracked frame after a disappearance of at least ``min_gap_frames``."""
    known = ~np.isnan(ball.xy[:, 0])
    starts, stops = mask_runs(~known)
    keep = (stops - starts >= min_gap_frames) & (stops < known.size) & (starts > 0)
    return np.asarray(stops[keep], dtype=np.int64)


def velocity_change(xy: FloatArray, fps: float, lag_frames: int) -> FloatArray:
    """Magnitude of the change in ball velocity across each frame, in m/s; NaN if unknown.

    The velocity over the ``lag_frames`` before a frame is compared with the velocity over the
    ``lag_frames`` after it. A free ball changes velocity slowly; a played ball breaks.
    """
    change = np.full(xy.shape[0], np.nan)
    if xy.shape[0] <= 2 * lag_frames:
        return change
    before = xy[lag_frames:-lag_frames] - xy[: -2 * lag_frames]
    after = xy[2 * lag_frames :] - xy[lag_frames:-lag_frames]
    change[lag_frames:-lag_frames] = np.linalg.norm(after - before, axis=1) * (fps / lag_frames)
    return change


def trajectory_kinks(ball: BallTrack, fps: float, config: EventsConfig) -> IntArray:
    """Frames where the velocity change is large and a local maximum (one frame per touch)."""
    change = np.nan_to_num(velocity_change(ball.xy, fps, config.kink_lag_frames), nan=0.0)
    half = config.kink_min_separation_frames
    padded = np.concatenate((np.zeros(half), change, np.zeros(half)))
    local_max = sliding_window_view(padded, 2 * half + 1).max(axis=1)
    is_peak = (change >= config.kink_min_velocity_change_mps) & (change >= local_max)
    # A flat-topped peak yields equal neighbours; keep the first frame of each cluster.
    is_peak[1:] &= ~is_peak[:-1]
    return np.flatnonzero(is_peak)


def restart_candidates(events: BallEvents, ball: BallTrack, config: EventsConfig) -> IntArray:
    """Frames at which play can have restarted: kicks from rest and moving reappearances.

    A ball that comes back into view already at passing speed has just been thrown in or
    kicked; one that reappears at rest is still waiting, and its restart is the kick that
    follows (already a ``restart_kick``).
    """
    speed = np.nan_to_num(ball.speed, nan=0.0)
    padded = np.concatenate((speed, np.zeros(config.kick_window_frames)))
    peak = sliding_window_view(padded, config.kick_window_frames).max(axis=1)[: speed.size]
    moving = events.reappearance[peak[events.reappearance] >= config.kick_speed_mps]
    return np.union1d(events.restart_kick, moving).astype(np.int64)


def detect_ball_events(
    ball: BallTrack, fps: float, config: EventsConfig, ball_config: BallConfig
) -> BallEvents:
    """Run every detector on the cleaned ball track."""
    out = np.union1d(
        line_crossings(ball, config.exit_margin_m),
        lost_at_line(ball, config, ball_config.short_gap_max_frames),
    )
    return BallEvents(
        out_of_play=out.astype(np.int64),
        restart_kick=restart_kicks(ball, config, ball_config.stationary_speed_mps),
        reappearance=reappearances(ball, ball_config.short_gap_max_frames),
        kink=trajectory_kinks(ball, fps, config),
    )
