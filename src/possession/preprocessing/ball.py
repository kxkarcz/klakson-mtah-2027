"""Ball-track cleaning: outlier rejection, short-gap interpolation and kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.config import BallConfig
from possession.runs import centred_mean, mask_runs
from possession.structures import BoolArray, FloatArray, Tracking


@dataclass(frozen=True)
class BallTrack:
    """The ball as the rest of the pipeline sees it."""

    observed: BoolArray  # the provider reported a (plausible) position on this frame
    xy: FloatArray  # (n, 2) metres; observed positions plus short gaps bridged linearly
    speed: FloatArray  # (n,) m/s, smoothed; NaN where the position is unknown
    edge_distance: FloatArray  # (n,) metres inside the pitch boundary; negative = outside


def reject_teleports(ball_xy: FloatArray, fps: float, max_speed_mps: float) -> FloatArray:
    """Drop isolated detections that would need an impossible speed both to reach and to leave.

    The detector occasionally latches onto a second ball or a white boot for a frame or two.
    Such a point is far from both of its neighbours, whereas a real hard shot is far from at
    most one of them per frame at this threshold.
    """
    cleaned = ball_xy.copy()
    step = np.linalg.norm(np.diff(ball_xy, axis=0), axis=1) * fps
    jump_in = np.concatenate(([False], step > max_speed_mps))
    jump_out = np.concatenate((step > max_speed_mps, [False]))
    cleaned[jump_in & jump_out] = np.nan
    return cleaned


def interpolate_short_gaps(ball_xy: FloatArray, max_gap_frames: int) -> FloatArray:
    """Bridge gaps of at most ``max_gap_frames`` by straight lines between their end points.

    Over half a second a ball cannot do much besides keep flying, so a line is a fair guess;
    over longer gaps it can be received, turned and passed, and inventing a path would feed
    the ownership model fabricated proximity.
    """
    filled = ball_xy.copy()
    missing = np.isnan(ball_xy[:, 0])
    starts, stops = mask_runs(missing)
    bridgeable = (starts > 0) & (stops < ball_xy.shape[0]) & (stops - starts <= max_gap_frames)
    for start, stop in zip(starts[bridgeable], stops[bridgeable], strict=True):
        weight = (np.arange(start, stop) - (start - 1)) / (stop - (start - 1))
        filled[start:stop] = ball_xy[start - 1] + weight[:, None] * (
            ball_xy[stop] - ball_xy[start - 1]
        )
    return filled


def ball_speed(ball_xy: FloatArray, fps: float, smoothing_frames: int) -> FloatArray:
    """Smoothed scalar ball speed in m/s from frame-to-frame displacement."""
    step = np.full(ball_xy.shape[0], np.nan)
    step[1:] = np.linalg.norm(np.diff(ball_xy, axis=0), axis=1) * fps
    # The first frame of a visible run has no predecessor; borrow the following step so a
    # reappearing ball is not reported as having unknown speed exactly when it restarts play.
    head = np.isnan(step) & ~np.isnan(ball_xy[:, 0])
    head[-1] = False
    step[head] = np.concatenate((step[1:], [np.nan]))[head]
    return centred_mean(step, smoothing_frames)


def distance_inside_pitch(xy: FloatArray, pitch_length: float, pitch_width: float) -> FloatArray:
    """Signed distance to the nearest boundary line; negative once the point is off the pitch."""
    distances = np.stack(
        (xy[:, 0], pitch_length - xy[:, 0], xy[:, 1], pitch_width - xy[:, 1]), axis=1
    )
    return np.asarray(distances.min(axis=1), dtype=np.float64)


def clean_ball(tracking: Tracking, config: BallConfig) -> BallTrack:
    """Produce the cleaned ball track used by every downstream stage."""
    plausible = reject_teleports(tracking.ball_xy, tracking.fps, config.max_plausible_speed_mps)
    xy = interpolate_short_gaps(plausible, config.max_interpolation_gap_frames)
    return BallTrack(
        observed=~np.isnan(plausible[:, 0]),
        xy=xy,
        speed=ball_speed(xy, tracking.fps, config.speed_smoothing_frames),
        edge_distance=distance_inside_pitch(xy, tracking.pitch_length, tracking.pitch_width),
    )
