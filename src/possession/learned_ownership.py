"""Ownership log-odds from a small supervised model.

A logistic regression over features of the ball-player geometry and of the touches, trained
on other matches whose tracking provider labels possession frame by frame (TECHNICAL.md,
section 4.2). Its weights are part of the configuration, so inference needs no file beyond
the YAML and no data from those matches.

Every feature is antisymmetric: swapping the two teams negates it. The model has no intercept,
so swapping the teams negates its output as well and neither side is favoured by construction.
"""

from __future__ import annotations

import itertools

import numpy as np

from possession.config import LearnedOwnershipConfig, TouchEvidenceConfig
from possession.preprocessing.ball import BallTrack
from possession.preprocessing.teams import TeamMasks
from possession.structures import BoolArray, FloatArray, IntArray

# Stand-in for "nobody there" in differences of distances; any value far beyond every radius.
_FAR_M = 20.0


def window_mean(values: FloatArray, start: int, stop: int) -> FloatArray:
    """Mean of ``values`` over frames ``[t + start, t + stop)``, zero beyond the ends."""
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(values.size)
    low = np.clip(index + start, 0, values.size)
    high = np.clip(index + stop, 0, values.size)
    return np.asarray((cumulative[high] - cumulative[low]) / (stop - start), dtype=np.float64)


def touch_recency(
    frames: IntArray, signs: FloatArray, n_frames: int, decay_frames: float, *, latest: bool
) -> FloatArray:
    """Sign of the latest (or the next) credited touch, fading with the distance in frames.

    ``decay_frames`` may be infinite, which carries the sign unchanged. Zero where there is
    no such touch.
    """
    out = np.zeros(n_frames)
    if frames.size == 0:
        return out
    index = np.arange(n_frames)
    if latest:
        position = np.searchsorted(frames, index, side="right") - 1
        exists = position >= 0
        position = np.maximum(position, 0)
        distance = index - frames[position]
    else:
        position = np.searchsorted(frames, index, side="left")
        exists = position < frames.size
        position = np.minimum(position, frames.size - 1)
        distance = frames[position] - index
    fade = np.exp(-np.maximum(distance, 0) / decay_frames) if np.isfinite(decay_frames) else 1.0
    out[exists] = (signs[position] * fade)[exists]
    return out


def credited_touches(
    touches: IntArray,
    home_nearest: FloatArray,
    away_nearest: FloatArray,
    config: TouchEvidenceConfig,
) -> tuple[IntArray, FloatArray]:
    """Touches that can be credited to one team, with +1 for home and -1 for away."""
    home, away = home_nearest[touches], away_nearest[touches]
    by_home = (home <= config.touch_radius_m) & (away - home > config.duel_margin_m)
    by_away = (away <= config.touch_radius_m) & (home - away > config.duel_margin_m)
    credited = by_home | by_away
    return touches[credited], np.where(by_home[credited], 1.0, -1.0)


def _two_nearest(distance: FloatArray, mask: BoolArray) -> tuple[FloatArray, FloatArray]:
    own = np.where(mask, distance, np.inf)
    nearest_two = np.partition(own, 1, axis=1)[:, :2]
    return nearest_two[:, 0], nearest_two[:, 1]


def _signed_gap(theirs: FloatArray, ours: FloatArray) -> FloatArray:
    """``theirs - ours`` with missing sides replaced by a far distance."""
    return np.asarray(np.minimum(theirs, _FAR_M) - np.minimum(ours, _FAR_M), dtype=np.float64)


def ownership_features(
    player_xy: FloatArray,
    masks: TeamMasks,
    ball: BallTrack,
    touches: IntArray,
    *,
    config: LearnedOwnershipConfig,
    touch_config: TouchEvidenceConfig,
) -> FloatArray:
    """Feature matrix ``(n_frames, n_features)``; positive values speak for HOME.

    Geometry features are zero on frames without a ball position; touch features carry
    across such frames, which is what lets the model keep a possession through a gap.
    """
    n_frames = player_xy.shape[0]
    known = ~np.isnan(ball.xy[:, 0])
    delta = player_xy - ball.xy[:, None, :]
    distance = np.sqrt(np.einsum("fsk,fsk->fs", delta, delta))
    distance = np.where(np.isnan(distance), np.inf, distance)
    home, home_second = _two_nearest(distance, masks.home)
    away, away_second = _two_nearest(distance, masks.away)
    speed = np.nan_to_num(ball.speed, nan=0.0)
    seen = known.astype(np.float64)
    columns: list[FloatArray] = []

    def alone(radius: float) -> FloatArray:
        home_in, away_in = home < radius, away < radius
        value = (home_in & ~away_in).astype(np.float64) - (away_in & ~home_in).astype(np.float64)
        return np.asarray(value * seen, dtype=np.float64)

    columns += [alone(radius) for radius in config.radii_m]
    edges = (-np.inf, *config.speed_band_edges_mps, np.inf)
    for radius in config.banded_radii_m:
        base = alone(radius)
        columns += [
            base * ((speed >= low) & (speed < high)) for low, high in itertools.pairwise(edges)
        ]
    nearest = np.minimum(home, away)
    side = np.tanh(_signed_gap(away, home) / config.side_scale_m) * seen
    sides = [side * (nearest < radius) for radius in config.side_reach_radii_m]
    columns += sides
    columns.append(np.tanh(_signed_gap(away_second, home_second) / config.second_scale_m) * seen)
    home_count = np.where(masks.home, distance < config.support_radius_m, False).sum(axis=1)
    away_count = np.where(masks.away, distance < config.support_radius_m, False).sum(axis=1)
    support = (home_count - away_count) / config.support_scale_players
    columns.append(np.clip(support, -config.support_clip, config.support_clip) * seen)
    for base in (alone(config.context_radius_m), sides[0]):
        for window in config.context_windows_frames:
            columns.append(window_mean(base, -window, 0))
            columns.append(window_mean(base, 1, window + 1))

    frames, signs = credited_touches(touches, home, away, touch_config)
    columns += [
        touch_recency(frames, signs, n_frames, decay, latest=True)
        for decay in config.last_touch_decay_frames
    ]
    columns.append(touch_recency(frames, signs, n_frames, np.inf, latest=True))
    columns += [
        touch_recency(frames, signs, n_frames, decay, latest=False)
        for decay in config.next_touch_decay_frames
    ]
    now = np.zeros(n_frames)
    now[frames] = signs
    columns.append(now)
    repeated = np.concatenate(([False], signs[1:] == signs[:-1])) if frames.size else frames > 0
    columns.append(
        touch_recency(
            frames[repeated],
            signs[repeated],
            n_frames,
            config.repeated_touch_decay_frames,
            latest=True,
        )
    )
    return np.stack(columns, axis=1)


def learned_log_odds(features: FloatArray, config: LearnedOwnershipConfig) -> FloatArray:
    """Per-frame log-odds that HOME rather than AWAY has the ball (positive = home).

    The model was fitted as if frames were independent; neighbouring frames are not, so its
    output is scaled down before the decoder adds it up over time.
    """
    weights = np.asarray(config.weights, dtype=np.float64)
    return np.asarray(config.llr_scale * (features @ weights), dtype=np.float64)


def fit_weights(
    features: FloatArray, is_home: FloatArray, ridge_penalty: float, iterations: int
) -> FloatArray:
    """Ridge logistic regression without intercept by Newton steps (deterministic, no seed).

    Used only by the training script. Columns are standardised for the fit and the weights
    are returned for the raw features.
    """
    scale = features.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    design = features / scale
    weights = np.zeros(design.shape[1])
    penalty = ridge_penalty * np.eye(design.shape[1])
    for _ in range(iterations):
        probability = 1.0 / (1.0 + np.exp(-(design @ weights)))
        gradient = design.T @ (probability - is_home) + penalty @ weights
        curvature = (design * (probability * (1.0 - probability))[:, None]).T @ design + penalty
        weights = weights - np.linalg.solve(curvature, gradient)
    return np.asarray(weights / scale, dtype=np.float64)
