"""Make the clean validation tracking look like automatic tracking.

In the public sample match the ball position is missing exactly while the ball is dead, which
would hand the answer to any live/dead detector. Automatic tracking is not like that: it loses
the ball before it leaves the pitch, picks it up again some time after the restart, loses it
in open play, and smooths what it has. This module imposes those faults, reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from possession.runs import centred_mean, mask_runs
from possession.structures import DEAD, FloatArray, LabelArray, Tracking


@dataclass(frozen=True)
class DegradationConfig:
    """How the clean ball track is damaged. Frames at 25 fps."""

    seed: int = 0
    # The provided tracker spreads a touch over about a third of a second.
    ball_smoothing_frames: int = 9
    # Share of stoppages before which the ball is lost early, and by how long on average.
    early_loss_probability: float = 0.5
    early_loss_mean_frames: float = 25.0
    # Share of restarts after which the ball is picked up late, and by how long on average.
    late_pickup_probability: float = 0.6
    late_pickup_mean_frames: float = 40.0
    # Longest extension of a gap on either side.
    max_extension_frames: int = 150
    # Additional share of live frames without a ball, in blocks of this median length.
    live_gap_share: float = 0.06
    live_gap_median_frames: float = 100.0
    live_gap_log_sigma: float = 0.8
    # Share of stoppages during which the ball is not tracked at all (a ball out of the
    # picture or in a ball boy's hands). Zero keeps whatever the clean track has.
    stoppage_loss_probability: float = 0.0
    # A slowly varying error of the ball position relative to the players (detection on the
    # wrong pixel, projection of a ball in the air, calibration drift): standard deviation per
    # axis in metres, and how many frames it stays roughly constant. Zero adds none.
    ball_offset_m: float = 0.0
    ball_offset_correlation_frames: int = 50
    # Blocks of frames on which the ball is reported metres away from where it is (another
    # object taken for the ball): share of frames, median block length, typical distance
    # (each block draws between half and one and a half times it). Zero adds none.
    wrong_ball_share: float = 0.0
    wrong_ball_median_frames: float = 50.0
    wrong_ball_distance_m: float = 6.0


def smooth_ball(ball_xy: FloatArray, window: int) -> FloatArray:
    """Smooth the ball track with a moving average that never fills a missing frame."""
    smoothed = np.stack([centred_mean(ball_xy[:, axis], window) for axis in range(2)], axis=1)
    smoothed[np.isnan(ball_xy[:, 0])] = np.nan
    return smoothed


def extended_gap_mask(
    labels: LabelArray, rng: np.random.Generator, config: DegradationConfig
) -> np.ndarray:
    """Frames around stoppages on which the ball is additionally missing."""
    missing = np.zeros(labels.size, dtype=bool)
    starts, stops = mask_runs(labels == DEAD)
    for start, stop in zip(starts, stops, strict=True):
        early = rng.random() < config.early_loss_probability
        late = rng.random() < config.late_pickup_probability
        before = min(rng.exponential(config.early_loss_mean_frames), config.max_extension_frames)
        after = min(rng.exponential(config.late_pickup_mean_frames), config.max_extension_frames)
        if early:
            missing[max(0, start - round(before)) : start] = True
        if late:
            missing[stop : stop + round(after)] = True
    return missing


def live_gap_mask(
    labels: LabelArray, rng: np.random.Generator, config: DegradationConfig
) -> np.ndarray:
    """Blocks of live play on which the ball is lost, up to the configured share."""
    live = np.flatnonzero((labels != DEAD) & (labels < DEAD + 1))
    missing = np.zeros(labels.size, dtype=bool)
    target = config.live_gap_share * live.size
    while live.size and missing.sum() < target:
        start = int(live[rng.integers(live.size)])
        length = round(
            float(rng.lognormal(np.log(config.live_gap_median_frames), config.live_gap_log_sigma))
        )
        missing[start : start + max(1, length)] = True
    return missing


def stoppage_loss_mask(labels: LabelArray, config: DegradationConfig) -> np.ndarray:
    """Whole stoppages on which the ball is missing, each with the configured probability.

    Drawn from its own random stream so that switching it on leaves the other faults as they
    were.
    """
    missing = np.zeros(labels.size, dtype=bool)
    if config.stoppage_loss_probability <= 0.0:
        return missing
    rng = np.random.default_rng([config.seed, 1])
    starts, stops = mask_runs(labels == DEAD)
    for start, stop in zip(starts, stops, strict=True):
        if rng.random() < config.stoppage_loss_probability:
            missing[start:stop] = True
    return missing


def ball_offset(n_frames: int, config: DegradationConfig) -> FloatArray:
    """Smooth random offset ``(n_frames, 2)`` with the configured spread and correlation time.

    Drawn from its own random stream so that switching it on leaves the other faults as they
    were.
    """
    if config.ball_offset_m <= 0.0:
        return np.zeros((n_frames, 2))
    rng = np.random.default_rng([config.seed, 2])
    noise = rng.standard_normal((n_frames, 2))
    window = max(1, config.ball_offset_correlation_frames)
    smooth = np.stack([centred_mean(noise[:, axis], window) for axis in range(2)], axis=1)
    spread = smooth.std(axis=0)
    return np.asarray(smooth / np.where(spread > 0, spread, 1.0) * config.ball_offset_m)


def wrong_ball_offset(n_frames: int, config: DegradationConfig) -> FloatArray:
    """Offset ``(n_frames, 2)`` that is zero except on blocks where the ball is misplaced."""
    offset = np.zeros((n_frames, 2))
    if config.wrong_ball_share <= 0.0:
        return offset
    rng = np.random.default_rng([config.seed, 3])
    covered = np.zeros(n_frames, dtype=bool)
    while covered.sum() < config.wrong_ball_share * n_frames:
        start = int(rng.integers(n_frames))
        length = max(1, round(float(rng.lognormal(np.log(config.wrong_ball_median_frames), 0.8))))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        distance = config.wrong_ball_distance_m * rng.uniform(0.5, 1.5)
        offset[start : start + length] = distance * np.array([np.cos(angle), np.sin(angle)])
        covered[start : start + length] = True
    return offset


def degrade_tracking(tracking: Tracking, labels: LabelArray, config: DegradationConfig) -> Tracking:
    """Return a copy of the tracking with a smoothed, gappier ball track.

    The reference labels are used only to know where stoppages are, so that the simulated
    tracker can be late or early around them; they are not otherwise exposed.
    """
    rng = np.random.default_rng(config.seed)
    offset = (
        tracking.ball_xy
        + ball_offset(tracking.n_frames, config)
        + wrong_ball_offset(tracking.n_frames, config)
    )
    ball = smooth_ball(offset, config.ball_smoothing_frames)
    missing = extended_gap_mask(labels, rng, config) | live_gap_mask(labels, rng, config)
    missing |= stoppage_loss_mask(labels, config)
    ball[missing] = np.nan
    return replace(tracking, ball_xy=ball)


def provided_tracker_profile(seed: int, *, wrong_ball_share: float = 0.0) -> DegradationConfig:
    """Damage fitted so a clean validation track resembles the provided automatic tracking.

    Fitted without labels, by matching statistics of the match to be labelled on the frames
    the pipeline calls live: frequency and strength of ball-velocity changes, touches per
    minute, distance from a touch or a slow ball to the nearest player (TECHNICAL.md,
    section 4.2). Smoothing over 21 frames and a 0.7 m offset varying over 6 s reproduce them
    except for the far tail of the last; ``wrong_ball_share`` adds misplaced-ball blocks for
    that tail.
    """
    return DegradationConfig(
        seed=seed,
        ball_smoothing_frames=21,
        ball_offset_m=0.7,
        ball_offset_correlation_frames=150,
        wrong_ball_share=wrong_ball_share,
    )
