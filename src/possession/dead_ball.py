"""Dead-ball detector: an unsupervised two-state (live / dead) hidden Markov model.

No frame of this match is labelled, so the detector learns what "dead" looks like from the
structure of the data itself. Two cues are observed on every frame:

* the **ball observation class** - tracked and moving, tracked and still, tracked off the
  pitch, or not tracked at all (split by how long the disappearance lasts);
* the **collective activity** - the mean speed of all tracked players.

Expectation-maximisation fits, for each hidden state, a table over ball classes and a normal
distribution over activity, plus the transition probabilities. The only prior knowledge
injected is the starting point: frames where the players are walking are provisionally called
dead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from possession.config import BallConfig, LivenessConfig
from possession.decoding import forward_backward
from possession.preprocessing.ball import BallTrack
from possession.runs import run_length_per_frame
from possession.structures import BoolArray, FloatArray, IntArray

LOGGER = logging.getLogger(__name__)

LIVE, DEAD_BALL = 0, 1

# Ball observation classes.
MISSING_SHORT, MISSING_MEDIUM, MISSING_LONG, OUT_OF_BOUNDS, STATIONARY, SLOW, FAST = range(7)
N_BALL_CLASSES = 7
BALL_CLASS_NAMES: tuple[str, ...] = (
    "missing_short",
    "missing_medium",
    "missing_long",
    "out_of_bounds",
    "stationary",
    "slow",
    "fast",
)


def ball_observation_class(ball: BallTrack, config: BallConfig) -> IntArray:
    """Classify every frame by what the tracker says about the ball.

    The length of a disappearance is known offline and is informative: half a second without
    the ball is an occlusion, while a ball that stays lost for ten seconds has usually left the
    pitch. Positions bridged by interpolation count as tracked, because a bridged gap is by
    construction a short one inside continuous play.
    """
    known = ~np.isnan(ball.xy[:, 0])
    gap_length = run_length_per_frame(~known)
    classes = np.full(known.size, SLOW, dtype=np.int64)
    classes[known & (ball.speed >= config.fast_speed_mps)] = FAST
    classes[known & (ball.speed < config.stationary_speed_mps)] = STATIONARY
    classes[known & (ball.edge_distance < -config.out_of_bounds_margin_m)] = OUT_OF_BOUNDS
    classes[~known] = MISSING_LONG
    classes[~known & (gap_length <= config.medium_gap_max_frames)] = MISSING_MEDIUM
    classes[~known & (gap_length <= config.short_gap_max_frames)] = MISSING_SHORT
    return classes


@dataclass(frozen=True)
class LivenessModel:
    """Fitted parameters of the live/dead HMM. Rows are (LIVE, DEAD_BALL)."""

    ball_table: FloatArray  # (2, N_BALL_CLASSES) P(ball class | state)
    activity_mean: FloatArray  # (2,) m/s
    activity_std: FloatArray  # (2,) m/s
    transitions: FloatArray  # (2, 2) row-stochastic
    log_likelihood_trace: tuple[float, ...]


def _ball_table(responsibility: FloatArray, ball_class: IntArray, floor: float) -> FloatArray:
    """Responsibility-weighted class frequencies per state, floored so no class is impossible."""
    table = np.stack(
        [
            np.bincount(ball_class, weights=responsibility[:, state], minlength=N_BALL_CLASSES)
            for state in range(responsibility.shape[1])
        ]
    )
    table = np.maximum(table / table.sum(axis=1, keepdims=True), floor)
    return np.asarray(table / table.sum(axis=1, keepdims=True), dtype=np.float64)


def _activity_moments(
    responsibility: FloatArray, activity: FloatArray, min_std: float
) -> tuple[FloatArray, FloatArray]:
    """Responsibility-weighted mean and standard deviation of activity per state."""
    known = np.isfinite(activity)
    weights = responsibility[known]
    values = activity[known][:, None]
    total = weights.sum(axis=0)
    mean = (weights * values).sum(axis=0) / total
    variance = (weights * (values - mean) ** 2).sum(axis=0) / total
    return mean, np.maximum(np.sqrt(variance), min_std)


def _m_step(
    responsibility: FloatArray,
    ball_class: IntArray,
    activity: FloatArray,
    *,
    transitions: FloatArray,
    trace: tuple[float, ...],
    config: LivenessConfig,
) -> LivenessModel:
    """Re-estimate the emission parameters from the state responsibilities."""
    mean, std = _activity_moments(responsibility, activity, config.min_activity_std_mps)
    return LivenessModel(
        _ball_table(responsibility, ball_class, config.probability_floor),
        mean,
        std,
        transitions,
        trace,
    )


def liveness_log_emissions(
    model: LivenessModel, ball_class: IntArray, activity: FloatArray, temperature: float
) -> FloatArray:
    """Per-frame log-likelihood of each state, shape ``(n_frames, 2)``.

    The two cues are treated as conditionally independent given the state; a frame with too
    few tracked players to measure activity is scored on the ball class alone. Consecutive
    frames are far from independent, so the sum is tempered: ``temperature`` below one says
    that a second of evidence is worth fewer than twenty-five independent observations.
    """
    log_ball = np.log(model.ball_table)[:, ball_class].T
    standardised = (activity[:, None] - model.activity_mean) / model.activity_std
    log_activity = -0.5 * standardised**2 - np.log(model.activity_std)
    log_activity = np.where(np.isfinite(activity)[:, None], log_activity, 0.0)
    return np.asarray(temperature * (log_ball + log_activity), dtype=np.float64)


def initial_transitions(config: LivenessConfig) -> FloatArray:
    """Transition matrix implied by the expected spell lengths (geometric durations)."""
    leave_live = 1.0 / config.expected_live_spell_frames
    leave_dead = 1.0 / config.expected_dead_spell_frames
    return np.array([[1.0 - leave_live, leave_live], [leave_dead, 1.0 - leave_dead]])


def fit_liveness(
    ball_class: IntArray, activity: FloatArray, fit_mask: BoolArray, config: LivenessConfig
) -> LivenessModel:
    """Fit the live/dead HMM by EM on the frames of ``fit_mask``.

    The frames of the mask are treated as one sequence; the handful of artificial joins (one at
    half-time) is negligible against tens of thousands of real frames. The starting point calls
    "dead" every frame on which the players average less than a brisk walk; EM is then free to
    move both distributions, and does.
    """
    if not fit_mask.any():
        raise ValueError("no frames to fit the liveness model on")
    ball_fit, activity_fit = ball_class[fit_mask], activity[fit_mask]
    walking = np.nan_to_num(activity_fit, nan=np.inf) < config.seed_activity_split_mps
    responsibility = np.stack((~walking, walking), axis=1).astype(np.float64)
    initial = np.array([1.0 - config.initial_dead_probability, config.initial_dead_probability])
    model = _m_step(
        responsibility,
        ball_fit,
        activity_fit,
        transitions=initial_transitions(config),
        trace=(),
        config=config,
    )
    for iteration in range(config.em_iterations):
        posterior = forward_backward(
            liveness_log_emissions(model, ball_fit, activity_fit, config.emission_temperature),
            model.transitions,
            initial,
        )
        counts = posterior.transition_counts
        model = _m_step(
            posterior.state,
            ball_fit,
            activity_fit,
            transitions=counts / counts.sum(axis=1, keepdims=True),
            trace=(*model.log_likelihood_trace, posterior.log_likelihood),
            config=config,
        )
        LOGGER.debug(
            "liveness EM iteration %d: log-likelihood %.1f", iteration, posterior.log_likelihood
        )
    return model


def states_are_identified(model: LivenessModel) -> bool:
    """Check that the DEAD_BALL state is the low-activity one.

    EM is free to swap its two states, so the football meaning of each is checked rather than
    assumed.
    """
    return bool(model.activity_mean[DEAD_BALL] < model.activity_mean[LIVE])
