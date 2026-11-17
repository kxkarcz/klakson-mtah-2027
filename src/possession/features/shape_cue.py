"""Ball-free ownership cue: a small logistic model on team shape and movement.

On 41% of graded frames there is no ball to measure distances to. What remains visible is how
the two teams are arranged: the side with the ball spreads out and advances, the side without
it contracts and retreats. The mapping from those quantities to "who has it" is learnt from
pseudo-labels - frames of this match where the ball *is* tracked and a first decoding pass is
confident about the owner - and then applied where the ball is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.config import CollectiveConfig
from possession.features.collective import TeamShape
from possession.structures import HOME, BoolArray, FloatArray, LabelArray


def home_attack_sign(home: TeamShape, away: TeamShape, second_half_start: int) -> FloatArray:
    """+1 where HOME attacks towards increasing x, -1 otherwise, per frame.

    A team's average position over a half lies on its own side of the pitch, so the sign of
    the mean centroid offset gives the direction of play without any annotation. The halves
    are estimated separately because the teams change ends.
    """
    offset = away.centroid[:, 0] - home.centroid[:, 0]
    sign = np.ones(offset.size)
    for start, stop in ((0, second_half_start), (second_half_start, offset.size)):
        if stop > start and np.isfinite(offset[start:stop]).any():
            sign[start:stop] = 1.0 if np.nanmean(offset[start:stop]) >= 0 else -1.0
    return sign


def shape_features(
    home: TeamShape, away: TeamShape, attack_sign: FloatArray, lag_frames: int, fps: float
) -> FloatArray:
    """Feature matrix ``(n_frames, 6)``, oriented so that positive means "good for HOME".

    Columns: joint-centroid advance speed along HOME's attack, joint-centroid position along
    it, centroid separation, length and width differences, and the lateral speed difference.
    Rows with any missing input are NaN.
    """
    joint = 0.5 * (home.centroid + away.centroid)
    velocity = np.full_like(joint, np.nan)
    velocity[lag_frames:] = (joint[lag_frames:] - joint[:-lag_frames]) * (fps / lag_frames)
    home_velocity = np.full_like(joint, np.nan)
    home_velocity[lag_frames:] = (home.centroid[lag_frames:] - home.centroid[:-lag_frames]) * (
        fps / lag_frames
    )
    away_velocity = np.full_like(joint, np.nan)
    away_velocity[lag_frames:] = (away.centroid[lag_frames:] - away.centroid[:-lag_frames]) * (
        fps / lag_frames
    )
    return np.stack(
        (
            attack_sign * velocity[:, 0],
            attack_sign * (joint[:, 0] - np.nanmean(joint[:, 0])),
            attack_sign * (away.centroid[:, 0] - home.centroid[:, 0]),
            home.spread[:, 0] - away.spread[:, 0],
            home.spread[:, 1] - away.spread[:, 1],
            attack_sign * (home_velocity[:, 0] - away_velocity[:, 0]),
        ),
        axis=1,
    )


@dataclass(frozen=True)
class ShapeModel:
    """Standardisation and weights of the fitted logistic model."""

    mean: FloatArray
    scale: FloatArray
    weights: FloatArray  # (n_features + 1,), bias last
    n_training_frames: int
    training_accuracy: float


def fit_shape_model(
    features: FloatArray, pseudo_labels: LabelArray, usable: BoolArray, config: CollectiveConfig
) -> ShapeModel | None:
    """Ridge-regularised logistic regression by Newton iterations (deterministic, no seed).

    Returns None when there is nothing to learn from, in which case the cue is switched off
    rather than guessed.
    """
    rows = usable & np.isfinite(features).all(axis=1)
    if int(rows.sum()) <= features.shape[1] + 1:
        return None
    x_raw = features[rows]
    target = (pseudo_labels[rows] == HOME).astype(np.float64)
    if target.min() == target.max():
        return None
    mean, scale = x_raw.mean(axis=0), x_raw.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    design = np.column_stack(((x_raw - mean) / scale, np.ones(x_raw.shape[0])))
    penalty = config.ridge_penalty * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.zeros(design.shape[1])
    for _ in range(config.fit_iterations):
        probability = 1.0 / (1.0 + np.exp(-(design @ weights)))
        gradient = design.T @ (probability - target) + penalty @ weights
        curvature = (design * (probability * (1.0 - probability))[:, None]).T @ design + penalty
        weights = weights - np.linalg.solve(curvature, gradient)
    accuracy = float(np.mean(((design @ weights) > 0) == (target > 0.5)))
    return ShapeModel(mean, scale, weights, int(rows.sum()), accuracy)


def shape_log_odds(
    model: ShapeModel | None, features: FloatArray, ball_known: BoolArray, config: CollectiveConfig
) -> FloatArray:
    """Home-vs-away log-odds from team shape, non-zero only where the ball is not tracked.

    Where the ball is tracked its geometry is the better witness and the shape cue stays
    silent; tempering and the cap keep a long ball-less stretch from overturning a possession
    on shape alone.
    """
    out = np.zeros(features.shape[0])
    if model is None:
        return out
    rows = ~ball_known & np.isfinite(features).all(axis=1)
    design = np.column_stack(
        ((features[rows] - model.mean) / model.scale, np.ones(int(rows.sum())))
    )
    out[rows] = np.clip(
        config.temperature * (design @ model.weights),
        -config.max_llr_per_frame,
        config.max_llr_per_frame,
    )
    return out
