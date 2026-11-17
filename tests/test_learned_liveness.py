from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import yaml

from possession.config import ConfigError, config_from_mapping, liveness_feature_count
from possession.dead_ball import N_BALL_CLASSES, ball_observation_class
from possession.features.ball_events import detect_ball_events
from possession.features.collective import collective_activity, team_shape
from possession.io.config_file import load_config
from possession.learned_liveness import (
    event_recency,
    frames_since,
    learned_liveness_emissions,
    liveness_features,
)
from possession.pipeline import run_pipeline
from possession.preprocessing.ball import clean_ball
from possession.preprocessing.teams import assign_slots_to_teams
from possession.structures import DEAD
from test_submission_and_pipeline import synthetic_match

from synthetic import make_tracking
from tests_paths import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
LIVE = CONFIG.learned_liveness


def test_frames_since_counts_from_the_last_true_frame() -> None:
    mask = np.array([False, True, False, False, True, False])
    assert frames_since(mask).tolist()[1:] == [0, 1, 2, 0, 1]
    assert frames_since(mask)[0] > 10**6  # before the first event: effectively never


def test_event_recency_fades_behind_and_ahead() -> None:
    frames = np.array([10])
    behind = event_recency(frames, 30, 10.0, latest=True)
    ahead = event_recency(frames, 30, 10.0, latest=False)
    assert behind[9] == 0.0 and behind[10] == 1.0
    np.testing.assert_allclose(behind[20], np.exp(-1.0))
    assert ahead[11] == 0.0
    np.testing.assert_allclose(ahead[0], np.exp(-1.0))


def match_features() -> tuple[np.ndarray, np.ndarray]:
    home, away, ball, speed, truth = synthetic_match()
    tracking = make_tracking(truth.size, home, away, ball, speed)
    masks = assign_slots_to_teams(tracking)
    track = clean_ball(tracking, CONFIG.ball)
    features = liveness_features(
        tracking,
        ball=track,
        ball_class=ball_observation_class(track, CONFIG.ball),
        activity_now=collective_activity(
            tracking, masks, dataclasses.replace(CONFIG.activity, lookahead_enabled=False)
        ),
        shapes=(
            team_shape(tracking.player_xy, masks.home, CONFIG.collective.min_players_per_team),
            team_shape(tracking.player_xy, masks.away, CONFIG.collective.min_players_per_team),
        ),
        events=detect_ball_events(track, tracking.fps, CONFIG.events, CONFIG.ball),
        config=LIVE,
    )
    return features, truth


def test_features_are_finite_with_an_intercept_and_the_declared_width() -> None:
    features, truth = match_features()
    assert features.shape == (truth.size, liveness_feature_count(LIVE))
    assert np.isfinite(features).all()
    assert set(features[:, 0]) == {1.0}
    one_hot = features[:, 1 : 1 + N_BALL_CLASSES]
    assert set(one_hot.sum(axis=1)) == {1.0}  # exactly one ball class per frame


def test_emissions_are_scaled_log_probabilities_of_the_two_states() -> None:
    weights = np.zeros(liveness_feature_count(LIVE))
    weights[0] = np.log(3.0)  # intercept only: live three times as likely as dead
    config = dataclasses.replace(LIVE, weights=tuple(weights), llr_scale=0.5)
    intercept_only = np.zeros((4, weights.size))
    intercept_only[:, 0] = 1.0
    emissions = learned_liveness_emissions(intercept_only, config)
    np.testing.assert_allclose(emissions[:, 0], 0.5 * np.log(0.75))
    np.testing.assert_allclose(emissions[:, 1], 0.5 * np.log(0.25))


def test_config_demands_one_weight_per_feature() -> None:
    mapping = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    mapping["learned_liveness"]["weights"] = [0.0] * (liveness_feature_count(LIVE) + 1)
    with pytest.raises(ConfigError, match="one weight per feature"):
        config_from_mapping(mapping)


def test_pipeline_uses_the_model_when_enabled_and_stays_deterministic() -> None:
    home, away, ball, speed, truth = synthetic_match()
    tracking = make_tracking(truth.size, home, away, ball, speed)
    features, _ = match_features()
    # A model that has learned "the ball is dead while it is not seen".
    weights = np.zeros(liveness_feature_count(LIVE))
    visibility = 1 + N_BALL_CLASSES + 2 + len(LIVE.activity_windows_frames)
    weights[0] = -6.0
    weights[visibility] = 12.0  # ball seen over the surrounding second: live
    config = dataclasses.replace(
        CONFIG,
        learned_liveness=dataclasses.replace(LIVE, enabled=True, weights=tuple(weights)),
        activity=dataclasses.replace(CONFIG.activity, lookahead_enabled=False),
    )
    first, second = run_pipeline(tracking, [], config), run_pipeline(tracking, [], config)
    assert np.array_equal(first.labels, second.labels)
    assert np.mean((first.decoded == DEAD) == (truth == DEAD)) > 0.9
    assert features.shape[0] == truth.size
