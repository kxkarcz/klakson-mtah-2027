from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import yaml

from possession.config import ConfigError, config_from_mapping, ownership_feature_names
from possession.io.config_file import load_config
from possession.learned_ownership import (
    credited_touches,
    fit_weights,
    learned_log_odds,
    ownership_features,
    touch_recency,
    window_mean,
)
from possession.pipeline import run_pipeline
from possession.preprocessing.ball import clean_ball
from possession.preprocessing.teams import assign_slots_to_teams
from possession.structures import Tracking
from test_submission_and_pipeline import synthetic_match

from synthetic import grid_team, make_tracking
from tests_paths import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
LEARNED = CONFIG.learned_ownership
NAMES = ownership_feature_names(LEARNED)


def test_window_mean_looks_back_and_ahead_with_zero_padding() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(window_mean(values, -2, 0), [0.0, 0.5, 1.5, 2.5])
    np.testing.assert_allclose(window_mean(values, 1, 3), [2.5, 3.5, 2.0, 0.0])


def test_touch_recency_fades_from_the_latest_touch_and_towards_the_next() -> None:
    frames, signs = np.array([10, 30]), np.array([1.0, -1.0])
    latest = touch_recency(frames, signs, 40, 10.0, latest=True)
    assert latest[9] == 0.0 and latest[10] == 1.0
    np.testing.assert_allclose(latest[20], np.exp(-1.0))
    assert latest[30] == -1.0
    following = touch_recency(frames, signs, 40, 10.0, latest=False)
    np.testing.assert_allclose(following[0], np.exp(-1.0))
    assert following[11] < 0.0 and following[31] == 0.0
    carried = touch_recency(frames, signs, 40, np.inf, latest=True)
    assert set(carried[10:30]) == {1.0} and set(carried[30:]) == {-1.0}
    assert not touch_recency(frames[:0], signs[:0], 40, 10.0, latest=True).any()


def test_touches_are_credited_only_to_a_team_clearly_at_the_ball() -> None:
    home = np.array([0.5, 0.5, 9.0, 3.0])
    away = np.array([9.0, 0.6, 0.5, 9.0])
    strict = dataclasses.replace(CONFIG.touch_evidence, duel_margin_m=0.5)
    frames, signs = credited_touches(np.arange(4), home, away, strict)
    assert frames.tolist() == [0, 2] and signs.tolist() == [1.0, -1.0]


def match(swap: bool = False) -> Tracking:
    """Home dribbles for 4 s, passes; away collects and dribbles back."""
    n = 300
    home_xy, away_xy = grid_team(30.0), grid_team(70.0)
    ball = np.empty((n, 2))
    ball[:100] = np.column_stack((np.linspace(30.0, 34.0, 100), np.full(100, 30.0)))
    ball[100:150] = np.column_stack((np.linspace(34.0, 68.0, 50), np.full(50, 30.0)))
    ball[150:] = np.column_stack((np.linspace(68.0, 62.0, 150), np.full(150, 30.0)))
    home = np.broadcast_to(home_xy, (n, *home_xy.shape)).copy()
    away = np.broadcast_to(away_xy, (n, *away_xy.shape)).copy()
    beside = np.array([0.0, 0.5])
    home[:100, 0] = ball[:100] + beside
    away[150:, 0] = ball[150:] + beside
    return make_tracking(n, away, home, ball) if swap else make_tracking(n, home, away, ball)


def features_of(tracking: Tracking) -> np.ndarray:
    ball = clean_ball(tracking, CONFIG.ball)
    touches = np.array([99, 150])
    return ownership_features(
        tracking.player_xy,
        assign_slots_to_teams(tracking),
        ball,
        touches,
        config=LEARNED,
        touch_config=CONFIG.touch_evidence,
    )


def test_features_have_one_column_per_name_and_flip_when_the_teams_are_swapped() -> None:
    features = features_of(match())
    assert features.shape == (300, len(NAMES)) and len(set(NAMES)) == len(NAMES)
    np.testing.assert_allclose(features_of(match(swap=True)), -features)
    alone = features[:, NAMES.index("alone_within_1.0")]
    assert set(alone[:100]) == {1.0} and set(alone[160:]) == {-1.0} and alone[125] == 0.0
    last = features[:, NAMES.index("last_touch")]
    assert set(last[99:150]) == {1.0} and set(last[150:]) == {-1.0}


def test_geometry_is_silent_without_a_ball_but_touch_memory_carries_on() -> None:
    tracking = match()
    ball_xy = tracking.ball_xy.copy()
    ball_xy[200:260] = np.nan
    features = features_of(dataclasses.replace(tracking, ball_xy=ball_xy))
    assert not features[220, NAMES.index("alone_within_1.5")]
    assert not features[220, NAMES.index("closer_side_reach_4.0")]
    assert features[220, NAMES.index("last_touch")] == -1.0


def test_fit_recovers_which_feature_carries_the_label() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((4000, 3)) * [1.0, 5.0, 0.2]
    is_home = (features[:, 0] + 0.3 * rng.standard_normal(4000) > 0).astype(np.float64)
    weights = fit_weights(features, is_home, 1.0, 30)
    assert weights[0] > 1.0 and abs(weights[1]) < 0.1
    assert np.mean(((features @ weights) > 0) == (is_home > 0.5)) > 0.85
    np.testing.assert_array_equal(weights, fit_weights(features, is_home, 1.0, 30))


def test_log_odds_are_the_scaled_weighted_sum() -> None:
    weights = np.zeros(len(NAMES))
    weights[NAMES.index("last_touch")] = 2.0
    config = dataclasses.replace(LEARNED, weights=tuple(weights), llr_scale=0.5)
    log_odds = learned_log_odds(features_of(match()), config)
    assert set(log_odds[99:150]) == {1.0} and set(log_odds[150:]) == {-1.0}


def test_config_demands_one_weight_per_feature() -> None:
    mapping = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    mapping["learned_ownership"]["weights"] = [0.0] * (len(NAMES) - 1)
    with pytest.raises(ConfigError, match="one weight per feature"):
        config_from_mapping(mapping)


def test_pipeline_uses_the_model_when_enabled_and_stays_deterministic() -> None:
    home, away, ball, speed, truth = synthetic_match()
    tracking = make_tracking(truth.size, home, away, ball, speed)
    weights = np.zeros(len(NAMES))
    weights[NAMES.index("alone_within_1.5")] = 4.0
    config = dataclasses.replace(
        CONFIG,
        learned_ownership=dataclasses.replace(LEARNED, enabled=True, weights=tuple(weights)),
        # Synthetic players change speed instantly: use present activity, not the look-ahead.
        activity=dataclasses.replace(CONFIG.activity, lookahead_enabled=False),
    )
    first, second = run_pipeline(tracking, [], config), run_pipeline(tracking, [], config)
    assert np.array_equal(first.labels, second.labels)
    assert np.mean(first.decoded == truth) > 0.95
    silent = dataclasses.replace(
        config, learned_ownership=dataclasses.replace(config.learned_ownership, llr_scale=1e-9)
    )
    assert np.mean(run_pipeline(tracking, [], silent).decoded == truth) < 0.95
