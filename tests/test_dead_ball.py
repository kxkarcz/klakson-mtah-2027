from __future__ import annotations

import dataclasses

import numpy as np

from possession.dead_ball import (
    DEAD_BALL,
    FAST,
    LIVE,
    MISSING_LONG,
    MISSING_MEDIUM,
    MISSING_SHORT,
    OUT_OF_BOUNDS,
    SLOW,
    STATIONARY,
    ball_observation_class,
    fit_liveness,
    liveness_log_emissions,
    states_are_identified,
)
from possession.features.collective import anticipate_restart, collective_activity
from possession.features.kickoff import kickoff_formation
from possession.io.config_file import load_config
from possession.preprocessing.ball import (
    clean_ball,
    interpolate_short_gaps,
    reject_teleports,
)
from possession.preprocessing.teams import assign_slots_to_teams

from synthetic import FPS, grid_team, make_tracking
from tests_paths import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
# The synthetic players change speed instantly, so these tests use present activity only.
INSTANT = dataclasses.replace(CONFIG.activity, lookahead_enabled=False)


def moving_ball(n_frames: int, speed_mps: float, start: tuple[float, float]) -> np.ndarray:
    """A ball bouncing left-right inside the pitch at constant speed."""
    travelled = np.arange(n_frames) * speed_mps / FPS
    x = start[0] + 20.0 * np.abs(((travelled / 20.0) % 2.0) - 1.0)
    return np.column_stack((x, np.full(n_frames, start[1])))


def test_ball_classes_cover_each_scenario() -> None:
    n = 1200
    ball = moving_ball(n, 10.0, (30.0, 30.0))
    ball[100:130] = np.nan  # 30 frames: short gap
    ball[200:300] = np.nan  # 100 frames: medium gap
    ball[400:700] = np.nan  # 300 frames: long gap
    ball[800:850] = [50.0, -3.0]  # held beyond the touchline
    ball[900:950] = [50.0, 30.0]  # placed for a free kick
    ball[1000:1100] = moving_ball(100, 3.0, (30.0, 30.0))  # dribble
    tracking = make_tracking(n, grid_team(30.0), grid_team(70.0), ball)
    classes = ball_observation_class(clean_ball(tracking, CONFIG.ball), CONFIG.ball)
    assert set(classes[100:130]) == {MISSING_SHORT}
    assert set(classes[200:300]) == {MISSING_MEDIUM}
    assert set(classes[400:700]) == {MISSING_LONG}
    assert set(classes[805:845]) == {OUT_OF_BOUNDS}
    assert set(classes[905:945]) == {STATIONARY}
    assert set(classes[1010:1090]) == {SLOW}
    assert set(classes[10:90]) == {FAST}


def test_short_gaps_are_bridged_and_long_ones_are_not() -> None:
    ball = np.column_stack((np.arange(40.0), np.zeros(40)))
    ball[5:9] = np.nan
    ball[20:35] = np.nan
    filled = interpolate_short_gaps(ball, max_gap_frames=12)
    np.testing.assert_allclose(filled[5:9, 0], [5.0, 6.0, 7.0, 8.0])
    assert np.isnan(filled[20:35, 0]).all()


def test_isolated_teleport_is_removed_but_a_shot_is_kept() -> None:
    ball = np.column_stack((np.arange(10.0) * 0.4, np.zeros(10)))  # 10 m/s
    ball[4] = [80.0, 50.0]  # a second ball for one frame
    cleaned = reject_teleports(ball, FPS, max_speed_mps=45.0)
    assert np.isnan(cleaned[4]).all()
    assert not np.isnan(np.delete(cleaned, 4, axis=0)).any()


def synthetic_match(n_cycles: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating 20 s of play (fast players, tracked ball) and 12 s of stoppage."""
    live, dead = 500, 300
    rng = np.random.default_rng(0)
    ball_parts, speed_parts, truth_parts = [], [], []
    for _ in range(n_cycles):
        ball_parts += [moving_ball(live, 9.0, (30.0, 30.0)), np.full((dead, 2), np.nan)]
        speed_parts += [
            2.4 + 0.3 * rng.standard_normal(live),
            1.0 + 0.2 * rng.standard_normal(dead),
        ]
        truth_parts += [np.full(live, LIVE), np.full(dead, DEAD_BALL)]
    return np.concatenate(ball_parts), np.concatenate(speed_parts), np.concatenate(truth_parts)


def test_em_separates_play_from_stoppages_without_labels() -> None:
    ball_xy, speed, truth = synthetic_match()
    tracking = make_tracking(truth.size, grid_team(30.0), grid_team(70.0), ball_xy, speed)
    masks = assign_slots_to_teams(tracking)
    activity = collective_activity(tracking, masks, INSTANT)
    classes = ball_observation_class(clean_ball(tracking, CONFIG.ball), CONFIG.ball)
    model = fit_liveness(classes, activity, np.ones(truth.size, dtype=bool), CONFIG.liveness)
    assert states_are_identified(model)
    emissions = liveness_log_emissions(model, classes, activity, 1.0)
    predicted = emissions.argmax(axis=1)
    assert np.mean(predicted == truth) > 0.97
    assert model.ball_table[DEAD_BALL, MISSING_LONG] > model.ball_table[LIVE, MISSING_LONG]


def test_activity_ignores_players_without_a_speed() -> None:
    tracking = make_tracking(60, grid_team(30.0), grid_team(70.0), np.zeros((60, 2)), 2.0)
    tracking.player_speed[:, 0] = np.nan
    activity = collective_activity(tracking, assign_slots_to_teams(tracking), INSTANT)
    np.testing.assert_allclose(activity, 2.0)


def test_kickoff_line_up_is_detected_only_when_sustained() -> None:
    n = 200
    home = np.broadcast_to(grid_team(30.0), (n, 11, 2)).copy()
    away = np.broadcast_to(grid_team(70.0), (n, 11, 2)).copy()
    home[100:, 0, 0] = 65.0  # one home player crosses the halfway line: open play
    home[40:50, 0, 0] = 65.0  # a ten-frame interruption splits the line-up in two runs
    tracking = make_tracking(n, home, away, np.full((n, 2), np.nan))
    lined_up = kickoff_formation(tracking, assign_slots_to_teams(tracking), CONFIG.kickoff)
    assert lined_up[:40].all() and lined_up[50:100].all()
    assert not lined_up[40:50].any() and not lined_up[100:].any()


def test_lookahead_removes_the_restart_lag_and_leaves_the_stop_alone() -> None:
    # Play, a stop at 100 (players slow at once), a restart at 300 after which the block
    # needs 80 frames to get back up to speed.
    activity = np.concatenate(
        (np.full(100, 2.4), np.full(200, 1.0), np.linspace(1.0, 2.4, 80), np.full(120, 2.4))
    )
    ahead = anticipate_restart(activity, 75)
    midpoint = 1.7
    assert np.flatnonzero(ahead < midpoint)[0] == 100  # the stop is where it was
    last_slow = int(np.flatnonzero(ahead < midpoint)[-1])
    assert last_slow < 300 <= int(np.flatnonzero(activity < midpoint)[-1])
    assert np.isnan(anticipate_restart(np.array([np.nan, np.nan, 1.0]), 1)[0])
    np.testing.assert_allclose(
        anticipate_restart(np.array([1.0, np.nan, 3.0]), 2), [3.0, np.nan, 3.0]
    )
