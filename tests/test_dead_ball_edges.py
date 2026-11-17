from __future__ import annotations

import dataclasses

import numpy as np

from possession.features.ball_events import BallEvents, restart_candidates
from possession.io.config_file import load_config
from possession.preprocessing.ball import clean_ball
from possession.refinement.dead_ball_edges import (
    nearest_in_window,
    refine_dead_ball_edges,
    strongest_deceleration,
)
from possession.structures import AWAY, DEAD, HOME

from synthetic import FPS, grid_team, make_tracking
from tests_paths import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
EDGES = dataclasses.replace(
    CONFIG.dead_edges, enabled=True, end_window_before_frames=75, end_window_after_frames=75
)
NONE = np.array([], dtype=np.int64)

# H 0..300 | D 300..600 | A 600..900
LABELS = np.array([HOME] * 300 + [DEAD] * 300 + [AWAY] * 300, dtype=np.int8)
ACTIVITY = np.full(900, 2.0)
FREE = np.zeros(900, dtype=bool)


def events(out: list[int]) -> BallEvents:
    return BallEvents(np.array(out, dtype=np.int64), NONE, NONE, NONE)


def refine(out: list[int], restarts: list[int], config=EDGES, forced=FREE, activity=ACTIVITY):
    return refine_dead_ball_edges(
        LABELS,
        events(out),
        np.array(restarts, dtype=np.int64),
        activity=activity,
        forced_dead=forced,
        config=config,
    )


def test_nearest_event_in_window() -> None:
    candidates = np.array([10, 40, 90])
    assert nearest_in_window(candidates, 50, before=20, after=20) == 40
    assert nearest_in_window(candidates, 50, before=5, after=5) is None
    assert nearest_in_window(candidates, 65, before=30, after=30) == 40  # tie: the earlier one
    assert nearest_in_window(NONE, 50, before=20, after=20) is None


def test_start_moves_back_to_the_exit_and_end_to_the_restart() -> None:
    result = refine(out=[260], restarts=[630])
    assert result.labels.tolist() == [HOME] * 260 + [DEAD] * 370 + [AWAY] * 270
    assert result.start_shifts.tolist() == [-40]
    assert result.end_shifts.tolist() == [30]


def test_end_can_move_earlier_and_gives_the_frames_to_the_restarting_team() -> None:
    result = refine(out=[], restarts=[550])
    assert result.labels.tolist() == [HOME] * 300 + [DEAD] * 250 + [AWAY] * 350
    assert result.start_shifts.size == 0


def test_without_an_event_in_the_window_the_decoder_edge_stays() -> None:
    result = refine(out=[100], restarts=[800])
    assert np.array_equal(result.labels, LABELS)
    assert result.start_shifts.size == 0 and result.end_shifts.size == 0


def test_edges_cannot_swallow_a_neighbour_or_the_stoppage() -> None:
    wide = dataclasses.replace(
        EDGES,
        start_window_before_frames=400,
        start_window_after_frames=400,
        end_window_before_frames=400,
        end_window_after_frames=400,
    )
    # An exit 10 frames into the previous possession and a restart 10 frames before its end.
    assert np.array_equal(refine(out=[10], restarts=[890], config=wide).labels, LABELS)
    # A start pushed to within a second of the stoppage end is refused as well.
    assert np.array_equal(refine(out=[590], restarts=[], config=wide).labels, LABELS)


def test_forced_dead_frames_never_become_live() -> None:
    forced = FREE.copy()
    forced[560:600] = True  # kick-off line-up
    assert np.array_equal(refine(out=[], restarts=[550], forced=forced).labels, LABELS)


def test_deceleration_is_used_only_when_enabled_and_no_exit_is_found() -> None:
    activity = np.concatenate((np.full(270, 2.5), np.full(630, 1.0)))
    assert np.array_equal(refine(out=[], restarts=[], activity=activity).labels, LABELS)
    with_whistle = dataclasses.replace(EDGES, use_deceleration=True)
    result = refine(out=[], restarts=[], activity=activity, config=with_whistle)
    assert abs(int(result.start_shifts[0]) + 30) <= EDGES.deceleration_lag_frames
    assert strongest_deceleration(np.full(100, 2.0), 50, 20, 20, 5) is None


def test_first_stoppage_start_and_last_stoppage_end_are_left_alone() -> None:
    labels = np.array([DEAD] * 100 + [HOME] * 100 + [DEAD] * 100, dtype=np.int8)
    result = refine_dead_ball_edges(
        labels,
        events([5, 195]),
        np.array([95, 290]),
        activity=np.full(300, 2.0),
        forced_dead=np.zeros(300, bool),
        config=EDGES,
    )
    assert result.labels.tolist() == [DEAD] * 95 + [HOME] * 100 + [DEAD] * 105


def test_reappearance_counts_as_restart_only_if_the_ball_is_moving() -> None:
    n = 400
    xy = np.full((n, 2), np.nan)
    xy[:50] = [20.0, 30.0]
    xy[150:250] = [50.0, 1.0] + np.arange(100)[:, None] * np.array([0.0, 12.0]) / FPS  # thrown in
    xy[350:] = [70.0, 30.0]  # comes back resting
    ball = clean_ball(make_tracking(n, grid_team(30.0), grid_team(70.0), xy), CONFIG.ball)
    found = BallEvents(NONE, NONE, np.array([150, 350]), NONE)
    assert restart_candidates(found, ball, CONFIG.events).tolist() == [150]
