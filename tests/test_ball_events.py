from __future__ import annotations

import numpy as np

from possession.consistency import (
    boundary_event_alignment,
    distance_to_nearest,
    possession_switches,
    set_piece_consistency,
    share_near,
    stoppage_starts,
)
from possession.features.ball_events import (
    BallEvents,
    detect_ball_events,
    line_crossings,
    lost_at_line,
    reappearances,
    restart_kicks,
    trajectory_kinks,
    velocity_change,
)
from possession.io.config_file import load_config
from possession.preprocessing.ball import clean_ball
from possession.structures import AWAY, DEAD, HOME, TaggedPhase

from synthetic import FPS, grid_team, make_tracking
from tests_paths import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)


def track(ball_xy: np.ndarray):
    tracking = make_tracking(ball_xy.shape[0], grid_team(30.0), grid_team(70.0), ball_xy)
    return clean_ball(tracking, CONFIG.ball)


def straight(n: int, start: tuple[float, float], velocity: tuple[float, float]) -> np.ndarray:
    """A ball moving at constant velocity (m/s)."""
    return np.asarray(start) + np.arange(n)[:, None] * np.asarray(velocity) / FPS


def test_line_crossing_is_the_first_frame_beyond_the_line() -> None:
    # 10 m/s towards the touchline y = 0 from 4 m inside: beyond the 0.5 m margin at frame 12.
    ball = track(straight(40, (50.0, 4.0), (0.0, -10.0)))
    assert line_crossings(ball, CONFIG.ball.out_of_bounds_margin_m).tolist() == [12]


def test_ball_lost_at_the_line_heading_out_is_an_exit() -> None:
    xy = straight(200, (50.0, 6.0), (0.0, -10.0))
    xy[12:] = np.nan  # lost 1.6 m inside the line, still heading out
    events = lost_at_line(track(xy), CONFIG.events, CONFIG.ball.short_gap_max_frames)
    assert events.tolist() == [12]


def test_ball_lost_in_midfield_or_heading_infield_is_not_an_exit() -> None:
    midfield = straight(200, (50.0, 30.0), (0.0, -10.0))
    midfield[12:] = np.nan
    infield = straight(200, (50.0, 1.0), (0.0, 10.0))
    infield[5:] = np.nan
    for xy in (midfield, infield):
        events = lost_at_line(track(xy), CONFIG.events, CONFIG.ball.short_gap_max_frames)
        assert events.size == 0


def test_restart_kick_is_the_first_moving_frame_after_rest() -> None:
    xy = np.concatenate((np.full((60, 2), [40.0, 30.0]), straight(40, (40.0, 30.0), (15.0, 0.0))))
    kicks = restart_kicks(track(xy), CONFIG.events, CONFIG.ball.stationary_speed_mps)
    assert kicks.size == 1
    assert abs(int(kicks[0]) - 60) <= 2  # speed is smoothed over a few frames


def test_nudged_or_briefly_resting_ball_is_not_a_restart() -> None:
    nudged = np.concatenate(
        (np.full((60, 2), [40.0, 30.0]), straight(40, (40.0, 30.0), (2.0, 0.0)))
    )
    brief = np.concatenate(
        (
            straight(30, (30.0, 30.0), (8.0, 0.0)),
            np.full((10, 2), [39.6, 30.0]),
            straight(40, (39.6, 30.0), (15.0, 0.0)),
        )
    )
    for xy in (nudged, brief):
        assert restart_kicks(track(xy), CONFIG.events, CONFIG.ball.stationary_speed_mps).size == 0


def test_reappearance_needs_a_long_gap() -> None:
    xy = straight(400, (20.0, 30.0), (3.0, 0.0))
    xy[50:80] = np.nan  # 30 frames: an occlusion
    xy[150:300] = np.nan  # 150 frames: a real disappearance
    assert reappearances(track(xy), CONFIG.ball.short_gap_max_frames).tolist() == [300]


def test_kink_marks_the_touch_and_ignores_free_flight() -> None:
    first = straight(40, (30.0, 30.0), (12.0, 0.0))
    second = straight(40, tuple(first[-1]), (0.0, 12.0))[1:]
    ball = track(np.concatenate((first, second)))
    kinks = trajectory_kinks(ball, FPS, CONFIG.events)
    assert kinks.size == 1
    assert abs(int(kinks[0]) - 39) <= 1
    flight = track(straight(80, (30.0, 30.0), (12.0, 0.0)))
    assert trajectory_kinks(flight, FPS, CONFIG.events).size == 0


def test_velocity_change_is_nan_where_the_ball_is_unknown() -> None:
    xy = straight(30, (30.0, 30.0), (10.0, 0.0))
    xy[10] = np.nan
    change = velocity_change(xy, FPS, 3)
    # Frame t uses positions t-3, t and t+3, so exactly three frames lose their estimate.
    assert np.flatnonzero(np.isnan(change[3:27])).tolist() == [4, 7, 10]
    np.testing.assert_allclose(change[14:27], 0.0, atol=1e-9)


def test_detect_ball_events_combines_the_detectors() -> None:
    xy = np.concatenate(
        (
            np.full((60, 2), [40.0, 30.0]),
            straight(100, (40.0, 30.0), (0.0, -15.0)),  # struck towards the touchline
        )
    )
    events = detect_ball_events(track(xy), FPS, CONFIG.events, CONFIG.ball)
    assert events.restart_kick.size == 1
    assert events.out_of_play.size == 1
    assert events.out_of_play[0] > events.restart_kick[0]


LABELS = np.array(
    [DEAD] * 100 + [HOME] * 200 + [AWAY] * 200 + [DEAD] * 100 + [AWAY] * 100, dtype=np.int8
)


def test_boundary_extractors() -> None:
    assert stoppage_starts(LABELS).tolist() == [500]
    assert possession_switches(LABELS).tolist() == [300]


def test_distance_to_nearest_event() -> None:
    frames = np.array([0, 50, 100])
    assert distance_to_nearest(frames, np.array([40, 130])).tolist() == [40.0, 10.0, 30.0]
    assert np.isinf(distance_to_nearest(frames, np.array([], dtype=np.int64))).all()
    assert share_near(frames, np.array([40, 130]), 10) == round(1 / 3, 4)
    assert share_near(np.array([], dtype=np.int64), frames, 10) is None


def test_boundary_alignment_reports_shares() -> None:
    events = BallEvents(
        out_of_play=np.array([495]),
        restart_kick=np.array([101]),
        reappearance=np.array([900]),
        kink=np.array([330]),
    )
    summary = boundary_event_alignment(LABELS, events, CONFIG.report)
    assert summary["stoppage_starts_at_ball_exit"] == 1.0
    assert summary["restarts_at_kick_or_reappearance"] == 0.5  # 100 has a kick, 600 has none
    assert summary["switches_at_trajectory_kink"] == 0.0  # 30 frames away


def test_set_piece_consistency_measures_spread_around_the_typical_offset() -> None:
    labels = np.array(([DEAD] * 100 + [HOME] * 400) * 4, dtype=np.int8)  # restarts at 100 + 500k
    leads = (125, 125, 125, 300)  # three tags open 5 s before the restart, one 12 s
    phases = [
        TaggedPhase("SET PIECES", 100 + 500 * i - lead, 100 + 500 * i + 50, HOME, "free kick")
        for i, lead in enumerate(leads)
    ]
    summary = set_piece_consistency(labels, phases, FPS, CONFIG.tagging, CONFIG.report)
    assert summary["median_offset_s"] == 5.0
    assert summary["within_tight"] == 0.75
    assert summary["team_agreement"] == 1.0
