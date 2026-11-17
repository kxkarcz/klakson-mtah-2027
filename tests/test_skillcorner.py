from __future__ import annotations

import json

import numpy as np

from possession.structures import ABSTAIN, AWAY, DEAD, HOME
from possession.validation.skillcorner import (
    build_skillcorner_tracking,
    parse_match,
    phase_mask,
    resample_labels,
    resample_positions,
    source_labels,
)

MATCH = json.dumps(
    {
        "home_team": {"id": 10},
        "away_team": {"id": 20},
        "players": [{"id": 1, "team_id": 10}, {"id": 2, "team_id": 20}],
        "match_periods": [
            {"period": 1, "start_frame": 1, "end_frame": 6},
            {"period": 2, "start_frame": None, "end_frame": None},
        ],
        "pitch_length": 100,
        "pitch_width": 60,
    }
)
PHASES = "frame_start,frame_end,team_in_possession_id\n2,3,10\n5,6,20\n"


def frame(
    number: int, group: str | None, ball: tuple[float, float] | None, detected: bool = True
) -> str:
    return json.dumps(
        {
            "frame": number,
            "possession": {"group": group, "player_id": None},
            "ball_data": {
                "x": None if ball is None else ball[0],
                "y": None if ball is None else ball[1],
                "z": 0.0,
                "is_detected": detected,
            },
            "player_data": [
                {"player_id": 1, "x": -50.0 + number, "y": 0.0, "is_detected": True},
                {"player_id": 2, "x": 10.0, "y": 5.0, "is_detected": False},
            ],
        }
    )


TRACKING = "\n".join(
    [
        frame(0, None, None),
        frame(1, None, (0.0, 0.0)),
        frame(2, "home team", (1.0, 0.0)),
        frame(3, None, (2.0, 0.0)),  # inside a phase without a group: carried forward
        frame(4, None, (3.0, 0.0), detected=False),  # between phases: dead; ball not detected
        frame(5, "away team", (4.0, 0.0)),
        frame(6, "away team", (5.0, 0.0)),
        frame(7, None, None),
    ]
)


def test_match_file_gives_teams_periods_and_pitch() -> None:
    match = parse_match(MATCH)
    assert match.team_of_player == {1: HOME, 2: AWAY}
    assert match.periods == [(1, 6)]
    assert (match.pitch_length, match.pitch_width) == (100.0, 60.0)


def test_labels_follow_the_group_inside_phases_and_are_dead_between_them() -> None:
    group = np.array([-1, -1, HOME, -1, -1, AWAY, AWAY, -1], dtype=np.int8)
    labels = source_labels(group, phase_mask(PHASES, 8), [(1, 6)])
    assert labels.tolist() == [ABSTAIN, DEAD, HOME, HOME, DEAD, AWAY, AWAY, ABSTAIN]


def test_resampling_interpolates_positions_and_takes_the_nearest_label() -> None:
    xy = np.array([[0.0], [10.0], [np.nan], [30.0]])
    np.testing.assert_allclose(resample_positions(xy, 2.5)[:3, 0], [0.0, 4.0, 8.0])
    assert np.isnan(resample_positions(xy, 2.5)[3:7, 0]).all()  # touches the missing frame
    labels = np.array([HOME, AWAY, DEAD], dtype=np.int8)
    assert resample_labels(labels, 2.5).tolist() == [HOME, HOME, AWAY, AWAY, DEAD, DEAD]


def test_tracking_is_resampled_shifted_and_keeps_only_a_detected_ball() -> None:
    tracking, labels = build_skillcorner_tracking(
        TRACKING,
        MATCH,
        PHASES,
        target_fps=25.0,
        speed_smoothing_frames=1,
        max_speed_mps=1000.0,
    )
    assert tracking.fps == 25.0 and tracking.n_frames == labels.size == 18
    assert tracking.slot_team.tolist() == [HOME, AWAY]
    np.testing.assert_allclose(tracking.player_xy[0, 0], [0.0, 30.0])  # corner origin
    np.testing.assert_allclose(tracking.player_xy[5, 0], [2.0, 30.0])  # two source frames on
    assert np.isnan(tracking.ball_xy[0]).all()
    np.testing.assert_allclose(tracking.ball_xy[5], [51.0, 30.0])
    assert np.isnan(tracking.ball_xy[10]).all()  # source frame 4: ball extrapolated, dropped
    assert labels[5] == HOME and labels[10] == DEAD and labels[-1] == ABSTAIN
