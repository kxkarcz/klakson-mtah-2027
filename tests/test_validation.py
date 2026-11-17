from __future__ import annotations

import io
import json

import numpy as np

from possession.structures import ABSTAIN, AWAY, DEAD, HOME
from possession.validation.degrade import (
    DegradationConfig,
    ball_offset,
    degrade_tracking,
    extended_gap_mask,
    live_gap_mask,
    provided_tracker_profile,
    smooth_ball,
    stoppage_loss_mask,
    wrong_ball_offset,
)
from possession.validation.dfl import build_dfl_tracking, parse_match_information
from possession.validation.metrica import (
    CONTROLLED_ACTIONS_WITHOUT_CARRIES,
    MatchEvent,
    build_metrica_tracking,
    home_team_id,
    labels_from_events,
    parse_column_layout,
    parse_events,
    player_speed_from_positions,
    subtype_names,
)
from possession.validation.metrica_csv import build_metrica_csv_tracking, parse_events_csv

from synthetic import grid_team, make_tracking

METADATA = """<main><Metadata>
  <GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>
  <Sessions><Session id="s"><FieldSize><Width>100</Width><Height>50</Height></FieldSize>
  </Session></Sessions>
  <Teams><Team id="TA"><Name>A</Name></Team><Team id="TB"><Name>B</Name></Team></Teams>
  <Players>
    <Player id="P1" teamId="TA"/><Player id="P2" teamId="TB"/><Player id="P3" teamId="TA"/>
  </Players>
  <PlayerChannels>
    <PlayerChannel id="p1_x" playerId="P1"/><PlayerChannel id="p1_y" playerId="P1"/>
    <PlayerChannel id="p2_x" playerId="P2"/><PlayerChannel id="p2_y" playerId="P2"/>
    <PlayerChannel id="p3_x" playerId="P3"/><PlayerChannel id="p3_y" playerId="P3"/>
  </PlayerChannels></Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification startFrame="1" endFrame="2">
      <PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/>
      <PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/>
    </DataFormatSpecification>
    <DataFormatSpecification startFrame="3" endFrame="3">
      <PlayerChannelRef playerChannelId="p3_x"/><PlayerChannelRef playerChannelId="p3_y"/>
      <PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/>
    </DataFormatSpecification>
  </DataFormatSpecifications></main>"""
RAW = "1:0.1,0.2;0.5,0.5:0.3,0.3\n2:0.2,0.2;0.5,0.5:NaN,NaN\n3:0.9,0.8;0.5,0.6:0.4,0.4\n"


def test_columns_follow_substitutions() -> None:
    players, layouts = parse_column_layout(METADATA)
    assert players == ["P1", "P2", "P3"]
    assert layouts == [(1, 2, ["P1", "P2"]), (3, 3, ["P3", "P2"])]
    tracking = build_metrica_tracking(METADATA, RAW, 1, 100.0)
    assert tracking.slot_team.tolist() == [HOME, AWAY, HOME]
    assert tracking.player_xy[0, 0].tolist() == [10.0, 10.0]
    assert np.isnan(tracking.player_xy[2, 0]).all()  # P1 substituted off
    np.testing.assert_allclose(tracking.player_xy[2, 2], [90.0, 40.0])  # P3 in his column
    assert np.isnan(tracking.ball_xy[1]).all()
    assert home_team_id(METADATA) == "TA"


def test_speed_is_derived_from_positions_and_drops_impossible_steps() -> None:
    xy = np.zeros((4, 1, 2))
    xy[:, 0, 0] = [0.0, 0.2, 0.4, 30.0]
    speed = player_speed_from_positions(xy, 25.0, 1, 12.0)
    np.testing.assert_allclose(speed[1:3, 0], 5.0)
    assert np.isnan(speed[3, 0])


def event(kind: str, team: int, start: int, end: int, period: int = 1) -> MatchEvent:
    return MatchEvent(kind, team, start, end, period)


EVENTS = [
    event("SET PIECE", HOME, 10, 10),
    event("PASS", HOME, 10, 30),
    event("RECOVERY", AWAY, 40, 40),  # bare recovery, lost again at once
    event("BALL LOST", AWAY, 41, 45),
    event("RECOVERY", HOME, 50, 50),
    event("CARRY", HOME, 50, 60),
    event("RECOVERY", AWAY, 70, 70),
    event("PASS", AWAY, 72, 80),  # controlled: dated back to the recovery at 70
    event("BALL OUT", AWAY, 85, 90),
    event("SET PIECE", HOME, 120, 120),
    event("PASS", HOME, 120, 130),
]


def test_controlled_reading_ignores_a_bare_recovery_and_backdates_a_real_one() -> None:
    labels = labels_from_events(EVENTS, 150, controlled_only=True)
    assert set(labels[:10]) == {ABSTAIN} and set(labels[131:]) == {ABSTAIN}
    assert set(labels[10:70]) == {HOME}
    assert set(labels[70:91]) == {AWAY}
    assert set(labels[91:120]) == {DEAD}  # from the frame after the ball went out
    assert set(labels[120:131]) == {HOME}


def test_any_recovery_reading_changes_at_every_recovery() -> None:
    labels = labels_from_events(EVENTS, 150, controlled_only=False)
    assert set(labels[10:40]) == {HOME}
    assert set(labels[40:50]) == {AWAY}
    assert set(labels[50:70]) == {HOME}
    assert set(labels[70:91]) == {AWAY}


def test_periods_are_labelled_separately() -> None:
    events = [*EVENTS, event("SET PIECE", AWAY, 300, 300, 2), event("PASS", AWAY, 300, 320, 2)]
    labels = labels_from_events(events, 400, controlled_only=True)
    assert set(labels[131:300]) == {ABSTAIN}
    assert set(labels[300:321]) == {AWAY}


def test_events_are_parsed_to_zero_based_frames() -> None:
    payload = {
        "data": [
            {
                "team": {"id": "TB"},
                "type": {"name": "PASS"},
                "start": {"frame": 11},
                "end": {"frame": 21},
                "period": 2,
            }
        ]
    }
    assert parse_events(json.dumps(payload), "TA") == [MatchEvent("PASS", AWAY, 10, 20, 2)]


LABELS = np.array([HOME] * 400 + [DEAD] * 200 + [AWAY] * 400, dtype=np.int8)


def test_gaps_are_extended_only_around_stoppages() -> None:
    labels = np.tile(LABELS, 5)  # five stoppages, so the random extensions are not all tiny
    dead = labels == DEAD
    always = DegradationConfig(early_loss_probability=1.0, late_pickup_probability=1.0)
    mask = extended_gap_mask(labels, np.random.default_rng(0), always)
    assert mask.any() and not mask[dead].any()
    reach = always.max_extension_frames
    near_stoppage = np.convolve(dead, np.ones(2 * reach + 1), mode="same") > 0
    assert not mask[~near_stoppage].any()
    never = DegradationConfig(early_loss_probability=0.0, late_pickup_probability=0.0)
    assert not extended_gap_mask(labels, np.random.default_rng(0), never).any()


def test_live_gaps_reach_the_requested_share() -> None:
    config = DegradationConfig(live_gap_share=0.2, live_gap_median_frames=20.0)
    mask = live_gap_mask(LABELS, np.random.default_rng(0), config)
    assert mask.sum() >= 0.2 * 800
    assert not live_gap_mask(
        LABELS, np.random.default_rng(0), DegradationConfig(live_gap_share=0.0)
    ).any()


def test_smoothing_never_fills_a_missing_frame() -> None:
    ball = np.column_stack((np.arange(10.0), np.zeros(10)))
    ball[4] = np.nan
    smoothed = smooth_ball(ball, 3)
    assert np.isnan(smoothed[4]).all() and np.isfinite(np.delete(smoothed, 4, axis=0)).all()


def test_degradation_is_reproducible_and_only_touches_the_ball() -> None:
    ball = np.column_stack((np.linspace(10.0, 90.0, 1000), np.full(1000, 30.0)))
    ball[400:600] = np.nan
    tracking = make_tracking(1000, grid_team(30.0), grid_team(70.0), ball)
    first = degrade_tracking(tracking, LABELS, DegradationConfig(seed=3))
    second = degrade_tracking(tracking, LABELS, DegradationConfig(seed=3))
    other = degrade_tracking(tracking, LABELS, DegradationConfig(seed=4))
    assert np.array_equal(first.ball_xy, second.ball_xy, equal_nan=True)
    assert not np.array_equal(first.ball_xy, other.ball_xy, equal_nan=True)
    assert np.array_equal(first.player_xy, tracking.player_xy)
    assert np.isnan(first.ball_xy[:, 0]).sum() > 200


HOME_CSV = """,,,Home,,Home,,,
,,,1,,2,,,
Period,Frame,Time [s],Player1,,Player2,,Ball,
1,1,0.04,0.1,0.2,NaN,NaN,0.3,0.3
1,2,0.08,0.2,0.2,NaN,NaN,NaN,NaN
2,3,0.12,0.3,0.2,0.5,0.5,0.4,0.4
"""
AWAY_CSV = """,,,Away,,,
,,,3,,,
Period,Frame,Time [s],Player3,,Ball,
1,1,0.04,0.9,0.5,0.3,0.3
1,2,0.08,0.9,0.5,NaN,NaN
2,3,0.12,0.9,0.5,0.4,0.4
"""
EVENTS_CSV = """Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To
Home,SET PIECE,KICK OFF,1,1,0.04,1,0.04,Player1,
Away,RECOVERY,INTERCEPTION,1,2,0.08,2,0.08,Player3,
Away,PASS,,2,3,0.12,3,0.12,Player3,Player3
"""


def test_csv_tables_become_one_tracking_with_home_slots_first() -> None:
    tracking = build_metrica_csv_tracking(
        HOME_CSV,
        AWAY_CSV,
        pitch_length_m=100.0,
        pitch_width_m=50.0,
        speed_smoothing_frames=1,
        max_speed_mps=1000.0,
    )
    assert tracking.fps == 25.0
    assert tracking.slot_team.tolist() == [HOME, HOME, AWAY]
    np.testing.assert_allclose(tracking.player_xy[0, 0], [10.0, 10.0])
    assert np.isnan(tracking.player_xy[1, 1]).all()  # substitute not yet on the pitch
    np.testing.assert_allclose(tracking.player_xy[2, 2], [90.0, 25.0])
    assert np.isnan(tracking.ball_xy[1]).all()
    np.testing.assert_allclose(tracking.ball_xy[2], [40.0, 20.0])
    np.testing.assert_allclose(tracking.player_speed[1, 0], 250.0)  # 10 m in one frame


def test_csv_tables_must_describe_the_same_frames() -> None:
    shifted = AWAY_CSV.replace("0.4,0.4", "0.6,0.4")
    try:
        build_metrica_csv_tracking(
            HOME_CSV,
            shifted,
            pitch_length_m=100.0,
            pitch_width_m=50.0,
            speed_smoothing_frames=1,
            max_speed_mps=100.0,
        )
    except ValueError as error:
        assert "same frames" in str(error)
    else:
        raise AssertionError("mismatching ball columns were accepted")


def test_csv_events_use_zero_based_frames_and_team_names() -> None:
    events = parse_events_csv(EVENTS_CSV)
    assert [(e.kind, e.team, e.start_frame, e.end_frame, e.period) for e in events] == [
        ("SET PIECE", HOME, 0, 0, 1),
        ("RECOVERY", AWAY, 1, 1, 1),
        ("PASS", AWAY, 2, 2, 2),
    ]
    assert events[0].subtypes == ("KICK OFF",) and events[2].subtypes == ()


def test_goal_kick_is_a_restart_although_it_is_recorded_as_a_pass() -> None:
    events = [
        event("SET PIECE", HOME, 10, 10),
        event("SHOT", HOME, 20, 30),  # wide: no further event until the goal kick
        MatchEvent("PASS", AWAY, 80, 95, 1, ("GOAL KICK",)),
    ]
    labels = labels_from_events(events, 100, controlled_only=True)
    assert set(labels[10:31]) == {HOME}
    assert set(labels[31:80]) == {DEAD}
    assert set(labels[80:96]) == {AWAY}
    assert MatchEvent("BALL LOST", AWAY, 0, 1, 1, ("GOAL KICK", "INTERCEPTION")).is_restart
    assert not MatchEvent("PASS", AWAY, 0, 1, 1, ("HEAD",)).is_restart


def test_subtypes_are_read_from_null_object_or_list() -> None:
    assert subtype_names(None) == ()
    assert subtype_names({"name": "GOAL KICK"}) == ("GOAL KICK",)
    assert subtype_names([{"name": "FREE KICK"}, {"name": "INDIRECT"}]) == ("FREE KICK", "INDIRECT")


def test_without_carries_a_lost_ball_proves_the_team_had_it() -> None:
    events = [
        event("SET PIECE", HOME, 10, 10),
        event("PASS", HOME, 10, 30),
        event("RECOVERY", AWAY, 40, 40),
        event("BALL LOST", AWAY, 60, 65),  # dribbled for a second, then lost it
        event("RECOVERY", HOME, 66, 66),
        event("PASS", HOME, 70, 80),
    ]
    narrow = labels_from_events(events, 100, controlled_only=True)
    wide = labels_from_events(
        events,
        100,
        controlled_only=True,
        controlled_actions=CONTROLLED_ACTIONS_WITHOUT_CARRIES,
    )
    assert set(narrow[10:81]) == {HOME}
    assert set(wide[40:66]) == {AWAY} and set(wide[66:81]) == {HOME}


DFL_INFORMATION = """<PutDataRequest><MatchInformation>
  <General HomeTeamId="T-H" GuestTeamId="T-G"/>
  <Environment PitchX="100.00" PitchY="60.00"/>
</MatchInformation></PutDataRequest>"""


def dfl_frame(number: int, x: float, y: float, extra: str = "") -> str:
    stamp = f"2022-10-15T15:30:00.{(number % 25) * 40:03d}+02:00"
    return f'<Frame N="{number}" T="{stamp}" X="{x}" Y="{y}" S="0.0"{extra}/>'


def dfl_positions() -> bytes:
    def ball(number: int, status: int, team: int) -> str:
        return dfl_frame(number, 0.0, 0.0, f' Z="0" BallStatus="{status}" BallPossession="{team}"')

    def frame_set(team: str, person: str, section: str, frames: list[str]) -> str:
        head = f'<FrameSet GameSection="{section}" MatchId="M" TeamId="{team}" PersonId="{person}">'
        return head + "".join(frames) + "</FrameSet>"

    sets = [
        frame_set("T-H", "P1", "firstHalf", [dfl_frame(n, -50.0, -30.0) for n in (10, 11, 12)]),
        frame_set("T-G", "P2", "firstHalf", [dfl_frame(n, 10.0, 5.0) for n in (11, 12)]),
        frame_set(
            "BALL", "DFL-OBJ-0000XT", "firstHalf", [ball(10, 1, 1), ball(11, 1, 2), ball(12, 0, 2)]
        ),
        frame_set("T-H", "P1", "secondHalf", [dfl_frame(n, 50.0, 30.0) for n in (100, 101)]),
        frame_set("BALL", "DFL-OBJ-0000XT", "secondHalf", [ball(100, 0, 1), ball(101, 1, 1)]),
    ]
    return (
        "<PutDataRequest><Positions>" + "".join(sets) + "</Positions></PutDataRequest>"
    ).encode()


def test_dfl_labels_are_read_from_ball_status_and_possession() -> None:
    information = parse_match_information(DFL_INFORMATION)
    assert (information.home_team_id, information.pitch_length) == ("T-H", 100.0)
    tracking, labels = build_dfl_tracking(
        io.BytesIO(dfl_positions()),
        information,
        break_frames=2,
        speed_smoothing_frames=1,
        max_speed_mps=1000.0,
    )
    assert tracking.fps == 25.0 and tracking.n_frames == 3 + 2 + 2
    assert labels.tolist() == [HOME, AWAY, DEAD, ABSTAIN, ABSTAIN, DEAD, HOME]
    assert tracking.slot_team.tolist() == [HOME, AWAY]
    np.testing.assert_allclose(tracking.player_xy[0, 0], [0.0, 0.0])  # corner origin
    assert np.isnan(tracking.player_xy[0, 1]).all()  # second player appears a frame later
    np.testing.assert_allclose(tracking.player_xy[1, 1], [60.0, 35.0])
    np.testing.assert_allclose(tracking.player_xy[5, 0], [100.0, 60.0])  # second half
    assert np.isnan(tracking.ball_xy[3:5]).all()  # the break
    np.testing.assert_allclose(tracking.ball_xy[6], [50.0, 30.0])


def test_whole_stoppages_can_lose_the_ball_without_touching_other_faults() -> None:
    labels = np.tile(LABELS, 20)
    assert not stoppage_loss_mask(labels, DegradationConfig()).any()
    config = DegradationConfig(stoppage_loss_probability=0.5)
    mask = stoppage_loss_mask(labels, config)
    assert not mask[labels != DEAD].any()
    lost = mask.reshape(20, -1)[:, 400:600]
    assert set(lost.all(axis=1) | ~lost.any(axis=1)) == {True}  # a stoppage is lost whole or not
    assert 0 < lost.all(axis=1).sum() < 20
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    np.testing.assert_array_equal(
        extended_gap_mask(labels, rng_a, config),
        extended_gap_mask(labels, rng_b, DegradationConfig()),
    )


def test_ball_offset_has_the_configured_spread_and_changes_slowly() -> None:
    assert not ball_offset(1000, DegradationConfig()).any()
    config = DegradationConfig(ball_offset_m=0.7, ball_offset_correlation_frames=150)
    offset = ball_offset(20000, config)
    np.testing.assert_allclose(offset.std(axis=0), 0.7, rtol=1e-6)
    step = np.linalg.norm(np.diff(offset, axis=0), axis=1)
    assert step.mean() < 0.2 * 0.7  # white noise of that spread would move ~1 m per frame
    np.testing.assert_array_equal(offset, ball_offset(20000, config))


def test_wrong_ball_blocks_cover_the_share_at_metres_from_the_ball() -> None:
    assert not wrong_ball_offset(1000, DegradationConfig()).any()
    config = DegradationConfig(wrong_ball_share=0.1, wrong_ball_distance_m=6.0)
    offset = wrong_ball_offset(50000, config)
    moved = np.linalg.norm(offset, axis=1)
    assert 0.1 <= np.mean(moved > 0) < 0.15
    assert moved[moved > 0].min() >= 3.0 and moved.max() <= 9.0


def test_new_faults_leave_the_gaps_as_they_were() -> None:
    labels = np.tile(LABELS, 5)
    tracking = make_tracking(
        labels.size, grid_team(30.0), grid_team(70.0), np.full((labels.size, 2), 50.0)
    )
    plain = degrade_tracking(tracking, labels, DegradationConfig(seed=3))
    fitted = degrade_tracking(tracking, labels, provided_tracker_profile(3, wrong_ball_share=0.1))
    smoothing_only = degrade_tracking(
        tracking, labels, DegradationConfig(seed=3, ball_smoothing_frames=21)
    )
    np.testing.assert_array_equal(np.isnan(plain.ball_xy), np.isnan(smoothing_only.ball_xy))
    np.testing.assert_array_equal(np.isnan(fitted.ball_xy), np.isnan(smoothing_only.ball_xy))
    assert np.nanmax(np.abs(fitted.ball_xy - 50.0)) > 1.0
