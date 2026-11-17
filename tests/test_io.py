from __future__ import annotations

import numpy as np
import pytest

from possession.io.atd import (
    build_tracking,
    parse_raw_tracking,
    parse_slot_teams,
    parse_team_names,
)
from possession.io.smart_tagging import parse_smart_tagging, seconds_to_frame
from possession.structures import AWAY, HOME

from synthetic import METADATA_XML

RAW = (
    "1:0.5,0.5,1.0;0.25,0.75,2.0;NaN,NaN,NaN:0.1,0.2,0.0\n"
    "2:0.5,0.5,1.5;NaN,NaN,NaN;0.9,0.1,3.0:NaN,NaN,NaN\n"
)


def test_slot_teams_follow_metadata_order() -> None:
    assert parse_slot_teams(METADATA_XML).tolist() == [HOME, AWAY, HOME]
    assert parse_team_names(METADATA_XML) == ("Home FC", "Away United")


def test_raw_line_is_parsed_into_players_and_ball() -> None:
    player_xy, speed, ball_xy = parse_raw_tracking(RAW, n_slots=3)
    assert player_xy.shape == (2, 3, 2)
    assert player_xy[0, 1].tolist() == [0.25, 0.75]
    assert speed[1, 2] == 3.0
    assert np.isnan(player_xy[0, 2]).all() and np.isnan(speed[0, 2])
    assert ball_xy[0].tolist() == [0.1, 0.2]
    assert np.isnan(ball_xy[1]).all()


def test_frame_id_minus_one_is_the_row() -> None:
    tracking = build_tracking(METADATA_XML, RAW)
    # frame_id 2 is video frame 1: its only visible home player is slot 0.
    assert tracking.n_frames == 2
    assert tracking.player_xy[1, 0].tolist() == [50.0, 30.0]
    assert tracking.ball_xy[0].tolist() == [10.0, 12.0]
    assert tracking.fps == 25.0


def test_gap_in_frame_ids_is_rejected() -> None:
    broken = RAW.replace("2:", "3:")
    with pytest.raises(ValueError, match="frame ids"):
        parse_raw_tracking(broken, n_slots=3)


def test_wrong_field_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="multiple"):
        parse_raw_tracking("1:0.5,0.5,1.0:0.1,0.2,0.0\n", n_slots=3)


@pytest.mark.parametrize(
    ("seconds", "frame"), [(0.0, 0), (1.0, 25), (447.4, 11185), (0.02, 0), (0.03, 1)]
)
def test_seconds_round_to_frames(seconds: float, frame: int) -> None:
    assert seconds_to_frame(seconds, 25.0) == frame


TAGGING_XML = """<file><ALL_INSTANCES>
<instance><ID>2</ID><start>10.0</start><end>12.5</end><code>Set Pieces</code>
  <label><group>Team</group><text>N/A</text></label>
  <label><group>Type</group><text>Throw-in</text></label></instance>
<instance><ID>1</ID><start>2.0</start><end>4.0</end><code>BUILD UP</code>
  <label><group>Team</group><text>home fc</text></label></instance>
<instance><ID>3</ID><start>20.0</start><end>21.0</end><code>PROGRESSION</code>
  <label><group>Team</group><text>Away United</text></label></instance>
</ALL_INSTANCES></file>"""


def test_smart_tagging_maps_teams_and_sorts_by_start() -> None:
    phases = parse_smart_tagging(TAGGING_XML, 25.0, "Home FC", "Away United")
    assert [phase.code for phase in phases] == ["BUILD UP", "SET PIECES", "PROGRESSION"]
    assert [phase.team for phase in phases] == [HOME, None, AWAY]
    assert (phases[0].start_frame, phases[0].end_frame) == (50, 100)
    assert phases[1].set_piece_type == "throw-in"
    assert phases[0].set_piece_type is None
