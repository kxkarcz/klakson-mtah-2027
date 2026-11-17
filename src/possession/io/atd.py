"""Reader for the FIFA/EPTS automatic tracking export (metadata XML + raw text)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from possession.structures import AWAY, HOME, FloatArray, LabelArray, Tracking

_SEPARATORS_TO_COMMA = str.maketrans({":": ",", ";": ",", "\n": ","})
_VALUES_PER_OBJECT = 3  # x, y and one extra channel (speed for players)


def parse_team_names(metadata_xml: str) -> tuple[str, str]:
    """(home, away) club names; the first ``Team`` element is the home side (FIFA/EPTS)."""
    names = [team.findtext("Name") or "" for team in ET.fromstring(metadata_xml).iter("Team")]
    if len(names) != 2:
        raise ValueError(f"expected 2 teams in tracking metadata, found {len(names)}")
    return names[0], names[1]


def parse_slot_teams(metadata_xml: str) -> LabelArray:
    """Team of every track slot, in the column order of the raw file.

    The first ``Team`` element is the home side (FIFA/EPTS convention, also what kloppy and the
    official baseline assume), which is what ties the label "H" to a set of slots.
    """
    root = ET.fromstring(metadata_xml)
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"expected 2 teams in tracking metadata, found {len(team_ids)}")
    slot_team_ids = [player.get("teamId") for player in root.iter("Player")]
    if not slot_team_ids:
        raise ValueError("tracking metadata lists no track slots")
    if not set(slot_team_ids) <= set(team_ids):
        raise ValueError("a track slot references a team that is not declared")
    return np.array([HOME if tid == team_ids[0] else AWAY for tid in slot_team_ids], dtype=np.int8)


def parse_frame_rate(metadata_xml: str) -> float:
    """Frames per second declared by the provider."""
    text = ET.fromstring(metadata_xml).findtext(".//FrameRate")
    if text is None:
        raise ValueError("tracking metadata has no FrameRate")
    return float(text)


def parse_pitch_size(metadata_xml: str) -> tuple[float, float]:
    """(length, width) in metres; raw coordinates are fractions of these."""
    root = ET.fromstring(metadata_xml)
    length = root.findtext(".//FieldSize/Width")
    width = root.findtext(".//FieldSize/Height")
    if length is None or width is None:
        raise ValueError("tracking metadata has no FieldSize")
    return float(length), float(width)


def parse_raw_tracking(raw_text: str, n_slots: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Parse raw records into (player_xy, player_speed, ball_xy), in normalised pitch units.

    Each record is ``frame_id:x,y,s;...;x,y,s:bx,by,bz``. The frame id counts from 1, so record
    ``k`` lands on row ``frame_id - 1``; ids must be exactly 1..n because a silent gap would
    shift every later label by a frame, which the ranking metric punishes at each transition.
    """
    values_per_record = 1 + _VALUES_PER_OBJECT * (n_slots + 1)
    tokens = raw_text.strip().translate(_SEPARATORS_TO_COMMA).split(",")
    if not tokens or len(tokens) % values_per_record != 0:
        raise ValueError(
            f"raw tracking has {len(tokens)} values, not a multiple of {values_per_record}"
        )
    table = np.array(tokens, dtype=np.float64).reshape(-1, values_per_record)
    if not np.array_equal(table[:, 0], np.arange(1, table.shape[0] + 1)):
        raise ValueError("raw tracking frame ids are not exactly 1..n in order")
    players = table[:, 1 : 1 + _VALUES_PER_OBJECT * n_slots].reshape(-1, n_slots, 3)
    ball = table[:, 1 + _VALUES_PER_OBJECT * n_slots :]
    return (
        np.ascontiguousarray(players[:, :, :2]),
        np.ascontiguousarray(players[:, :, 2]),
        np.ascontiguousarray(ball[:, :2]),
    )


def build_tracking(metadata_xml: str, raw_text: str) -> Tracking:
    """Assemble a ``Tracking`` in metres from the two file contents."""
    slot_team = parse_slot_teams(metadata_xml)
    length, width = parse_pitch_size(metadata_xml)
    player_xy, player_speed, ball_xy = parse_raw_tracking(raw_text, int(slot_team.size))
    scale = np.array([length, width])
    return Tracking(
        player_xy=player_xy * scale,
        player_speed=player_speed,
        slot_team=slot_team,
        ball_xy=ball_xy * scale,
        fps=parse_frame_rate(metadata_xml),
        pitch_length=length,
        pitch_width=width,
    )


def read_tracking(metadata_path: Path, raw_path: Path) -> Tracking:
    """Read and parse the tracking pair from disk."""
    return build_tracking(
        metadata_path.read_text(encoding="utf-8"), raw_path.read_text(encoding="utf-8")
    )
