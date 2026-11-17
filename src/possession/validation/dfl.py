"""Reader for the open DFL matches (Bassek et al., Scientific Data 2025; CC-BY 4.0).

Seven Bundesliga and 2. Bundesliga matches with TRACAB tracking in the DFL position format.
Unlike the Metrica samples, every ball frame carries the provider's own ``BallStatus`` (in
play or not) and ``BallPossession`` (home or away), so the reference labels are read, not
derived: dead where the tracking says the ball is dead, otherwise the team the tracking names.
That is how the challenge describes its own ground truth. Nothing here is used at inference
time.

Positions are metres with the origin at the centre spot; they are shifted to the corner origin
used by the pipeline. Each half has its own frame numbering; the halves are laid end to end
with an empty break between them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import IO

import numpy as np

from possession.structures import (
    ABSTAIN,
    AWAY,
    DEAD,
    HOME,
    FloatArray,
    LabelArray,
    Tracking,
)
from possession.validation.metrica import player_speed_from_positions

BALL_TEAM = "BALL"
STATUS_ALIVE = 1
POSSESSION_HOME = 1
POSSESSION_AWAY = 2


@dataclass(frozen=True)
class MatchInformation:
    """What the match-information file contributes: who is at home and the pitch size."""

    home_team_id: str
    pitch_length: float
    pitch_width: float


@dataclass(frozen=True)
class _FrameSet:
    """One object's frames within one half."""

    team_id: str
    person_id: str
    section: str
    frame_numbers: np.ndarray
    xy: FloatArray
    status: np.ndarray
    possession: np.ndarray
    first_times: tuple[str, ...]


def parse_match_information(information_xml: str) -> MatchInformation:
    """Read the home team identifier and the pitch size (metres)."""
    root = ET.fromstring(information_xml)
    general = next(root.iter("General"))
    environment = next(root.iter("Environment"))
    return MatchInformation(
        home_team_id=general.get("HomeTeamId", ""),
        pitch_length=float(environment.get("PitchX", "nan")),
        pitch_width=float(environment.get("PitchY", "nan")),
    )


def _read_frame_sets(positions: str | IO[bytes]) -> list[_FrameSet]:
    """Stream the position file, keeping one compact record per object and half."""
    frame_sets: list[_FrameSet] = []
    header: dict[str, str] = {}
    numbers: list[int] = []
    coordinates: list[float] = []
    status: list[int] = []
    possession: list[int] = []
    times: list[str] = []
    for event, element in ET.iterparse(positions, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            header = dict(element.attrib)
            numbers, coordinates, status, possession, times = [], [], [], [], []
        elif event == "end" and element.tag == "Frame":
            numbers.append(int(element.get("N", "0")))
            coordinates.append(float(element.get("X", "nan")))
            coordinates.append(float(element.get("Y", "nan")))
            if "BallStatus" in element.attrib:
                status.append(int(element.get("BallStatus", "0")))
                possession.append(int(element.get("BallPossession", "0")))
                if len(times) < 2:  # two timestamps are enough to read the frame rate
                    times.append(element.get("T", ""))
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            frame_sets.append(
                _FrameSet(
                    team_id=header.get("TeamId", ""),
                    person_id=header.get("PersonId", ""),
                    section=header.get("GameSection", ""),
                    frame_numbers=np.array(numbers, dtype=np.int64),
                    xy=np.array(coordinates, dtype=np.float64).reshape(-1, 2),
                    status=np.array(status, dtype=np.int8),
                    possession=np.array(possession, dtype=np.int8),
                    first_times=tuple(times),
                )
            )
            element.clear()
    return frame_sets


def _frame_rate(ball: _FrameSet) -> float:
    """Frames per second from the first two ball timestamps."""
    first, second = (datetime.fromisoformat(stamp) for stamp in ball.first_times)
    return float(np.round(1.0 / (second - first).total_seconds()))


def build_dfl_tracking(
    positions: str | IO[bytes],
    information: MatchInformation,
    *,
    break_frames: int,
    speed_smoothing_frames: int,
    max_speed_mps: float,
) -> tuple[Tracking, LabelArray]:
    """Assemble a ``Tracking`` and the provider's per-frame H/A/D labels.

    ``positions`` is a path or an open binary file. One slot per player (NaN while he is off
    the pitch). ``break_frames`` empty frames, labelled X, separate consecutive halves.
    """
    frame_sets = _read_frame_sets(positions)
    balls = sorted(
        (item for item in frame_sets if item.team_id.upper() == BALL_TEAM),
        key=lambda item: int(item.frame_numbers[0]),
    )
    if not balls:
        raise ValueError("the position file has no ball frames")
    first_number: dict[str, int] = {}
    offset: dict[str, int] = {}
    total = 0
    for ball in balls:
        first_number[ball.section] = int(ball.frame_numbers.min())
        offset[ball.section] = total
        total += int(ball.frame_numbers.max()) - first_number[ball.section] + 1 + break_frames
    total -= break_frames

    players = [item for item in frame_sets if item.team_id.upper() != BALL_TEAM]
    slot_of_person = {
        person: slot for slot, person in enumerate(dict.fromkeys(p.person_id for p in players))
    }
    team_of_person = {item.person_id: item.team_id for item in players}
    shift = np.array([information.pitch_length / 2.0, information.pitch_width / 2.0])

    player_xy = np.full((total, len(slot_of_person), 2), np.nan)
    ball_xy = np.full((total, 2), np.nan)
    labels = np.full(total, ABSTAIN, dtype=np.int8)
    for item in players:
        rows = item.frame_numbers - first_number[item.section] + offset[item.section]
        player_xy[rows, slot_of_person[item.person_id]] = item.xy + shift
    for ball in balls:
        rows = ball.frame_numbers - first_number[ball.section] + offset[ball.section]
        ball_xy[rows] = ball.xy + shift
        team = np.where(ball.possession == POSSESSION_HOME, HOME, AWAY)
        known = np.isin(ball.possession, (POSSESSION_HOME, POSSESSION_AWAY))
        alive = ball.status == STATUS_ALIVE
        labels[rows] = np.where(alive, np.where(known, team, ABSTAIN), DEAD)

    fps = _frame_rate(balls[0])
    tracking = Tracking(
        player_xy=player_xy,
        player_speed=player_speed_from_positions(
            player_xy, fps, speed_smoothing_frames, max_speed_mps
        ),
        slot_team=np.array(
            [
                HOME if team_of_person[person] == information.home_team_id else AWAY
                for person in slot_of_person
            ],
            dtype=np.int8,
        ),
        ball_xy=ball_xy,
        fps=fps,
        pitch_length=information.pitch_length,
        pitch_width=information.pitch_width,
    )
    return tracking, labels
