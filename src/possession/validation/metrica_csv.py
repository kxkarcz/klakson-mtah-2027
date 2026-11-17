"""Reader for the CSV export of Metrica Sports' public sample matches (Sample Games 1 and 2).

The same provider published two further matches in an older, tabular layout: one event table
and one tracking table per team. They carry the same information as Sample Game 3 except that
the event list has no ``CARRY`` rows and the files do not state the pitch size, which the
provider documents separately. Events and tracking are mapped onto the structures used for
Sample Game 3 so the labelling rules and the scoring are shared. Nothing here is used at
inference time.
"""

from __future__ import annotations

import csv
import io

import numpy as np

from possession.structures import AWAY, HOME, FloatArray, Tracking
from possession.validation.metrica import MatchEvent, player_speed_from_positions

HOME_NAME = "Home"
SUBTYPE_SEPARATOR = "-"
HEADER_ROWS = 3
LEADING_COLUMNS = 3  # period, frame, time
BALL_COLUMNS = 2
MISSING = frozenset({"", "NaN", "nan"})


def parse_events_csv(events_csv: str) -> list[MatchEvent]:
    """Parse the provider's event table, in file (chronological) order.

    Event frames count from 1 like the tracking rows, so one is subtracted.
    """
    events = []
    for row in csv.DictReader(io.StringIO(events_csv)):
        events.append(
            MatchEvent(
                kind=row["Type"],
                team=HOME if row["Team"] == HOME_NAME else AWAY,
                start_frame=int(row["Start Frame"]) - 1,
                end_frame=int(row["End Frame"]) - 1,
                period=int(row["Period"]),
                subtypes=tuple(part for part in row["Subtype"].split(SUBTYPE_SEPARATOR) if part),
            )
        )
    return events


def _read_team_table(tracking_csv: str) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(time_s, player_xy, ball_xy)`` of one team's table in pitch fractions."""
    rows = list(csv.reader(io.StringIO(tracking_csv)))[HEADER_ROWS:]
    table = np.array(
        [[np.nan if value in MISSING else float(value) for value in row] for row in rows],
        dtype=np.float64,
    )
    if not np.array_equal(table[:, 1], np.arange(1, len(table) + 1)):
        raise ValueError("tracking frames are not exactly 1..n in order")
    players = table[:, LEADING_COLUMNS:-BALL_COLUMNS]
    return table[:, 2], players.reshape(len(table), -1, 2), table[:, -BALL_COLUMNS:]


def build_metrica_csv_tracking(
    home_csv: str,
    away_csv: str,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    speed_smoothing_frames: int,
    max_speed_mps: float,
) -> Tracking:
    """Assemble a ``Tracking`` from the two team tables (NaN while a player is off the pitch).

    The frame rate is read from the time column; the pitch size is not in the files and has
    to be supplied.
    """
    time_s, home_xy, ball = _read_team_table(home_csv)
    _, away_xy, away_ball = _read_team_table(away_csv)
    if len(home_xy) != len(away_xy) or not np.array_equal(ball, away_ball, equal_nan=True):
        raise ValueError("home and away tracking tables do not describe the same frames")
    fps = float(np.round(1.0 / np.median(np.diff(time_s))))
    scale = np.array([pitch_length_m, pitch_width_m])
    player_xy = np.concatenate([home_xy, away_xy], axis=1) * scale
    return Tracking(
        player_xy=player_xy,
        player_speed=player_speed_from_positions(
            player_xy, fps, speed_smoothing_frames, max_speed_mps
        ),
        slot_team=np.array(
            [HOME] * home_xy.shape[1] + [AWAY] * away_xy.shape[1],
            dtype=np.int8,
        ),
        ball_xy=ball * scale,
        fps=fps,
        pitch_length=pitch_length_m,
        pitch_width=pitch_width_m,
    )
