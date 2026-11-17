"""Reader for Metrica Sports' public sample match (Sample Game 3) and its event-based labels.

This is a *different* match from the one being labelled, published by the same tracking
provider with hand-checked tracking and event data. It is the only place where the pipeline
can be scored against something that is not its own output, so it is used to measure choices
that label-free diagnostics cannot separate. Nothing here is used at inference time.

Labels are derived from the events mechanically:

* the ball is **dead** from the end of the last on-ball event before a restart until that
  restart is taken. A restart is a set-piece event or a goal kick, which the provider records
  as a pass (or lost ball) with a ``GOAL KICK`` subtype rather than as a set piece;
* while it is alive, the ball belongs to the team of the latest controlled action.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np

from possession.runs import centred_mean
from possession.structures import (
    ABSTAIN,
    AWAY,
    DEAD,
    HOME,
    FloatArray,
    LabelArray,
    Tracking,
)

CONTROLLED_ACTIONS = frozenset({"PASS", "CARRY", "SHOT", "SET PIECE"})
ON_BALL_ACTIONS = frozenset(
    {"PASS", "CARRY", "SHOT", "BALL OUT", "BALL LOST", "RECOVERY", "CHALLENGE", "FAULT RECEIVED"}
)
# Exports without CARRY rows record a failed pass or a lost dribble only as BALL LOST or
# BALL OUT, so those have to count as proof that the team had the ball.
CONTROLLED_ACTIONS_WITHOUT_CARRIES = CONTROLLED_ACTIONS | {"BALL LOST", "BALL OUT"}
RECOVERY = "RECOVERY"
SET_PIECE = "SET PIECE"
GOAL_KICK = "GOAL KICK"


@dataclass(frozen=True)
class MatchEvent:
    """One provider event on the tracking timeline (frames are 0-based rows)."""

    kind: str
    team: int
    start_frame: int
    end_frame: int
    period: int
    subtypes: tuple[str, ...] = ()

    @property
    def is_restart(self) -> bool:
        """Whether this event puts a dead ball back into play."""
        return self.kind == SET_PIECE or GOAL_KICK in self.subtypes


def subtype_names(raw: dict[str, Any] | list[dict[str, Any]] | None) -> tuple[str, ...]:
    """Subtype names from the JSON field, which is null, one object or a list of objects."""
    if raw is None:
        return ()
    items = raw if isinstance(raw, list) else [raw]
    return tuple(str(item["name"]) for item in items)


def parse_events(events_json: str, home_team_id: str) -> list[MatchEvent]:
    """Parse the provider's event JSON, in file (chronological) order.

    Event frames count from 1 like the tracking records, so one is subtracted.
    """
    events = []
    for item in json.loads(events_json)["data"]:
        events.append(
            MatchEvent(
                kind=item["type"]["name"],
                team=HOME if item["team"]["id"] == home_team_id else AWAY,
                start_frame=int(item["start"]["frame"]) - 1,
                end_frame=int(item["end"]["frame"]) - 1,
                period=int(item["period"]),
                subtypes=subtype_names(item.get("subtypes")),
            )
        )
    return events


def labels_from_events(
    events: list[MatchEvent],
    n_frames: int,
    *,
    controlled_only: bool,
    controlled_actions: frozenset[str] = CONTROLLED_ACTIONS,
) -> LabelArray:
    """Per-frame H/A/D labels implied by the events; X outside the two periods.

    ``controlled_only`` decides what a bare recovery is worth. With it, possession changes only
    when the recovering team goes on to a controlled action (pass, carry, shot), and the change
    is dated back to the recovery; without it every recovery changes possession at once. The
    challenge states "a simple touch without control" is not a change, which is the first
    reading; the second is kept to check that conclusions do not hinge on the choice.
    ``controlled_actions`` names the event kinds that prove control; exports without ``CARRY``
    rows need a wider set (see ``CONTROLLED_ACTIONS_WITHOUT_CARRIES``).
    """
    labels = np.full(n_frames, ABSTAIN, dtype=np.int8)
    for period in sorted({event.period for event in events}):
        own = [event for event in events if event.period == period]
        start, stop = own[0].start_frame, max(event.end_frame for event in own) + 1
        current = own[0].team
        pending: tuple[int, int] | None = None
        change_frames: list[tuple[int, int]] = [(start, current)]
        for event in own:
            if event.kind == RECOVERY:
                if controlled_only:
                    pending = (event.team, event.start_frame)
                elif event.team != current:
                    current = event.team
                    change_frames.append((event.start_frame, current))
            elif event.kind in controlled_actions or event.is_restart:
                if event.team != current:
                    current = event.team
                    dated = pending[1] if pending and pending[0] == current else event.start_frame
                    change_frames.append((dated, current))
                pending = None
        frames = np.array([frame for frame, _ in change_frames])
        teams = np.array([team for _, team in change_frames], dtype=np.int8)
        index = np.searchsorted(frames, np.arange(start, stop), side="right") - 1
        labels[start:stop] = teams[np.maximum(index, 0)]
        on_ball_ends = np.array(
            sorted(event.end_frame for event in own if event.kind in ON_BALL_ACTIONS)
        )
        for event in own:
            if not event.is_restart:
                continue
            before = int(np.searchsorted(on_ball_ends, event.start_frame, side="left")) - 1
            dead_from = int(on_ball_ends[before]) + 1 if before >= 0 else start
            labels[max(start, dead_from) : event.start_frame] = DEAD
    return labels


def parse_column_layout(metadata_xml: str) -> tuple[list[str], list[tuple[int, int, list[str]]]]:
    """Player ids, and for each frame range the player occupying each raw column.

    Substitutions change which player a column holds, so the export carries one layout per
    range. Returns ``(all player ids, [(first_frame, last_frame, column players)])`` with
    1-based inclusive frames as written in the file.
    """
    root = ET.fromstring(metadata_xml)
    player_of_channel = {
        channel.get("id", ""): channel.get("playerId", "") for channel in root.iter("PlayerChannel")
    }
    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [ref.get("playerChannelId", "") for ref in spec.iter("PlayerChannelRef")]
        columns = [player_of_channel[ref] for ref in refs[::2]]
        layouts.append((int(spec.get("startFrame", "0")), int(spec.get("endFrame", "0")), columns))
    return [player.get("id", "") for player in root.iter("Player")], layouts


def player_speed_from_positions(
    player_xy: FloatArray, fps: float, smoothing_frames: int, max_speed_mps: float
) -> FloatArray:
    """Scalar speed per player from frame-to-frame displacement, lightly smoothed.

    This export has no speed channel, whereas the pipeline expects the provider's. Implausible
    values (a slot changing hands) are dropped rather than averaged in.
    """
    step = np.full(player_xy.shape[:2], np.nan)
    step[1:] = np.linalg.norm(np.diff(player_xy, axis=0), axis=2) * fps
    step[step > max_speed_mps] = np.nan
    smoothed = np.stack(
        [centred_mean(step[:, slot], smoothing_frames) for slot in range(player_xy.shape[1])],
        axis=1,
    )
    smoothed[np.isnan(player_xy[:, :, 0])] = np.nan
    return smoothed


def build_metrica_tracking(
    metadata_xml: str, raw_text: str, speed_smoothing_frames: int, max_speed_mps: float
) -> Tracking:
    """Assemble a ``Tracking`` with one slot per player (NaN while a player is off the pitch)."""
    root = ET.fromstring(metadata_xml)
    team_ids = [team.get("id") for team in root.iter("Team")]
    team_of_player = {player.get("id", ""): player.get("teamId") for player in root.iter("Player")}
    players, layouts = parse_column_layout(metadata_xml)
    slot_of_player = {player: slot for slot, player in enumerate(players)}
    length = float(root.findtext(".//FieldSize/Width") or "nan")
    width = float(root.findtext(".//FieldSize/Height") or "nan")
    fps = float(root.findtext(".//FrameRate") or "nan")

    records = raw_text.strip().split("\n")
    player_xy = np.full((len(records), len(players), 2), np.nan)
    ball_xy = np.full((len(records), 2), np.nan)
    for first, last, columns in layouts:
        slots = [slot_of_player[player] for player in columns]
        for row in range(first - 1, min(last, len(records))):
            frame_id, body, ball = records[row].split(":")
            if int(frame_id) != row + 1:
                raise ValueError("raw tracking frame ids are not exactly 1..n in order")
            values = np.array(body.replace(";", ",").split(","), dtype=np.float64)
            player_xy[row, slots] = values.reshape(-1, 2)
            ball_xy[row] = np.array(ball.split(",")[:2], dtype=np.float64)
    scale = np.array([length, width])
    player_xy *= scale
    return Tracking(
        player_xy=player_xy,
        player_speed=player_speed_from_positions(
            player_xy, fps, speed_smoothing_frames, max_speed_mps
        ),
        slot_team=np.array(
            [HOME if team_of_player[player] == team_ids[0] else AWAY for player in players],
            dtype=np.int8,
        ),
        ball_xy=ball_xy * scale,
        fps=fps,
        pitch_length=length,
        pitch_width=width,
    )


def home_team_id(metadata_xml: str) -> str:
    """Return the identifier of the first (home) team in the metadata."""
    team = next(iter(ET.fromstring(metadata_xml).iter("Team")))
    return team.get("id", "")
