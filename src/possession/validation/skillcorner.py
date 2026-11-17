"""Reader for SkillCorner's open broadcast-tracking matches (A-League 2024/25; MIT licence).

Every frame carries SkillCorner's own ``possession.group`` (home or away team). It is present on
97% of the frames inside a phase of play and on under 7% of the others, and phases of play are
defined only while the ball is in play, so the reference labels are: the group where a phase of
play is running, carried forward across the few frames of a phase without one; dead ball inside
a period but outside every phase; X outside the periods. Nothing here is used at inference time.

The data come at 10 frames per second with positions in metres from the centre spot. They are
resampled to the 25 fps of the challenge by linear interpolation of positions (labels by the
nearest original frame) and shifted to the corner origin. Players not detected on the broadcast
image are extrapolated by the provider and kept; the ball is kept only where it was detected,
because an extrapolated ball would hide exactly the gaps the pipeline must cope with.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

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

SOURCE_FPS = 10.0
HOME_GROUP = "home team"
AWAY_GROUP = "away team"


@dataclass(frozen=True)
class SkillCornerMatch:
    """What the match file contributes."""

    home_team_id: int
    team_of_player: dict[int, int]  # player id -> HOME / AWAY
    periods: list[tuple[int, int]]  # inclusive source frames
    pitch_length: float
    pitch_width: float


def parse_match(match_json: str) -> SkillCornerMatch:
    """Read teams, players, periods and pitch size from ``{id}_match.json``."""
    match = json.loads(match_json)
    home = int(match["home_team"]["id"])
    team_of_player = {
        int(player["id"]): HOME if int(player["team_id"]) == home else AWAY
        for player in match["players"]
    }
    periods = [
        (int(period["start_frame"]), int(period["end_frame"]))
        for period in match["match_periods"]
        if period.get("start_frame") is not None and period.get("end_frame") is not None
    ]
    return SkillCornerMatch(
        home_team_id=home,
        team_of_player=team_of_player,
        periods=periods,
        pitch_length=float(match["pitch_length"]),
        pitch_width=float(match["pitch_width"]),
    )


def phase_mask(phases_csv: str, n_frames: int) -> np.ndarray:
    """Source frames covered by a phase of play (the ball is in play)."""
    covered = np.zeros(n_frames, dtype=bool)
    for row in csv.DictReader(io.StringIO(phases_csv)):
        covered[int(row["frame_start"]) : int(row["frame_end"]) + 1] = True
    return covered


def source_labels(
    group: np.ndarray, in_phase: np.ndarray, periods: list[tuple[int, int]]
) -> LabelArray:
    """H/A/D/X per source frame from the possession group (-1 unknown) and the phase mask."""
    labels = np.full(group.size, ABSTAIN, dtype=np.int8)
    for start, stop in periods:
        span = slice(start, stop + 1)
        labels[span] = DEAD
        known = np.flatnonzero((group[span] >= 0) & in_phase[span]) + start
        live = np.flatnonzero(in_phase[span]) + start
        if known.size == 0:
            continue
        # carry the last known team forward (and the first one back) over unknown live frames
        position = np.searchsorted(known, live, side="right") - 1
        labels[live] = group[known[np.maximum(position, 0)]]
    return labels


def read_frames(
    tracking_jsonl: str | io.TextIOBase, match: SkillCornerMatch
) -> tuple[FloatArray, FloatArray, np.ndarray, list[int]]:
    """Return source-rate ``(player_xy, ball_xy, group, player ids)`` with one slot per player."""
    players = sorted(match.team_of_player)
    slot_of = {player: slot for slot, player in enumerate(players)}
    lines = tracking_jsonl.splitlines() if isinstance(tracking_jsonl, str) else tracking_jsonl
    records = [json.loads(line) for line in lines if line.strip()]
    n_frames = max(int(record["frame"]) for record in records) + 1
    player_xy = np.full((n_frames, len(players), 2), np.nan)
    ball_xy = np.full((n_frames, 2), np.nan)
    group = np.full(n_frames, -1, dtype=np.int8)
    for record in records:
        frame = int(record["frame"])
        for player in record.get("player_data") or []:
            slot = slot_of.get(int(player["player_id"]))
            if slot is not None and player.get("x") is not None:
                player_xy[frame, slot] = (player["x"], player["y"])
        ball = record.get("ball_data") or {}
        if ball.get("is_detected") and ball.get("x") is not None:
            ball_xy[frame] = (ball["x"], ball["y"])
        side = (record.get("possession") or {}).get("group")
        group[frame] = HOME if side == HOME_GROUP else AWAY if side == AWAY_GROUP else -1
    return player_xy, ball_xy, group, players


def resample_positions(xy: FloatArray, factor: float) -> FloatArray:
    """Linear interpolation along time to ``factor`` times the frame rate; NaN stays NaN.

    A target frame between two source frames gets a value only when both of them have one.
    """
    n_target = int(np.floor((xy.shape[0] - 1) * factor)) + 1
    time = np.arange(n_target) / factor
    low = np.floor(time).astype(int)
    high = np.minimum(low + 1, xy.shape[0] - 1)
    weight = (time - low).reshape((-1,) + (1,) * (xy.ndim - 1))
    return np.asarray(xy[low] * (1.0 - weight) + xy[high] * weight, dtype=np.float64)


def resample_labels(labels: LabelArray, factor: float) -> LabelArray:
    """Labels at ``factor`` times the frame rate, each from the nearest source frame."""
    n_target = int(np.floor((labels.size - 1) * factor)) + 1
    nearest = np.minimum(np.round(np.arange(n_target) / factor).astype(int), labels.size - 1)
    return np.asarray(labels[nearest], dtype=np.int8)


def build_skillcorner_tracking(
    tracking_jsonl: str | io.TextIOBase,
    match_json: str,
    phases_csv: str,
    *,
    target_fps: float,
    speed_smoothing_frames: int,
    max_speed_mps: float,
) -> tuple[Tracking, LabelArray]:
    """Assemble a ``Tracking`` at ``target_fps`` and SkillCorner's per-frame H/A/D labels."""
    match = parse_match(match_json)
    player_xy, ball_xy, group, players = read_frames(tracking_jsonl, match)
    labels = source_labels(group, phase_mask(phases_csv, group.size), match.periods)
    factor = target_fps / SOURCE_FPS
    shift = np.array([match.pitch_length / 2.0, match.pitch_width / 2.0])
    player_xy = resample_positions(player_xy, factor) + shift
    tracking = Tracking(
        player_xy=player_xy,
        player_speed=player_speed_from_positions(
            player_xy, target_fps, speed_smoothing_frames, max_speed_mps
        ),
        slot_team=np.array([match.team_of_player[player] for player in players], dtype=np.int8),
        ball_xy=resample_positions(ball_xy, factor) + shift,
        fps=target_fps,
        pitch_length=match.pitch_length,
        pitch_width=match.pitch_width,
    )
    return tracking, resample_labels(labels, factor)
