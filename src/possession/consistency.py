"""Label-free consistency checks: set-piece tags and ball events.

Set-piece tags are an automatic annotation produced independently of this model, and the
decoder never reads their timing or their team. That makes them a fair external yardstick for
choosing hyper-parameters without any ground truth: a good dead-ball detector ends a stoppage
close to where a set piece is tagged, and a good ownership model hands the ball to the team
the tag names.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from possession.config import ReportConfig, TaggingConfig
from possession.features.ball_events import BallEvents
from possession.runs import label_runs
from possession.structures import DEAD, FloatArray, IntArray, LabelArray, TaggedPhase
from possession.tagging_prior import transition_starts


def restart_frames(labels: LabelArray) -> tuple[Any, Any]:
    """Frames at which a stoppage ends, and the team that takes the ball there."""
    starts, _, values = label_runs(labels)
    after_dead = np.flatnonzero((values[:-1] == DEAD) & (values[1:] != DEAD)) + 1
    return starts[after_dead], values[after_dead]


def set_piece_consistency(
    labels: LabelArray,
    phases: list[TaggedPhase],
    fps: float,
    config: TaggingConfig,
    report: ReportConfig,
) -> dict[str, Any]:
    """How decoded restarts line up with tagged set pieces.

    For every tagged set piece the nearest decoded restart is found. ``offset`` is its time
    relative to the tag start; tags are windows opened a fixed lead before the restart, so
    accurate restarts give a narrow offset distribution, whatever its centre. ``spread_s`` is
    the interquartile range of the offsets, ``within_tight`` / ``within_loose`` the share of
    tags whose restart lies within the two report tolerances of the typical offset, and
    ``team_agreement`` the share of team-tagged set pieces after which the named team has the
    ball.
    """
    frames, teams = restart_frames(labels)
    set_pieces = [phase for phase in phases if phase.code == config.set_piece_code]
    if frames.size == 0 or not set_pieces:
        return {"n_set_pieces": len(set_pieces), "n_restarts": int(frames.size)}
    offsets, agreements = [], []
    for phase in set_pieces:
        nearest = int(np.argmin(np.abs(frames - phase.start_frame)))
        offsets.append(int(frames[nearest]) - phase.start_frame)
        if phase.team is not None:
            agreements.append(int(teams[nearest]) == phase.team)
    offset = np.array(offsets, dtype=np.float64)
    low, median, high = np.percentile(offset, [25, 50, 75])
    deviation = np.abs(offset - median)
    return {
        "n_set_pieces": len(set_pieces),
        "n_restarts": int(frames.size),
        "median_offset_s": round(float(median) / fps, 2),
        "spread_s": round(float(high - low) / fps, 2),
        "within_tight": round(
            float(np.mean(deviation <= report.restart_tight_tolerance_frames)), 4
        ),
        "within_loose": round(
            float(np.mean(deviation <= report.restart_loose_tolerance_frames)), 4
        ),
        "n_team_tagged": len(agreements),
        "team_agreement": round(float(np.mean(agreements)), 4) if agreements else None,
    }


def offset_spread(offsets: FloatArray, fps: float, report: ReportConfig) -> dict[str, Any]:
    """Centre and spread of boundary offsets relative to tag starts."""
    low, median, high = np.percentile(offsets, [25, 50, 75])
    deviation = np.abs(offsets - median)
    return {
        "median_offset_s": round(float(median) / fps, 2),
        "spread_s": round(float(high - low) / fps, 2),
        "within_tight": round(
            float(np.mean(deviation <= report.restart_tight_tolerance_frames)), 4
        ),
        "within_loose": round(
            float(np.mean(deviation <= report.restart_loose_tolerance_frames)), 4
        ),
    }


def transition_consistency(
    labels: LabelArray,
    phases: list[TaggedPhase],
    fps: float,
    config: TaggingConfig,
    report: ReportConfig,
) -> dict[str, Any]:
    """How decoded possession starts line up with tagged transitions.

    For every tagged transition the nearest start of a possession of the team that gains the
    ball is found. As with set pieces, well-timed changes give a narrow offset distribution.
    The measure is independent only while the transition discount of the decoder is off.
    """
    starts, _, values = label_runs(labels)
    transitions = transition_starts(phases, config)
    offsets = []
    for frame, team in transitions:
        own = starts[values == team]
        if own.size:
            offsets.append(float(own[int(np.argmin(np.abs(own - frame)))] - frame))
    if not offsets:
        return {"n_transitions": len(transitions)}
    return {
        "n_transitions": len(transitions),
        **offset_spread(np.array(offsets), fps, report),
    }


def stoppage_starts(labels: LabelArray) -> IntArray:
    """Frames at which a stoppage begins after live play."""
    starts, _, values = label_runs(labels)
    into_dead = np.flatnonzero((values[:-1] != DEAD) & (values[1:] == DEAD)) + 1
    return np.asarray(starts[into_dead], dtype=np.int64)


def possession_switches(labels: LabelArray) -> IntArray:
    """Frames at which possession passes directly from one team to the other."""
    starts, _, values = label_runs(labels)
    switch = np.flatnonzero((values[:-1] != DEAD) & (values[1:] != DEAD)) + 1
    return np.asarray(starts[switch], dtype=np.int64)


def distance_to_nearest(frames: IntArray, events: IntArray) -> FloatArray:
    """Absolute distance in frames from each frame to the nearest event; inf without events."""
    if events.size == 0:
        return np.full(frames.size, np.inf)
    right = np.clip(np.searchsorted(events, frames), 0, events.size - 1)
    left = np.clip(right - 1, 0, events.size - 1)
    return np.asarray(
        np.minimum(np.abs(events[right] - frames), np.abs(events[left] - frames)),
        dtype=np.float64,
    )


def share_near(frames: IntArray, events: IntArray, tolerance_frames: int) -> float | None:
    """Share of ``frames`` with an event within the tolerance; None when there are no frames."""
    if frames.size == 0:
        return None
    return round(float(np.mean(distance_to_nearest(frames, events) <= tolerance_frames)), 4)


def boundary_event_alignment(
    labels: LabelArray, events: BallEvents, report: ReportConfig
) -> dict[str, Any]:
    """How often decoded boundaries coincide with ball events.

    These shares are **not** evidence of accuracy once boundaries are snapped to events - they
    then rise by construction. They show that the snapping does what it says and how many
    boundaries have no event to snap to.
    """
    restarts, _ = restart_frames(labels)
    restart_events = np.union1d(events.restart_kick, events.reappearance)
    return {
        "stoppage_starts": int(stoppage_starts(labels).size),
        "stoppage_starts_at_ball_exit": share_near(
            stoppage_starts(labels), events.out_of_play, report.boundary_event_tolerance_frames
        ),
        "restarts": int(restarts.size),
        "restarts_at_kick_or_reappearance": share_near(
            np.asarray(restarts, dtype=np.int64),
            restart_events,
            report.boundary_event_tolerance_frames,
        ),
        "switches": int(possession_switches(labels).size),
        "switches_at_trajectory_kink": share_near(
            possession_switches(labels), events.kink, report.switch_kink_tolerance_frames
        ),
        "n_events": {
            "out_of_play": int(events.out_of_play.size),
            "restart_kick": int(events.restart_kick.size),
            "reappearance": int(events.reappearance.size),
            "kink": int(events.kink.size),
        },
    }
