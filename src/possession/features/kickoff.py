"""Match structure: where a half may start, and the kick-off line-up."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.config import KickoffConfig, TaggingConfig
from possession.preprocessing.teams import TeamMasks
from possession.runs import centred_mean, mask_runs
from possession.structures import BoolArray, FloatArray, TaggedPhase, Tracking


@dataclass(frozen=True)
class MatchStructure:
    """What is known about the match timeline before any possession is inferred."""

    forced_dead: BoolArray  # frames on which the ball cannot be in play
    in_match: BoolArray  # frames used to fit the unsupervised models
    first_kickoff_frame: int  # earliest frame the match can be live; 0 if unknown
    last_tagged_frame: int  # end of the last tagged phase; n_frames if no tagging
    second_half_start: int  # first frame attributed to the second half


def kickoff_formation(tracking: Tracking, masks: TeamMasks, config: KickoffConfig) -> BoolArray:
    """Frames on which each team stands entirely in its own half.

    The laws of the game require exactly this line-up at every kick-off and open play never
    produces it for long, so a sustained run of it is a dead ball waiting for a restart: after
    a goal it marks the stoppage end far more sharply than the celebrations around it.
    """
    halfway = tracking.pitch_length / 2.0
    x = tracking.player_xy[:, :, 0]
    left = np.nan_to_num(x, nan=np.inf) < halfway + config.halfway_tolerance_m
    right = np.nan_to_num(x, nan=-np.inf) > halfway - config.halfway_tolerance_m
    n_home, n_away = masks.home.sum(axis=1), masks.away.sum(axis=1)
    home_left = (masks.home & left).sum(axis=1) == n_home
    home_right = (masks.home & right).sum(axis=1) == n_home
    away_left = (masks.away & left).sum(axis=1) == n_away
    away_right = (masks.away & right).sum(axis=1) == n_away
    enough = (n_home >= config.min_players_per_team) & (n_away >= config.min_players_per_team)
    lined_up = enough & ((home_left & away_right) | (home_right & away_left))
    starts, stops = mask_runs(lined_up)
    sustained = np.zeros(tracking.n_frames, dtype=bool)
    for start, stop in zip(starts, stops, strict=True):
        if stop - start >= config.min_formation_frames:
            sustained[start:stop] = True
    return sustained


def tracked_player_count(masks: TeamMasks, smoothing_frames: int) -> FloatArray:
    """Smoothed number of tracked players; it collapses when the teams leave the pitch."""
    count = (masks.home.sum(axis=1) + masks.away.sum(axis=1)).astype(np.float64)
    return centred_mean(count, smoothing_frames)


def match_structure(
    tracking: Tracking,
    masks: TeamMasks,
    phases: list[TaggedPhase],
    config: KickoffConfig,
    tagging: TaggingConfig,
) -> MatchStructure:
    """Derive forced-dead frames and the model-fitting window from tags and tracking.

    The two "kick off (start)" tags anchor the halves. The interval between them is split at
    the moment the fewest players are tracked (the teams are in the dressing rooms): from there
    until the second kick-off the ball is dead by definition, including the minutes in which
    the players are back on the pitch warming up with a visible ball. Nothing here is a
    constant of this match; a recording with different timings yields different frames.
    """
    n_frames = tracking.n_frames
    forced = kickoff_formation(tracking, masks, config)
    half_starts = sorted(
        phase.start_frame
        for phase in phases
        if phase.code == tagging.set_piece_code
        and phase.set_piece_type == tagging.kickoff_start_type
    )
    last_tagged = max((phase.end_frame for phase in phases), default=n_frames)
    in_match = np.ones(n_frames, dtype=bool)
    in_match[min(last_tagged, n_frames) :] = False
    if len(half_starts) != 2:
        # Without both anchors nothing can be asserted about the halves; decode unconstrained.
        return MatchStructure(forced, in_match & ~forced, 0, last_tagged, n_frames)
    first, second = (max(0, start - config.tag_margin_frames) for start in half_starts)
    count = tracked_player_count(masks, config.halftime_smoothing_frames)
    interval_end = max(first + 1, second)
    break_frame = first + int(np.nanargmin(np.nan_to_num(count[first:interval_end], nan=np.inf)))
    forced[:first] = True
    forced[break_frame:second] = True
    first_half_tags = [phase.end_frame for phase in phases if phase.end_frame <= break_frame]
    in_match[max(first_half_tags, default=break_frame) : second] = False
    return MatchStructure(forced, in_match & ~forced, first, last_tagged, second)
