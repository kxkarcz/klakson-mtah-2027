"""Where the submission answers X instead of a label.

The ranking metric forward-fills every X with the previous label of the same half, so an X can
never beat simply repeating that label; it only says, honestly, "this frame was not observed".
It is therefore written in two places only: outside the match, where nothing is graded, and
(optionally) deep inside a possession during a long loss of the ball - never near a change of
state, where the fill would drag the old label across the hand-over.
"""

from __future__ import annotations

import numpy as np

from possession.config import AbstentionConfig
from possession.runs import label_runs, run_length_per_frame
from possession.structures import ABSTAIN, DEAD, BoolArray, LabelArray


def outside_match_mask(
    labels: LabelArray, first_kickoff_frame: int, last_tagged_frame: int, margin_frames: int
) -> BoolArray:
    """Frames before the first kick-off, and well after the last sign of play.

    The trailing region starts ``margin_frames`` after both the last decoded live frame and
    the last tagged phase, so a final stoppage keeps its D labels and the fill has a D to
    repeat.
    """
    mask = np.zeros(labels.size, dtype=bool)
    mask[:first_kickoff_frame] = True
    alive = np.flatnonzero(labels != DEAD)
    last_alive = int(alive[-1]) + 1 if alive.size else 0
    mask[max(last_alive, last_tagged_frame) + margin_frames :] = True
    return mask


def in_play_mask(
    labels: LabelArray, ball_known: BoolArray, min_gap_frames: int, guard_frames: int
) -> BoolArray:
    """Frames inside a possession where the ball has been lost for a long stretch.

    A frame qualifies only if it is at least ``guard_frames`` away from both ends of its
    possession, which keeps every abstention strictly interior.
    """
    long_gap = run_length_per_frame(~ball_known) >= min_gap_frames
    starts, stops, values = label_runs(labels)
    interior = np.zeros(labels.size, dtype=bool)
    for start, stop, value in zip(starts, stops, values, strict=True):
        if value != DEAD and stop - start > 2 * guard_frames:
            interior[start + guard_frames : stop - guard_frames] = True
    return long_gap & interior


def apply_abstention(
    labels: LabelArray,
    ball_known: BoolArray,
    first_kickoff_frame: int,
    last_tagged_frame: int,
    config: AbstentionConfig,
) -> LabelArray:
    """Return the labels with X written where the configuration allows it."""
    out = labels.copy()
    if config.in_play_enabled:
        out[
            in_play_mask(
                labels,
                ball_known,
                config.in_play_min_gap_frames,
                config.in_play_boundary_guard_frames,
            )
        ] = ABSTAIN
    if config.outside_match:
        out[
            outside_match_mask(
                labels, first_kickoff_frame, last_tagged_frame, config.post_match_margin_frames
            )
        ] = ABSTAIN
    return out
