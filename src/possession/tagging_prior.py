"""Smart Tagging as soft, self-calibrated evidence about the team in possession.

The tagged intervals are tactical phases, padded and automatic; they are never copied into
the output. Each phase casts a vote for the team it implies, and the weight of that vote is
measured on this very match by asking how often the votes agree with ownership the tracking
shows unambiguously. A tagger that is right 70% of the time gets the log-odds of 70%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.config import DecodingConfig, TaggingConfig
from possession.structures import AWAY, HOME, BoolArray, FloatArray, LabelArray, TaggedPhase


def implied_team(phase: TaggedPhase, config: TaggingConfig) -> int | None:
    """Team a phase says is in possession, or None when it says nothing usable.

    Attacking phases name the team with the ball; defending phases name the team without it.
    Phases without a team (every throw-in, both kick-offs) and set pieces are deliberately
    silent here: a set piece is a restart, and who restarts is exactly what the provider left
    blank on the restarts where it matters.
    """
    if phase.team is None:
        return None
    if phase.code in config.possession_codes:
        return phase.team
    if phase.code in config.opponent_codes:
        return AWAY if phase.team == HOME else HOME
    return None


def phase_votes(phases: list[TaggedPhase], n_frames: int, config: TaggingConfig) -> FloatArray:
    """Per-frame vote in {-1, 0, +1}: +1 when the phases covering the frame imply HOME.

    Phase edges are trimmed because the intervals are padded around the action, and
    overlapping phases that disagree cancel to zero instead of being arbitrated.
    """
    balance = np.zeros(n_frames + 1)
    for phase in phases:
        team = implied_team(phase, config)
        start = max(0, phase.start_frame + config.edge_trim_frames)
        stop = min(n_frames, phase.end_frame - config.edge_trim_frames)
        if team is None or stop <= start:
            continue
        sign = 1.0 if team == HOME else -1.0
        balance[start] += sign
        balance[stop] -= sign
    return np.asarray(np.sign(np.cumsum(balance[:-1])), dtype=np.float64)


@dataclass(frozen=True)
class TaggingCalibration:
    """How well the votes agreed with unambiguous tracking ownership."""

    n_frames: int
    agreement: float
    log_odds_per_frame: float


def calibrate_votes(
    votes: FloatArray, pseudo_labels: LabelArray, usable: BoolArray, config: TaggingConfig
) -> TaggingCalibration:
    """Measure vote reliability against algorithmic pseudo-labels.

    ``pseudo_labels`` come from a first decoding pass that never saw the tagging, restricted by
    ``usable`` to frames where the ball is tracked and a team is in possession. With too few
    such frames, or votes no better than chance, the tagging gets zero weight.
    """
    scored = usable & (votes != 0)
    n_scored = int(scored.sum())
    if n_scored < config.min_calibration_frames:
        return TaggingCalibration(n_scored, 0.5, 0.0)
    implied_home = votes[scored] > 0
    agreement = float(np.mean(implied_home == (pseudo_labels[scored] == HOME)))
    if agreement <= 0.5:
        return TaggingCalibration(n_scored, agreement, 0.0)
    clipped = min(agreement, 1.0 - 1.0 / n_scored)
    log_odds = config.temperature * float(np.log(clipped / (1.0 - clipped)))
    return TaggingCalibration(n_scored, agreement, min(log_odds, config.max_llr_per_frame))


def tagging_log_odds(votes: FloatArray, calibration: TaggingCalibration) -> FloatArray:
    """Per-frame home-vs-away log-odds contributed by the tagging."""
    return votes * calibration.log_odds_per_frame


def transition_starts(phases: list[TaggedPhase], config: TaggingConfig) -> list[tuple[int, int]]:
    """``(frame, team gaining the ball)`` for every tagged transition, without duplicates.

    The provider tags each turnover twice, as an attacking transition of one team and a
    defensive transition of the other; both describe the same instant and the same new owner.
    """
    found = {
        (phase.start_frame, team)
        for phase in phases
        if phase.code in config.transition_codes
        and (team := implied_team(phase, config)) is not None
    }
    return sorted(found)


def transition_switch_discount(
    phases: list[TaggedPhase], n_frames: int, tagging: TaggingConfig, decoding: DecodingConfig
) -> FloatArray:
    """Per-frame reduction of the switch cost around tagged transition starts.

    The tag says that possession changed about here, which is evidence for *a* change in the
    window, not for a label on any frame; so it lowers the price of changing and leaves the
    ball evidence to say exactly when, and whether at all.
    """
    discount = np.zeros(n_frames)
    for frame, _ in transition_starts(phases, tagging):
        start = max(0, frame - decoding.transition_window_before_frames)
        stop = min(n_frames, frame + decoding.transition_window_after_frames + 1)
        discount[start:stop] = decoding.transition_discount
    return discount
