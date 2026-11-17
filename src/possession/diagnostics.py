"""Ground-truth-free diagnostics: what the output claims about the match.

These are the numbers hyper-parameters are judged by. None of them needs a label: a football
match has a known order of magnitude of possessions and stoppages, and an output that reports
900 five-second possessions is wrong whatever its per-frame agreement with anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from possession.config import Config
from possession.consistency import (
    boundary_event_alignment,
    set_piece_consistency,
    transition_consistency,
)
from possession.dead_ball import BALL_CLASS_NAMES, DEAD_BALL, LIVE
from possession.pipeline import PipelineResult
from possession.runs import label_runs
from possession.structures import (
    ABSTAIN,
    AWAY,
    DEAD,
    HOME,
    LABEL_NAMES,
    FloatArray,
    IntArray,
    LabelArray,
    TaggedPhase,
)


def fill_abstentions(labels: LabelArray) -> LabelArray:
    """Forward-fill X with the previous label (back-filling a leading run), like the scorer."""
    known = labels != ABSTAIN
    if not known.any():
        return labels.copy()
    carry = np.maximum.accumulate(np.where(known, np.arange(labels.size), -1))
    carry[carry < 0] = int(np.flatnonzero(known)[0])
    return np.asarray(labels[carry], dtype=np.int8)


def segment_summary(labels: LabelArray, fps: float) -> dict[str, Any]:
    """Summarise counts and durations of possessions and stoppages in a label sequence."""
    starts, stops, values = label_runs(labels)
    seconds = (stops - starts) / fps
    possession = (values == HOME) | (values == AWAY)
    dead = values == DEAD
    alive_frames = int(np.sum((labels == HOME) | (labels == AWAY)))

    def spread(durations: Any) -> dict[str, float]:
        if durations.size == 0:
            return {"n": 0}
        low, median, high = np.percentile(durations, [10, 50, 90])
        return {
            "n": int(durations.size),
            "mean_s": round(float(durations.mean()), 2),
            "p10_s": round(float(low), 2),
            "median_s": round(float(median), 2),
            "p90_s": round(float(high), 2),
        }

    return {
        "possessions": spread(seconds[possession]),
        "home_possessions": int(np.sum(values == HOME)),
        "away_possessions": int(np.sum(values == AWAY)),
        "stoppages": spread(seconds[dead]),
        "home_share_of_alive": round(float(np.sum(labels == HOME)) / alive_frames, 4)
        if alive_frames
        else None,
        "label_share": {
            name: round(float(np.mean(labels == code)), 4) for code, name in enumerate(LABEL_NAMES)
        },
    }


def shift_summary(shifts: IntArray, fps: float) -> dict[str, Any]:
    """How many boundaries a refinement moved and by how much (negative = earlier)."""
    if shifts.size == 0:
        return {"n": 0}
    low, median, high = np.percentile(shifts / fps, [10, 50, 90])
    return {
        "n": int(shifts.size),
        "p10_s": round(float(low), 2),
        "median_s": round(float(median), 2),
        "p90_s": round(float(high), 2),
        "median_abs_s": round(float(np.median(np.abs(shifts)) / fps), 2),
    }


def vote_agreement(labels: LabelArray, votes: FloatArray) -> float | None:
    """Share of voted live frames on which the label is the team the tagging votes for."""
    voted = (labels != DEAD) & (votes != 0)
    if not voted.any():
        return None
    return round(float(np.mean((votes[voted] > 0) == (labels[voted] == HOME))), 4)


def independent_checks(
    result: PipelineResult, phases: list[TaggedPhase], config: Config, fps: float
) -> dict[str, Any]:
    """Measures that no boundary refinement can improve by construction.

    Set-piece tags are never read by the decoder or by any refinement, and the tagging votes
    used here are compared with the tracking-only pass, which has not seen them either.
    """
    window = np.zeros(result.decoded.size, dtype=bool)
    window[result.structure.first_kickoff_frame : result.structure.last_tagged_frame] = True
    return {
        "set_pieces": set_piece_consistency(
            result.decoded, phases, fps, config.tagging, config.report
        ),
        "transitions": transition_consistency(
            result.decoded, phases, fps, config.tagging, config.report
        ),
        "tag_vote_agreement_tracking_only": vote_agreement(
            result.first_pass[window], result.votes[window]
        ),
        "tag_vote_agreement_final": vote_agreement(result.decoded[window], result.votes[window]),
    }


def diagnostics_report(
    result: PipelineResult, phases: list[TaggedPhase], config: Config, fps: float
) -> dict[str, Any]:
    """Assemble the report logged and saved next to every submission."""
    match_start = result.structure.first_kickoff_frame
    match_stop = min(result.structure.last_tagged_frame, result.decoded.size)
    in_window = np.zeros(result.decoded.size, dtype=bool)
    in_window[match_start:match_stop] = True
    decoded = result.decoded[in_window]
    tracked = result.ball.observed[in_window]
    voted = result.votes[in_window]
    alive = decoded != DEAD
    vote_frames = alive & (voted != 0)
    report: dict[str, Any] = {
        "window_frames": [int(match_start), int(match_stop)],
        "independent_checks": independent_checks(result, phases, config, fps),
        "sanity_checks_circular": boundary_event_alignment(
            result.decoded, result.events, config.report
        ),
        "final": segment_summary(decoded, fps),
        "first_pass": segment_summary(result.first_pass[in_window], fps),
        "abstained_frames_in_window": int(np.sum(result.labels[in_window] == ABSTAIN)),
        "ball_tracked_share": {
            "alive": round(float(tracked[alive].mean()), 4) if alive.any() else None,
            "dead": round(float(tracked[~alive].mean()), 4) if (~alive).any() else None,
        },
        "tagging": {
            "calibration_frames": result.tagging.n_frames,
            "agreement_with_first_pass": round(result.tagging.agreement, 4),
            "log_odds_per_frame": round(result.tagging.log_odds_per_frame, 4),
            "final_agreement_on_voted_alive_frames": round(
                float(np.mean((voted[vote_frames] > 0) == (decoded[vote_frames] == HOME))), 4
            )
            if vote_frames.any()
            else None,
        },
        "liveness": {
            "ball_class_given_live": dict(
                zip(
                    BALL_CLASS_NAMES,
                    np.round(result.liveness.ball_table[LIVE], 4).tolist(),
                    strict=True,
                )
            ),
            "ball_class_given_dead": dict(
                zip(
                    BALL_CLASS_NAMES,
                    np.round(result.liveness.ball_table[DEAD_BALL], 4).tolist(),
                    strict=True,
                )
            ),
            "activity_mean_mps_live_dead": np.round(result.liveness.activity_mean, 3).tolist(),
            "activity_std_mps_live_dead": np.round(result.liveness.activity_std, 3).tolist(),
            "transitions": np.round(result.liveness.transitions, 6).tolist(),
            "log_likelihood_first_last": [
                round(result.liveness.log_likelihood_trace[0], 1),
                round(result.liveness.log_likelihood_trace[-1], 1),
            ]
            if result.liveness.log_likelihood_trace
            else None,
        },
        "shape_model": None
        if result.shape_model is None
        else {
            "training_frames": result.shape_model.n_training_frames,
            "training_accuracy": round(result.shape_model.training_accuracy, 4),
            "weights": np.round(result.shape_model.weights, 4).tolist(),
        },
        "dead_edge_refinement": None
        if result.edge_refinement is None
        else {
            "starts_moved": shift_summary(result.edge_refinement.start_shifts, fps),
            "ends_moved": shift_summary(result.edge_refinement.end_shifts, fps),
            "frames_changed": int(np.sum(result.viterbi != result.decoded)),
        },
        "touch_snap": None
        if result.touch_snap is None
        else shift_summary(result.touch_snap.shifts, fps),
        "first_vs_final_pass_agreement": round(
            float(np.mean(result.first_pass[in_window] == decoded)), 4
        ),
    }
    return report
