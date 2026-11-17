"""Snap the edges of decoded stoppages to the ball event that caused them.

The decoder decides *that* a stoppage happened from seconds of smoothed evidence; its edges
inherit that blur. The instant itself is usually visible in the ball track, and the ranking
metric rewards getting it right: a second of error at both ends of a ten-second possession
costs a fifth of its IoU.

Rules, in order of trust (timings measured on a public match of another game, Metrica Sample
Game 3, whose events are hand-timed):

* a stoppage **starts** when the ball crosses a boundary line (median error 0.08 s there);
* a stoppage **ends** when a resting ball is struck, or when a lost ball comes back into view
  already moving. That match cannot test this rule (its ball is simply absent during
  stoppages), and on the provided match it did not survive the check against set-piece tags,
  so its windows are zero in the default configuration;
* with no such event in the window, the common deceleration of the players can locate a
  whistle, but only to about +-1.5 s, so that rule is optional and off by default.

An edge with no event in its window keeps the decoder's position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.config import DeadEdgeConfig
from possession.features.ball_events import BallEvents
from possession.runs import label_runs
from possession.structures import DEAD, BoolArray, FloatArray, IntArray, LabelArray


@dataclass(frozen=True)
class EdgeRefinement:
    """Refined labels and what was moved."""

    labels: LabelArray
    start_shifts: IntArray  # frames each moved stoppage start was shifted by (negative = earlier)
    end_shifts: IntArray  # the same for stoppage ends


def nearest_in_window(events: IntArray, frame: int, before: int, after: int) -> int | None:
    """Event closest to ``frame`` inside ``[frame - before, frame + after]``, else None.

    Ties go to the earlier event, so the result does not depend on platform or ordering.
    """
    low = int(np.searchsorted(events, frame - before, side="left"))
    high = int(np.searchsorted(events, frame + after, side="right"))
    if high <= low:
        return None
    window = events[low:high]
    return int(window[int(np.argmin(np.abs(window - frame)))])


def strongest_deceleration(
    activity: FloatArray, frame: int, before: int, after: int, lag: int
) -> int | None:
    """Frame in the window at which collective speed drops the most across ``2 * lag`` frames."""
    low, high = max(lag, frame - before), min(activity.size - lag, frame + after + 1)
    if high <= low:
        return None
    frames = np.arange(low, high)
    drop = activity[frames - lag] - activity[frames + lag]
    if not np.isfinite(drop).any() or np.nanmax(drop) <= 0:
        return None
    return int(frames[int(np.nanargmax(drop))])


def refine_dead_ball_edges(
    labels: LabelArray,
    events: BallEvents,
    restart_events: IntArray,
    *,
    activity: FloatArray,
    forced_dead: BoolArray,
    config: DeadEdgeConfig,
) -> EdgeRefinement:
    """Move stoppage edges to nearby ball events.

    Each stoppage is treated on its own and may only grow into, or give way to, its two
    neighbours: an edge never crosses another edge, never shortens a neighbour or the stoppage
    itself below ``min_remaining_frames``, and never makes a forced-dead frame live.
    """
    out = labels.copy()
    starts, stops, values = label_runs(labels)
    start_shifts: list[int] = []
    end_shifts: list[int] = []
    for run in np.flatnonzero(values == DEAD):
        start, stop = int(starts[run]), int(stops[run])
        new_start, new_stop = start, stop
        if run > 0:
            target = nearest_in_window(
                events.out_of_play,
                start,
                config.start_window_before_frames,
                config.start_window_after_frames,
            )
            if target is None and config.use_deceleration:
                target = strongest_deceleration(
                    activity,
                    start,
                    config.start_window_before_frames,
                    config.start_window_after_frames,
                    config.deceleration_lag_frames,
                )
            earliest = int(starts[run - 1]) + config.min_remaining_frames
            if target is not None and earliest <= target <= stop - config.min_remaining_frames:
                new_start = target
        if run < values.size - 1:
            target = nearest_in_window(
                restart_events,
                stop,
                config.end_window_before_frames,
                config.end_window_after_frames,
            )
            latest = int(stops[run + 1]) - config.min_remaining_frames
            if (
                target is not None
                and new_start + config.min_remaining_frames <= target <= latest
                and not forced_dead[target:stop].any()
            ):
                new_stop = target
        if new_start != start:
            out[min(start, new_start) : max(start, new_start)] = (
                DEAD if new_start < start else values[run - 1]
            )
            start_shifts.append(new_start - start)
        if new_stop != stop:
            out[min(stop, new_stop) : max(stop, new_stop)] = (
                values[run + 1] if new_stop < stop else DEAD
            )
            end_shifts.append(new_stop - stop)
    return EdgeRefinement(
        labels=out,
        start_shifts=np.array(start_shifts, dtype=np.int64),
        end_shifts=np.array(end_shifts, dtype=np.int64),
    )
