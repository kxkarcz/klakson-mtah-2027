"""Run-length helpers: the pipeline reasons about intervals, not frames."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from possession.structures import BoolArray, FloatArray, IntArray


def mask_runs(mask: BoolArray) -> tuple[IntArray, IntArray]:
    """Half-open ``[start, stop)`` bounds of every maximal run of True."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def label_runs(labels: NDArray[np.integer]) -> tuple[IntArray, IntArray, IntArray]:
    """Return ``(starts, stops, values)`` of every maximal constant run of a sequence."""
    if labels.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [labels.size]))
    return starts, stops, labels[starts].astype(np.int64)


def run_length_per_frame(mask: BoolArray) -> IntArray:
    """For every frame inside a True run, the length of that run; 0 elsewhere.

    Offline we know how long a ball gap will last, and a twelve-second disappearance means
    something very different from a half-second occlusion, so the length is a feature.
    """
    starts, stops = mask_runs(mask)
    out = np.zeros(mask.size, dtype=np.int64)
    lengths = stops - starts
    out[mask] = np.repeat(lengths, lengths)
    return out


def centred_mean(values: FloatArray, window: int) -> FloatArray:
    """NaN-aware moving average over a centred window of ``window`` frames.

    Frames whose whole window is missing stay NaN, so smoothing never invents an observation.
    """
    if window <= 1:
        return values.copy()
    finite = np.isfinite(values)
    cumulative = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(values.size)
    low = np.maximum(0, index - window // 2)
    high = np.minimum(values.size, index + (window - window // 2))
    total = cumulative[high] - cumulative[low]
    n = counts[high] - counts[low]
    out = np.full(values.size, np.nan)
    np.divide(total, n, out=out, where=n > 0)
    return out
