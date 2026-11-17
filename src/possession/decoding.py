"""Hidden-Markov machinery: Viterbi decoding, forward-backward and minimum durations.

The ranking metric scores whole possessions, so the label sequence is decoded jointly rather
than frame by frame: a transition has to pay a cost that only sustained evidence can repay,
which is the probabilistic form of "a touch is not a change of possession".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.config import DecodingConfig
from possession.runs import label_runs
from possession.structures import AWAY, DEAD, HOME, N_STATES, FloatArray, LabelArray


def transition_log_scores(config: DecodingConfig) -> FloatArray:
    """Log-score matrix ``[from, to]`` over (HOME, AWAY, DEAD) built from the configured costs.

    Scores are costs rather than normalised probabilities; Viterbi only compares paths, so the
    missing normaliser is a constant that cannot change the arg-max.
    """
    scores = np.zeros((N_STATES, N_STATES))
    scores[HOME, AWAY] = scores[AWAY, HOME] = -config.switch_cost
    scores[HOME, DEAD] = scores[AWAY, DEAD] = -config.dead_enter_cost
    scores[DEAD, HOME] = scores[DEAD, AWAY] = -config.dead_exit_cost
    return scores


def viterbi(
    log_emissions: FloatArray,
    log_transitions: FloatArray,
    log_initial: FloatArray,
    switch_discount: FloatArray | None = None,
) -> LabelArray:
    """Return the most likely state path.

    ``switch_discount``, when given, holds one non-negative value per frame that is added to
    the HOME<->AWAY transition scores for transitions *into* that frame: it makes a change of
    possession cheaper where outside evidence says one is due, without touching the cost of
    stoppages. Ties resolve to the lowest state index, so the result is exact and reproducible
    on any platform.
    """
    n_frames, n_states = log_emissions.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n_frames, n_states), dtype=np.int8)
    score = log_initial + log_emissions[0]
    for frame in range(1, n_frames):
        candidates = score[:, None] + log_transitions
        if switch_discount is not None:
            candidates[HOME, AWAY] += switch_discount[frame]
            candidates[AWAY, HOME] += switch_discount[frame]
        best = candidates.argmax(axis=0)
        back[frame] = best
        score = candidates[best, np.arange(n_states)] + log_emissions[frame]
    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(score.argmax())
    for frame in range(n_frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


@dataclass(frozen=True)
class Posteriors:
    """Outputs of forward-backward needed by EM."""

    state: FloatArray  # (n_frames, n_states) posterior of each state
    transition_counts: FloatArray  # (n_states, n_states) expected transition counts
    log_likelihood: float


def forward_backward(
    log_emissions: FloatArray, transitions: FloatArray, initial: FloatArray
) -> Posteriors:
    """Scaled forward-backward for a discrete-state HMM with row-stochastic ``transitions``."""
    n_frames, n_states = log_emissions.shape
    # Per-frame rescaling keeps the recursion in range; the shift is undone in the likelihood.
    shift = log_emissions.max(axis=1)
    emissions = np.exp(log_emissions - shift[:, None])
    alpha = np.empty((n_frames, n_states))
    scale = np.empty(n_frames)
    current = initial * emissions[0]
    for frame in range(n_frames):
        if frame:
            current = (current @ transitions) * emissions[frame]
        scale[frame] = current.sum()
        current = current / scale[frame]
        alpha[frame] = current
    beta = np.ones(n_states)
    posterior = np.empty((n_frames, n_states))
    posterior[-1] = alpha[-1]
    counts = np.zeros((n_states, n_states))
    for frame in range(n_frames - 2, -1, -1):
        weighted = emissions[frame + 1] * beta / scale[frame + 1]
        counts += alpha[frame][:, None] * transitions * weighted[None, :]
        beta = transitions @ weighted
        posterior[frame] = alpha[frame] * beta
    posterior /= posterior.sum(axis=1, keepdims=True)
    return Posteriors(
        state=posterior,
        transition_counts=counts,
        log_likelihood=float(np.log(scale).sum() + shift.sum()),
    )


def absorb_short_runs(labels: LabelArray, min_frames: FloatArray) -> LabelArray:
    """Remove runs shorter than their state's minimum by merging them into a neighbour.

    A run between two runs of the same label rejoins them (a two-frame "dead ball" inside a
    possession is tracking noise, and leaving it would split one possession into two). Between
    different labels it joins the longer neighbour, which moves the boundary the least in
    relative terms. Shortest runs go first so that a cluster of flicker collapses from the
    inside out; ties resolve to the earliest run.
    """
    out = labels.copy()
    while True:
        starts, stops, values = label_runs(out)
        if starts.size <= 1:
            return out
        lengths = stops - starts
        short = np.flatnonzero(lengths < min_frames[values])
        if short.size == 0:
            return out
        run = int(short[np.argmin(lengths[short])])
        if run == 0:
            neighbour = 1
        elif run == starts.size - 1 or values[run - 1] == values[run + 1]:
            neighbour = run - 1
        else:
            neighbour = run - 1 if lengths[run - 1] >= lengths[run + 1] else run + 1
        out[starts[run] : stops[run]] = values[neighbour]


def minimum_run_frames(config: DecodingConfig) -> FloatArray:
    """Minimum run length per state, indexed by state code."""
    minimum = np.empty(N_STATES)
    minimum[HOME] = minimum[AWAY] = config.min_possession_frames
    minimum[DEAD] = config.min_dead_frames
    return minimum


def decode_possession(
    log_emissions: FloatArray,
    config: DecodingConfig,
    initial_state: int = DEAD,
    switch_discount: FloatArray | None = None,
) -> LabelArray:
    """Decode the joint H/A/D sequence and enforce the minimum durations.

    The recording starts long before kick-off, so the chain starts in the dead state.
    """
    log_initial = np.full(N_STATES, -np.inf)
    log_initial[initial_state] = 0.0
    path = viterbi(log_emissions, transition_log_scores(config), log_initial, switch_discount)
    return absorb_short_runs(path, minimum_run_frames(config))
