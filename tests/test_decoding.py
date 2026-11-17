from __future__ import annotations

import itertools

import numpy as np

from possession.decoding import (
    absorb_short_runs,
    forward_backward,
    transition_log_scores,
    viterbi,
)
from possession.io.config_file import load_config
from possession.structures import AWAY, DEAD, HOME

from tests_paths import DEFAULT_CONFIG


def emissions_for(sequence: list[int], strength: float = 1.0) -> np.ndarray:
    out = np.full((len(sequence), 3), -strength)
    out[np.arange(len(sequence)), sequence] = 0.0
    return out


def test_viterbi_follows_sustained_evidence() -> None:
    truth = [DEAD] * 20 + [HOME] * 40 + [AWAY] * 40 + [DEAD] * 20
    config = load_config(DEFAULT_CONFIG).decoding
    initial = np.array([-np.inf, -np.inf, 0.0])
    path = viterbi(emissions_for(truth), transition_log_scores(config), initial)
    assert path.tolist() == truth


def test_viterbi_ignores_a_brief_touch() -> None:
    # Three frames of away evidence inside a home possession cannot repay two switches.
    noisy = [HOME] * 30 + [AWAY] * 3 + [HOME] * 30
    config = load_config(DEFAULT_CONFIG).decoding
    initial = np.array([0.0, -np.inf, -np.inf])
    path = viterbi(emissions_for(noisy), transition_log_scores(config), initial)
    assert path.tolist() == [HOME] * 63


def test_viterbi_matches_brute_force() -> None:
    rng = np.random.default_rng(7)
    log_emissions = rng.normal(size=(6, 3))
    log_transitions = rng.normal(size=(3, 3))
    log_initial = rng.normal(size=3)

    def score(path: tuple[int, ...]) -> float:
        total = log_initial[path[0]] + log_emissions[0, path[0]]
        for frame in range(1, len(path)):
            total += (
                log_transitions[path[frame - 1], path[frame]] + log_emissions[frame, path[frame]]
            )
        return float(total)

    best = max(itertools.product(range(3), repeat=6), key=score)
    assert viterbi(log_emissions, log_transitions, log_initial).tolist() == list(best)


def test_forward_backward_matches_enumeration() -> None:
    rng = np.random.default_rng(3)
    log_emissions = rng.normal(size=(5, 2))
    transitions = np.array([[0.9, 0.1], [0.3, 0.7]])
    initial = np.array([0.6, 0.4])
    posterior = forward_backward(log_emissions, transitions, initial)

    weights = np.zeros((5, 2))
    total = 0.0
    for path in itertools.product(range(2), repeat=5):
        probability = initial[path[0]] * np.exp(log_emissions[0, path[0]])
        for frame in range(1, 5):
            probability *= transitions[path[frame - 1], path[frame]]
            probability *= np.exp(log_emissions[frame, path[frame]])
        total += probability
        for frame, state in enumerate(path):
            weights[frame, state] += probability
    np.testing.assert_allclose(posterior.state, weights / total, atol=1e-12)
    np.testing.assert_allclose(posterior.log_likelihood, np.log(total), atol=1e-12)
    np.testing.assert_allclose(posterior.transition_counts.sum(), 4.0, atol=1e-12)


def test_short_run_between_equal_neighbours_is_bridged() -> None:
    labels = np.array([HOME] * 10 + [DEAD] * 2 + [HOME] * 10, dtype=np.int8)
    out = absorb_short_runs(labels, np.array([3.0, 3.0, 5.0]))
    assert out.tolist() == [HOME] * 22


def test_short_run_joins_the_longer_neighbour() -> None:
    labels = np.array([HOME] * 10 + [AWAY] * 2 + [DEAD] * 30, dtype=np.int8)
    out = absorb_short_runs(labels, np.array([5.0, 5.0, 5.0]))
    assert out.tolist() == [HOME] * 10 + [DEAD] * 32


def test_runs_at_or_above_minimum_are_kept() -> None:
    labels = np.array([HOME] * 5 + [AWAY] * 5 + [DEAD] * 5, dtype=np.int8)
    assert absorb_short_runs(labels, np.array([5.0, 5.0, 5.0])).tolist() == labels.tolist()
