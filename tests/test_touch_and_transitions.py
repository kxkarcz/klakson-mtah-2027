from __future__ import annotations

import dataclasses

import numpy as np

from possession.consistency import transition_consistency
from possession.decoding import decode_possession, transition_log_scores, viterbi
from possession.features.geometry import BallProximity
from possession.io.config_file import load_config
from possession.ownership import ownership_log_emissions, touch_log_odds
from possession.refinement.touch_snap import snap_switches_to_touches, touches_by_team
from possession.structures import AWAY, DEAD, HOME, TaggedPhase
from possession.tagging_prior import transition_starts, transition_switch_discount

from synthetic import FPS
from tests_paths import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)
SNAP = dataclasses.replace(CONFIG.touch_snap, enabled=True)

# H 0..200 | A 200..400 | D 400..500 | H 500..700
LABELS = np.array([HOME] * 200 + [AWAY] * 200 + [DEAD] * 100 + [HOME] * 200, dtype=np.int8)


def proximity(n: int, home_near: list[int], away_near: list[int]) -> BallProximity:
    home, away = np.full(n, 20.0), np.full(n, 20.0)
    home[home_near] = 0.5
    away[away_near] = 0.5
    return BallProximity(home=home, away=away)


def test_kinks_are_attributed_to_the_nearer_team_within_reach() -> None:
    kinks = np.array([10, 20, 30, 40])
    near = proximity(50, home_near=[10, 30], away_near=[20, 30])
    home_touches, away_touches = touches_by_team(kinks, near, SNAP.touch_radius_m)
    assert home_touches.tolist() == [10]  # 30 is contested, 40 is next to nobody
    assert away_touches.tolist() == [20]


def test_change_moves_to_the_touch_of_the_team_taking_over() -> None:
    kinks = np.array([150, 180, 210])
    near = proximity(700, home_near=[150, 210], away_near=[180])
    result = snap_switches_to_touches(LABELS, kinks, near, SNAP)
    assert result.shifts.tolist() == [-20]
    assert result.labels.tolist() == ([HOME] * 180 + [AWAY] * 220 + [DEAD] * 100 + [HOME] * 200)


def test_touches_of_the_losing_team_and_far_touches_are_ignored() -> None:
    near = proximity(700, home_near=[190], away_near=[100])
    result = snap_switches_to_touches(LABELS, np.array([100, 190]), near, SNAP)
    assert result.shifts.size == 0
    assert np.array_equal(result.labels, LABELS)


def test_restarts_and_stoppage_starts_are_not_touched() -> None:
    near = proximity(700, home_near=[490, 510], away_near=[390, 410])
    result = snap_switches_to_touches(LABELS, np.array([390, 410, 490, 510]), near, SNAP)
    assert np.array_equal(result.labels, LABELS)


def test_change_cannot_erase_a_possession() -> None:
    labels = np.array([HOME] * 30 + [AWAY] * 300, dtype=np.int8)
    near = proximity(330, home_near=[], away_near=[10])
    result = snap_switches_to_touches(labels, np.array([10]), near, SNAP)
    assert np.array_equal(result.labels, labels)


def phase(code: str, team: int | None, start: int) -> TaggedPhase:
    return TaggedPhase(code, start, start + 300, team, None)


PHASES = [
    phase("ATTACKING TRANSITION", AWAY, 200),
    phase("DEFENSIVE TRANSITION", HOME, 200),  # the same turnover, tagged from the other side
    phase("ATTACKING TRANSITION", HOME, 520),
    phase("BUILD UP", HOME, 50),
    phase("ATTACKING TRANSITION", None, 600),
]


def test_transition_starts_are_deduplicated_and_name_the_gaining_team() -> None:
    assert transition_starts(PHASES, CONFIG.tagging) == [(200, AWAY), (520, HOME)]


def test_transition_consistency_measures_offsets_to_possession_starts() -> None:
    summary = transition_consistency(LABELS, PHASES, FPS, CONFIG.tagging, CONFIG.report)
    assert summary["n_transitions"] == 2
    assert summary["median_offset_s"] == -0.4  # offsets 0 and -20 frames
    assert summary["within_tight"] == 1.0


def test_discount_covers_the_window_around_each_transition() -> None:
    decoding = dataclasses.replace(
        CONFIG.decoding,
        transition_discount=2.0,
        transition_window_before_frames=10,
        transition_window_after_frames=20,
    )
    discount = transition_switch_discount(PHASES, 700, CONFIG.tagging, decoding)
    assert set(discount[190:221]) == {2.0} and set(discount[510:541]) == {2.0}
    assert discount.sum() == 2.0 * 62


def test_discount_lets_weak_evidence_change_possession_only_where_tagged() -> None:
    # 14 frames of away evidence at 0.4 nats: 5.6 nats, short of the 6-nat switch cost.
    emissions = np.zeros((300, 3))
    emissions[:, DEAD] = -5.0
    emissions[:100, AWAY] = -0.4
    emissions[100:114, HOME] = -0.4
    emissions[114:, AWAY] = -0.01
    config = dataclasses.replace(CONFIG.decoding, min_possession_frames=1, switch_cost=6.0)
    plain = decode_possession(emissions, config, initial_state=HOME)
    assert set(plain[100:114]) == {HOME}
    discount = np.zeros(300)
    discount[90:130] = 3.0
    eased = decode_possession(emissions, config, initial_state=HOME, switch_discount=discount)
    assert set(eased[100:114]) == {AWAY}
    elsewhere = np.zeros(300)
    elsewhere[200:240] = 3.0
    assert np.array_equal(
        decode_possession(emissions, config, initial_state=HOME, switch_discount=elsewhere), plain
    )


def test_discount_does_not_change_stoppage_costs() -> None:
    truth = [HOME] * 50 + [DEAD] * 50 + [HOME] * 50
    emissions = np.full((150, 3), -1.0)
    emissions[np.arange(150), truth] = 0.0
    initial = np.array([0.0, -np.inf, -np.inf])
    scores = transition_log_scores(CONFIG.decoding)
    assert np.array_equal(
        viterbi(emissions, scores, initial), viterbi(emissions, scores, initial, np.full(150, 1.0))
    )


TOUCH = dataclasses.replace(CONFIG.touch_evidence, enabled=True)


def test_touch_is_credited_to_the_team_at_the_ball_and_only_on_the_touch_frame() -> None:
    near = proximity(100, home_near=[10, 30], away_near=[30, 50])
    near.home[70], near.away[70] = 3.0, 9.0  # beyond the touch radius: nobody played it
    near.home[80] = np.nan  # ball position unknown
    log_odds = touch_log_odds(np.array([10, 30, 50, 70, 80]), near, TOUCH)
    assert log_odds[10] == TOUCH.llr_per_touch
    assert log_odds[50] == -TOUCH.llr_per_touch
    assert log_odds[30] == 0.0  # both sides at the ball: a duel
    assert log_odds[70] == 0.0 and log_odds[80] == 0.0
    assert np.count_nonzero(log_odds) == 2


def test_duel_margin_silences_a_touch_with_the_opponent_almost_as_close() -> None:
    near = proximity(20, home_near=[5], away_near=[])
    near.away[5] = 0.8
    assert touch_log_odds(np.array([5]), near, TOUCH)[5] == TOUCH.llr_per_touch
    strict = dataclasses.replace(TOUCH, duel_margin_m=0.5)
    assert touch_log_odds(np.array([5]), near, strict)[5] == 0.0


def decode_touches(away_touches: list[int], home_again_from: int | None) -> np.ndarray:
    """Home holds the ball for 4 s, then away touches it; home may take control back."""
    log_odds = np.zeros(200)
    log_odds[:100] = CONFIG.ownership.control_llr_per_frame
    log_odds[away_touches] = -TOUCH.llr_per_touch
    if home_again_from is not None:
        log_odds[home_again_from:] = CONFIG.ownership.control_llr_per_frame
    emissions = np.zeros((200, 3))
    emissions[:, :2] = ownership_log_emissions(log_odds)
    emissions[:, DEAD] = -50.0
    return decode_possession(emissions, CONFIG.decoding, initial_state=HOME)


def test_a_deflection_does_not_change_possession_but_two_touches_do() -> None:
    # One away touch, then home in control again: two switches (4 nats at a switch cost of 2)
    # for 3 nats of evidence.
    assert set(decode_touches([120], home_again_from=140)) == {HOME}
    # Two away touches a second apart repay the switch, dated at the first touch.
    two = decode_touches([120, 145], home_again_from=None)
    assert set(two[:120]) == {HOME} and set(two[120:]) == {AWAY}
