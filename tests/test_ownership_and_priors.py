from __future__ import annotations

import dataclasses

import numpy as np

from possession.abstention import apply_abstention, in_play_mask, outside_match_mask
from possession.features.geometry import BallProximity
from possession.io.config_file import load_config
from possession.ownership import control_log_odds, ownership_log_emissions
from possession.structures import ABSTAIN, AWAY, DEAD, HOME, TaggedPhase
from possession.tagging_prior import calibrate_votes, implied_team, phase_votes

from tests_paths import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)


def odds(home: float, away: float, speed: float = 2.0) -> float:
    proximity = BallProximity(home=np.array([home]), away=np.array([away]))
    return float(control_log_odds(proximity, np.array([speed]), CONFIG.ownership)[0])


def test_closer_team_is_favoured() -> None:
    assert odds(0.5, 5.0) > 0 > odds(5.0, 0.5)
    assert odds(0.5, 5.0) == -odds(5.0, 0.5)


def test_duel_loose_ball_and_missing_ball_say_nothing() -> None:
    assert odds(0.8, 1.0) == 0.0  # both within playing distance of each other
    assert odds(9.0, 20.0) == 0.0  # nobody can reach it
    assert odds(np.nan, np.nan) == 0.0  # ball not tracked


def test_fast_ball_counts_less_than_a_carried_one() -> None:
    assert 0 < odds(0.5, 5.0, speed=20.0) < odds(0.5, 5.0, speed=2.0)


def test_single_tracked_team_does_not_produce_nan() -> None:
    assert odds(0.5, np.inf) > 0


def test_ownership_emissions_never_reward_being_alive() -> None:
    emissions = ownership_log_emissions(np.array([2.0, -3.0, 0.0]))
    assert emissions.max() == 0.0
    np.testing.assert_allclose(emissions, [[0.0, -2.0], [-3.0, 0.0], [0.0, 0.0]])


def phase(code: str, team: int | None, start: int, end: int) -> TaggedPhase:
    return TaggedPhase(code, start, end, team, None)


def test_defending_phase_implies_the_opponent() -> None:
    assert implied_team(phase("BUILD UP", HOME, 0, 10), CONFIG.tagging) == HOME
    assert implied_team(phase("DEFENDING IN MIDDLE THIRD", HOME, 0, 10), CONFIG.tagging) == AWAY
    assert implied_team(phase("SET PIECES", HOME, 0, 10), CONFIG.tagging) is None
    assert implied_team(phase("BUILD UP", None, 0, 10), CONFIG.tagging) is None


def test_votes_are_trimmed_and_conflicts_cancel() -> None:
    tagging = dataclasses.replace(CONFIG.tagging, edge_trim_frames=10)
    phases = [
        phase("BUILD UP", HOME, 0, 100),
        phase("PROGRESSION", AWAY, 50, 150),
        phase("SET PIECES", None, 150, 200),
    ]
    votes = phase_votes(phases, 200, tagging)
    assert votes[:10].tolist() == [0.0] * 10  # trimmed edge
    assert set(votes[10:60]) == {1.0}
    assert set(votes[60:90]) == {0.0}  # home and away phases overlap
    assert set(votes[90:140]) == {-1.0}
    assert set(votes[140:]) == {0.0}


def test_vote_weight_is_the_measured_agreement() -> None:
    tagging = dataclasses.replace(
        CONFIG.tagging, min_calibration_frames=10, temperature=1.0, max_llr_per_frame=10.0
    )
    votes = np.ones(100)
    labels = np.array([HOME] * 80 + [AWAY] * 20, dtype=np.int8)
    calibration = calibrate_votes(votes, labels, np.ones(100, dtype=bool), tagging)
    assert calibration.agreement == 0.8
    np.testing.assert_allclose(calibration.log_odds_per_frame, np.log(4.0))


def test_uninformative_or_scarce_votes_get_zero_weight() -> None:
    tagging = dataclasses.replace(CONFIG.tagging, min_calibration_frames=10)
    labels = np.array([HOME, AWAY] * 50, dtype=np.int8)
    assert (
        calibrate_votes(np.ones(100), labels, np.ones(100, bool), tagging).log_odds_per_frame == 0
    )
    few = np.zeros(100, dtype=bool)
    few[:5] = True
    assert calibrate_votes(np.ones(100), labels, few, tagging).log_odds_per_frame == 0


def test_outside_match_abstention_keeps_a_trailing_stoppage() -> None:
    labels = np.array([DEAD] * 10 + [HOME] * 20 + [DEAD] * 30, dtype=np.int8)
    mask = outside_match_mask(labels, first_kickoff_frame=8, last_tagged_frame=25, margin_frames=5)
    assert mask[:8].all() and not mask[8:35].any() and mask[35:].all()


def test_in_play_abstention_is_strictly_interior() -> None:
    labels = np.array([DEAD] * 10 + [HOME] * 100 + [AWAY] * 40, dtype=np.int8)
    ball_known = np.ones(150, dtype=bool)
    ball_known[5:60] = False  # starts in the stoppage, runs deep into the possession
    ball_known[100:125] = False  # straddles the change of possession
    mask = in_play_mask(labels, ball_known, min_gap_frames=30, guard_frames=10)
    assert mask[20:60].all()
    assert not mask[:20].any() and not mask[60:].any()


def test_apply_abstention_respects_switches() -> None:
    labels = np.array([DEAD] * 10 + [HOME] * 100 + [DEAD] * 10, dtype=np.int8)
    ball_known = np.ones(120, dtype=bool)
    ball_known[30:90] = False
    off = dataclasses.replace(CONFIG.abstention, outside_match=False, in_play_enabled=False)
    assert apply_abstention(labels, ball_known, 0, 120, off).tolist() == labels.tolist()
    on = dataclasses.replace(
        off, in_play_enabled=True, in_play_min_gap_frames=30, in_play_boundary_guard_frames=25
    )
    out = apply_abstention(labels, ball_known, 0, 120, on)
    assert set(out[35:85]) == {ABSTAIN}
    assert ABSTAIN not in set(out[:35]) | set(out[85:])
