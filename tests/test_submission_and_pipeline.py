from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from possession.config import ConfigError, config_fingerprint, config_from_mapping
from possession.diagnostics import fill_abstentions, segment_summary
from possession.io.config_file import load_config
from possession.pipeline import run_pipeline
from possession.runs import centred_mean, label_runs, mask_runs, run_length_per_frame
from possession.structures import ABSTAIN, AWAY, DEAD, HOME, TaggedPhase
from possession.submission import format_submission, write_submission

from synthetic import FPS, grid_team, make_tracking
from tests_paths import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)


def test_submission_format_is_exact() -> None:
    labels = np.array([ABSTAIN, HOME, AWAY, DEAD], dtype=np.int8)
    assert format_submission(labels) == "frame,label\n0,X\n1,H\n2,A\n3,D\n"


def test_submission_rejects_unknown_codes() -> None:
    with pytest.raises(ValueError, match="alphabet"):
        format_submission(np.array([0, 7], dtype=np.int8))


def test_written_bytes_are_platform_independent(tmp_path: Path) -> None:
    labels = np.array([HOME, DEAD], dtype=np.int8)
    digest = write_submission(tmp_path / "nested" / "out.csv", labels)
    payload = (tmp_path / "nested" / "out.csv").read_bytes()
    assert payload == b"frame,label\n0,H\n1,D\n"
    assert digest == hashlib.sha256(payload).hexdigest()


def test_default_config_loads_and_rejects_typos() -> None:
    mapping = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config_from_mapping(mapping) == CONFIG
    mapping["decoding"]["swich_cost"] = 1.0
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_mapping(mapping)


def test_config_rejects_wrong_types_and_meaningless_values() -> None:
    mapping = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    mapping["decoding"]["min_dead_frames"] = 2.5
    with pytest.raises(ConfigError, match="integer"):
        config_from_mapping(mapping)
    mapping["decoding"]["min_dead_frames"] = 0
    with pytest.raises(ConfigError, match="invalid configuration"):
        config_from_mapping(mapping)


def test_fingerprint_tracks_only_the_named_sections() -> None:
    mapping = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    mapping["decoding"]["switch_cost"] = 99.0
    changed = config_from_mapping(mapping)
    assert config_fingerprint(changed) != config_fingerprint(CONFIG)
    assert config_fingerprint(changed, "data") == config_fingerprint(CONFIG, "data")


def test_run_helpers() -> None:
    mask = np.array([0, 1, 1, 0, 1, 0, 0], dtype=bool)
    starts, stops = mask_runs(mask)
    assert starts.tolist() == [1, 4] and stops.tolist() == [3, 5]
    assert run_length_per_frame(mask).tolist() == [0, 2, 2, 0, 1, 0, 0]
    starts, stops, values = label_runs(np.array([2, 2, 0, 0, 0, 1]))
    assert (starts.tolist(), stops.tolist(), values.tolist()) == ([0, 2, 5], [2, 5, 6], [2, 0, 1])
    smoothed = centred_mean(np.array([np.nan, 1.0, 3.0, np.nan, np.nan, np.nan]), 3)
    np.testing.assert_allclose(smoothed[:4], [1.0, 2.0, 2.0, 3.0])
    assert np.isnan(smoothed[4:]).all()


def test_abstentions_are_filled_like_the_scorer() -> None:
    labels = np.array([ABSTAIN, HOME, ABSTAIN, ABSTAIN, AWAY, ABSTAIN], dtype=np.int8)
    assert fill_abstentions(labels).tolist() == [HOME, HOME, HOME, HOME, AWAY, AWAY]


def test_segment_summary_counts_possessions_not_frames() -> None:
    labels = np.array([HOME] * 50 + [DEAD] * 25 + [HOME] * 50 + [AWAY] * 25, dtype=np.int8)
    summary = segment_summary(labels, FPS)
    assert summary["possessions"]["n"] == 3
    assert summary["stoppages"]["n"] == 1
    assert summary["home_share_of_alive"] == 0.8


def synthetic_match() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kick-off, then alternating home and away possessions separated by stoppages."""
    # Overlapping blocks: a line-up split by the halfway line would read as a kick-off.
    home, away = grid_team(45.0), grid_team(55.0)
    blocks = [
        (DEAD, 400),
        (HOME, 500),
        (DEAD, 300),
        (AWAY, 500),
        (DEAD, 300),
        (HOME, 500),
        (AWAY, 400),
        (DEAD, 400),
    ]
    rng = np.random.default_rng(1)
    ball, speed, truth = [], [], []
    for state, length in blocks:
        truth.append(np.full(length, state))
        if state == DEAD:
            ball.append(np.full((length, 2), np.nan))
            speed.append(1.0 + 0.2 * rng.standard_normal(length))
            continue
        owner = home if state == HOME else away
        carrier = owner[np.arange(length) // 50 % owner.shape[0]]
        ball.append(
            carrier + np.column_stack((0.4 * np.sin(np.arange(length) / 4.0), np.zeros(length)))
        )
        speed.append(2.5 + 0.3 * rng.standard_normal(length))
    return home, away, np.concatenate(ball), np.concatenate(speed), np.concatenate(truth)


def test_pipeline_recovers_a_synthetic_match_and_is_deterministic() -> None:
    home, away, ball, speed, truth = synthetic_match()
    tracking = make_tracking(truth.size, home, away, ball, speed)
    phases = [
        TaggedPhase("BUILD UP", 500, 800, HOME, None),
        TaggedPhase("PROGRESSION", 2600, truth.size, AWAY, None),
    ]
    # Synthetic players change speed instantly: use present activity, not the look-ahead.
    config = dataclasses.replace(
        CONFIG, activity=dataclasses.replace(CONFIG.activity, lookahead_enabled=False)
    )
    first = run_pipeline(tracking, phases, config)
    second = run_pipeline(tracking, phases, config)
    assert np.array_equal(first.labels, second.labels)
    assert format_submission(first.labels) == format_submission(second.labels)
    agreement = np.mean(first.decoded == truth)
    assert agreement > 0.95
    assert segment_summary(first.decoded, FPS)["possessions"]["n"] == 4
