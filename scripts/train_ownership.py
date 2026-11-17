"""Train the supervised ownership model on other matches and score it on held-out ones.

    python scripts/train_ownership.py --config config/default.yaml \
        --train-dirs ../external-data/dfl-J03WMX ../external-data/dfl-J03WN1 ... \
        --test-dirs ../external-data/metrica-sample-game-3 \
        --official-repo ../more-than-a-hack-2027-main --leave-one-out --write-config

The training matches must carry per-frame possession labels from their tracking provider (the
open DFL matches do; see TECHNICAL.md, section 4.2). None of them is in this repository, and
the match to be labelled is never read here.

``--leave-one-out`` refits the model without each training match in turn and scores that
match with the official scorer, with and without the model: the only honest estimate of what
the model is worth. ``--test-dirs`` are scored with the model fitted on all training matches;
they are never trained on. ``--write-config`` stores the final weights in the configuration
file, which is all that inference needs.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from possession.config import Config, ownership_feature_names
from possession.features.ball_events import detect_ball_events
from possession.io.config_file import load_config
from possession.learned_ownership import fit_weights, ownership_features
from possession.pipeline import run_pipeline
from possession.preprocessing.ball import clean_ball
from possession.preprocessing.teams import assign_slots_to_teams
from possession.structures import AWAY, HOME, FloatArray, LabelArray, Tracking
from possession.validation.degrade import (
    DegradationConfig,
    degrade_tracking,
    provided_tracker_profile,
)

sys.path.insert(0, str(Path(__file__).parent))
from validate_on_metrica import load_match

LOGGER = logging.getLogger("train")
# The model must work on a ball track that is smoothed and has gaps, so every training match
# is used twice: as published and with the damage profile of validation/degrade.py.
TRAIN_DAMAGE_SEED = 1
# Held-out matches are scored on a damaged copy drawn with a different seed.
TEST_DAMAGE_SEED = 0
# Neighbouring frames are nearly identical; every third one keeps the information and makes
# the fit three times cheaper.
ROW_STRIDE = 3
# Ridge penalty on standardised weights. With several hundred thousand rows and thirty
# features it only has to keep collinear features (the touch memories) from drifting apart.
RIDGE_PENALTY = 10.0
# Newton steps; the fit converges in under ten.
FIT_ITERATIONS = 30
TRACKING_FIELDS = ("player_xy", "player_speed", "slot_team", "ball_xy")
WEIGHTS_LINE = re.compile(r"^(  weights: )\[[^\]]*\]$", re.MULTILINE)


def cached_match(data_dir: Path, cache_dir: Path) -> tuple[Tracking, LabelArray]:
    """Return parsed tracking and reference labels of one match, cached as ``.npz``.

    Matches with two readings of the labels use the one that changes at every recovery.
    """
    path = cache_dir / f"{data_dir.name}.npz"
    if path.exists():
        with np.load(path) as stored:
            tracking = Tracking(
                **{name: stored[name] for name in TRACKING_FIELDS},
                fps=float(stored["fps"]),
                pitch_length=float(stored["pitch_length"]),
                pitch_width=float(stored["pitch_width"]),
            )
            return tracking, stored["labels"]
    tracking, references = load_match(data_dir)
    labels = references.get("provider", references.get("any-recovery"))
    if labels is None:
        raise ValueError(f"{data_dir} has no usable reference labels")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{name: getattr(tracking, name) for name in TRACKING_FIELDS},
        fps=tracking.fps,
        pitch_length=tracking.pitch_length,
        pitch_width=tracking.pitch_width,
        labels=labels,
    )
    return tracking, labels


def match_features(tracking: Tracking, config: Config) -> FloatArray:
    """Feature matrix of one match, computed exactly as the pipeline does."""
    ball = clean_ball(tracking, config.ball)
    events = detect_ball_events(ball, tracking.fps, config.events, config.ball)
    return ownership_features(
        tracking.player_xy,
        assign_slots_to_teams(tracking),
        ball,
        events.kink,
        config=config.learned_ownership,
        touch_config=config.touch_evidence,
    )


# Misplaced-ball share of the second test view (the far tail of ball-to-player distances).
WRONG_BALL_SHARE = 0.1
DAMAGE_PROFILES = ("standard", "provided")


def damage_config(profile: str, seed: int, wrong_ball_share: float = 0.0) -> DegradationConfig:
    """Damage of one named profile: the generic one, or the one fitted to the provided tracker."""
    if profile == "provided":
        return provided_tracker_profile(seed, wrong_ball_share=wrong_ball_share)
    return DegradationConfig(seed=seed)


def damaged(tracking: Tracking, labels: LabelArray, config: DegradationConfig) -> Tracking:
    """Return a copy of the tracking with the ball track damaged as configured."""
    return degrade_tracking(tracking, labels, config)


def training_rows(
    data_dir: Path, cache_dir: Path, config: Config, profile: str
) -> tuple[FloatArray, FloatArray]:
    """Features and targets (1 = home) of the live frames of one match, clean and damaged."""
    tracking, labels = cached_match(data_dir, cache_dir)
    rows = np.flatnonzero((labels == HOME) | (labels == AWAY))[::ROW_STRIDE]
    target = (labels[rows] == HOME).astype(np.float64)
    views = (tracking, damaged(tracking, labels, damage_config(profile, TRAIN_DAMAGE_SEED)))
    features = np.concatenate([match_features(view, config)[rows] for view in views])
    return features, np.concatenate([target] * len(views))


def with_weights(config: Config, weights: FloatArray) -> Config:
    """Return the configuration with the learned model switched on and given these weights."""
    learned = dataclasses.replace(
        config.learned_ownership, enabled=True, weights=tuple(float(w) for w in weights)
    )
    return dataclasses.replace(config, learned_ownership=learned)


def without_model(config: Config) -> Config:
    """Return the configuration with the learned model switched off (hand-set evidence)."""
    learned = dataclasses.replace(config.learned_ownership, enabled=False)
    return dataclasses.replace(config, learned_ownership=learned)


@dataclasses.dataclass(frozen=True)
class Setup:
    """What every worker needs besides the match itself."""

    cache_dir: Path
    official_repo: Path
    config: Config
    profile: str


def score_match(data_dir: Path, weights: FloatArray, setup: Setup) -> str:
    """spellF1 of one match without and with the model, on the clean and damaged tracks."""
    cache_dir, official_repo, config, profile = (
        setup.cache_dir,
        setup.official_repo,
        setup.config,
        setup.profile,
    )
    sys.path.insert(0, str(official_repo / "evaluation"))
    metrics: Any = importlib.import_module("possession_metrics")
    logging.getLogger("possession").setLevel(logging.WARNING)
    tracking, labels = cached_match(data_dir, cache_dir)
    cells = []
    views = {
        "clean": tracking,
        "damaged": damaged(tracking, labels, damage_config(profile, TEST_DAMAGE_SEED)),
    }
    if profile == "provided":
        views["misplaced"] = damaged(
            tracking, labels, damage_config(profile, TEST_DAMAGE_SEED, WRONG_BALL_SHARE)
        )
    for view_name, view in views.items():
        scores = []
        for variant in (without_model(config), with_weights(config, weights)):
            result = metrics.evaluate(labels, run_pipeline(view, [], variant).labels)
            scores.append(f"{result['spell_f1_mean']:.3f} (n {result['n_pred_spells']})")
        cells.append(f"{view_name} hand-set {scores[0]} learned {scores[1]}")
    return f"{data_dir.name:24s} " + " | ".join(cells)


def fold_weights(parts: list[tuple[FloatArray, FloatArray]], held: int) -> FloatArray:
    """Weights fitted on every match except ``held``."""
    kept = [part for index, part in enumerate(parts) if index != held]
    return fit_weights(
        np.concatenate([features for features, _ in kept]),
        np.concatenate([target for _, target in kept]),
        RIDGE_PENALTY,
        FIT_ITERATIONS,
    )


def write_weights(
    config_path: Path, weights: FloatArray, section: str = "learned_ownership"
) -> None:
    """Replace the ``weights`` line of one section of the configuration file, nothing else."""
    text = config_path.read_text(encoding="utf-8")
    header = re.search(rf"^{re.escape(section)}:$", text, re.MULTILINE)
    if header is None:
        raise ValueError(f"{config_path} has no {section} section")
    following = re.search(r"^\S", text[header.end() :], re.MULTILINE)
    end = header.end() + (following.start() if following else len(text) - header.end())
    values = ", ".join(f"{weight:.6f}" for weight in weights)
    body, count = WEIGHTS_LINE.subn(
        lambda match: f"{match.group(1)}[{values}]", text[header.end() : end]
    )
    if count != 1:
        raise ValueError(f"{config_path} does not have exactly one {section}.weights line")
    config_path.write_text(text[: header.end()] + body + text[end:], encoding="utf-8")


def main() -> None:
    """Fit, optionally cross-validate and test, and optionally store the weights."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--test-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--official-repo", type=Path, help="needed for any scoring")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/validation"))
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="matches scored in parallel")
    parser.add_argument(
        "--damage",
        choices=DAMAGE_PROFILES,
        default="provided",
        help="ball-track damage used for training and scoring",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("possession").setLevel(logging.WARNING)
    config = load_config(args.config)
    scoring = args.leave_one_out or bool(args.test_dirs)
    if scoring and args.official_repo is None:
        parser.error("--official-repo is required with --leave-one-out or --test-dirs")

    for data_dir in (*args.train_dirs, *args.test_dirs):  # parse once, before any worker
        cached_match(data_dir, args.cache_dir)
    setup = Setup(args.cache_dir, args.official_repo, config, args.damage)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        parts = list(
            pool.map(
                training_rows,
                args.train_dirs,
                [args.cache_dir] * len(args.train_dirs),
                [config] * len(args.train_dirs),
                [args.damage] * len(args.train_dirs),
            )
        )
        if args.leave_one_out:
            LOGGER.info("held-out scores (model fitted without the match it is scored on):")
            folds = [
                pool.submit(score_match, data_dir, fold_weights(parts, held), setup)
                for held, data_dir in enumerate(args.train_dirs)
            ]
            for fold in folds:
                LOGGER.info("  %s", fold.result())
        features = np.concatenate([part[0] for part in parts])
        target = np.concatenate([part[1] for part in parts])
        weights = fit_weights(features, target, RIDGE_PENALTY, FIT_ITERATIONS)
        accuracy = float(np.mean(((features @ weights) > 0) == (target > 0.5)))
        LOGGER.info(
            "final fit: %d matches, %d rows, training frame accuracy %.3f",
            len(args.train_dirs),
            target.size,
            accuracy,
        )
        for name, weight in zip(
            ownership_feature_names(config.learned_ownership), weights, strict=True
        ):
            LOGGER.info("  %-36s %9.4f", name, weight)

        if args.test_dirs:
            LOGGER.info("test matches (never trained on), final weights:")
            tests = [
                pool.submit(score_match, data_dir, weights, setup) for data_dir in args.test_dirs
            ]
            for test in tests:
                LOGGER.info("  %s", test.result())

    if args.write_config:
        write_weights(args.config, weights)
        LOGGER.info("weights written to %s", args.config)


if __name__ == "__main__":
    main()
