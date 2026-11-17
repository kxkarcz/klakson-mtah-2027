"""Train the supervised live/dead model on other matches and score it on held-out ones.

    python scripts/train_liveness.py --config config/default.yaml \
        --train-dirs ../external-data/dfl-J03WMX ... ../external-data/skillcorner-1874553 ... \
        --test-dirs ../external-data/metrica-sample-game-3 \
        --official-repo ../more-than-a-hack-2027-main --leave-one-out --write-config

The training matches must carry per-frame ball status from their tracking provider: the open
DFL matches (``BallStatus``) or SkillCorner's open matches (phases of play). The submitted
weights are fitted on the seven DFL matches only, with SkillCorner and Metrica as tests:
SkillCorner's players behave unlike the provided tracking during stoppages (see
``LAYOUT_COLUMNS``). None of these matches is in this repository, and the match to be
labelled is never read here.

Each match is used on the ball track closest to the provided automatic tracking: SkillCorner
as published (it is automatic broadcast tracking); DFL, which tracks the ball through every
stoppage, damaged with the fitted profile and with the ball also lost on most stoppages;
Metrica, used only as a test, with the fitted profile. ``--leave-one-out`` refits without each
training match in turn and scores that match with and without the model.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from possession.config import Config
from possession.dead_ball import ball_observation_class
from possession.features.ball_events import detect_ball_events
from possession.features.collective import collective_activity, team_shape
from possession.io.config_file import load_config
from possession.learned_liveness import liveness_features
from possession.learned_ownership import fit_weights
from possession.pipeline import run_pipeline
from possession.preprocessing.ball import clean_ball
from possession.preprocessing.teams import assign_slots_to_teams
from possession.structures import ABSTAIN, DEAD, FloatArray, LabelArray, Tracking
from possession.validation.degrade import degrade_tracking, provided_tracker_profile

sys.path.insert(0, str(Path(__file__).parent))
from train_ownership import cached_match, write_weights

LOGGER = logging.getLogger("train")
# DFL tracks the ball through every stoppage; the provided tracking sees it on about a third
# of dead frames. Losing it on 65% of stoppages brings DFL to about that share.
DFL_STOPPAGE_LOSS = 0.65
DAMAGE_SEED = 0
# Neighbouring frames are nearly identical; every fourth keeps the information.
ROW_STRIDE = 4
# Ridge penalty on standardised weights; with over a million rows it only steadies collinear
# features (the overlapping windows).
RIDGE_PENALTY = 1.0
FIT_ITERATIONS = 25
# Feature columns left out of the fit (their weights stay zero): the player-layout columns
# (share sprinting, speed spread, team gap, spread along each axis, distance from halfway).
# On SkillCorner's broadcast tracking the players off the screen are extrapolated and bunch
# together during stoppages, which the provided tracking never shows; a model that leans on
# layout finds 88 stoppages on the match to be labelled instead of about 105.
LAYOUT_COLUMNS = (29, 30, 31, 32, 33, 34)


def view_of(data_dir: Path, tracking: Tracking, labels: LabelArray) -> Tracking:
    """Return the match on the ball track closest to the provided automatic tracking."""
    if data_dir.name.startswith("skillcorner-"):
        return tracking
    profile = provided_tracker_profile(DAMAGE_SEED)
    if data_dir.name.startswith("dfl-"):
        profile = dataclasses.replace(profile, stoppage_loss_probability=DFL_STOPPAGE_LOSS)
    return degrade_tracking(tracking, labels, profile)


def match_features(tracking: Tracking, config: Config) -> FloatArray:
    """Feature matrix of one match, computed exactly as the pipeline does."""
    masks = assign_slots_to_teams(tracking)
    ball = clean_ball(tracking, config.ball)
    no_lookahead = dataclasses.replace(config.activity, lookahead_enabled=False)
    minimum = config.collective.min_players_per_team
    return liveness_features(
        tracking,
        ball=ball,
        ball_class=ball_observation_class(ball, config.ball),
        activity_now=collective_activity(tracking, masks, no_lookahead),
        shapes=(
            team_shape(tracking.player_xy, masks.home, minimum),
            team_shape(tracking.player_xy, masks.away, minimum),
        ),
        events=detect_ball_events(ball, tracking.fps, config.events, config.ball),
        config=config.learned_liveness,
    )


def training_rows(data_dir: Path, cache_dir: Path, config: Config) -> tuple[FloatArray, FloatArray]:
    """Features and targets (1 = ball in play) of the frames inside the periods of one match."""
    tracking, labels = cached_match(data_dir, cache_dir)
    rows = np.flatnonzero(labels != ABSTAIN)[::ROW_STRIDE]
    features = match_features(view_of(data_dir, tracking, labels), config)
    return features[rows], (labels[rows] != DEAD).astype(np.float64)


def with_weights(config: Config, weights: FloatArray) -> Config:
    """Return the configuration with the learned model switched on and given these weights."""
    learned = dataclasses.replace(
        config.learned_liveness, enabled=True, weights=tuple(float(w) for w in weights)
    )
    return dataclasses.replace(config, learned_liveness=learned)


def without_model(config: Config) -> Config:
    """Return the configuration with the learned model switched off."""
    learned = dataclasses.replace(config.learned_liveness, enabled=False)
    return dataclasses.replace(config, learned_liveness=learned)


@dataclasses.dataclass(frozen=True)
class Setup:
    """What every worker needs besides the match itself."""

    cache_dir: Path
    official_repo: Path
    config: Config


def score_match(data_dir: Path, weights: FloatArray, setup: Setup) -> str:
    """spellF1 of one match without and with the model."""
    sys.path.insert(0, str(setup.official_repo / "evaluation"))
    metrics: Any = importlib.import_module("possession_metrics")
    logging.getLogger("possession").setLevel(logging.WARNING)
    tracking, labels = cached_match(data_dir, setup.cache_dir)
    view = view_of(data_dir, tracking, labels)
    cells = []
    for name, variant in (
        ("unsupervised", without_model(setup.config)),
        ("learned", with_weights(setup.config, weights)),
    ):
        result = metrics.evaluate(labels, run_pipeline(view, [], variant).labels)
        cells.append(f"{name} {result['spell_f1_mean']:.3f} (n {result['n_pred_spells']})")
    return f"{data_dir.name:24s} " + " | ".join(cells)


def fold_weights(parts: list[tuple[FloatArray, FloatArray]], held: int) -> FloatArray:
    """Weights fitted on every match except ``held``, zero on the excluded columns."""
    kept = [part for index, part in enumerate(parts) if index != held]
    n_columns = kept[0][0].shape[1]
    used = [column for column in range(n_columns) if column not in LAYOUT_COLUMNS]
    weights = np.zeros(n_columns)
    weights[used] = fit_weights(
        np.concatenate([features[:, used] for features, _ in kept]),
        np.concatenate([target for _, target in kept]),
        RIDGE_PENALTY,
        FIT_ITERATIONS,
    )
    return weights


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
    parser.add_argument("--jobs", type=int, default=1, help="matches processed in parallel")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("possession").setLevel(logging.WARNING)
    config = load_config(args.config)
    if (args.leave_one_out or args.test_dirs) and args.official_repo is None:
        parser.error("--official-repo is required with --leave-one-out or --test-dirs")

    for data_dir in (*args.train_dirs, *args.test_dirs):  # parse once, before any worker
        cached_match(data_dir, args.cache_dir)
    setup = Setup(args.cache_dir, args.official_repo, config)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        parts = list(
            pool.map(
                training_rows,
                args.train_dirs,
                [args.cache_dir] * len(args.train_dirs),
                [config] * len(args.train_dirs),
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
        weights = fold_weights(parts, -1)
        features = np.concatenate([part[0] for part in parts])
        target = np.concatenate([part[1] for part in parts])
        accuracy = float(np.mean(((features @ weights) > 0) == (target > 0.5)))
        LOGGER.info(
            "final fit: %d matches, %d rows, training frame accuracy %.3f",
            len(args.train_dirs),
            target.size,
            accuracy,
        )
        if args.test_dirs:
            LOGGER.info("test matches (never trained on), final weights:")
            tests = [
                pool.submit(score_match, data_dir, weights, setup) for data_dir in args.test_dirs
            ]
            for test in tests:
                LOGGER.info("  %s", test.result())
    if args.write_config:
        write_weights(args.config, weights, section="learned_liveness")
        LOGGER.info("weights written to %s", args.config)


if __name__ == "__main__":
    main()
