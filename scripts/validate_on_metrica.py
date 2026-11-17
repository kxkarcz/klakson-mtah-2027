"""Score the pipeline on a public match of another game with the official scorer.

    python scripts/validate_on_metrica.py --config config/default.yaml \
        --data-dir ../external-data/metrica-sample-game-3 \
        --official-repo ../more-than-a-hack-2027-main

The match is one of Metrica Sports' sample games (https://github.com/metrica-sports/sample-data);
none is in this repository. ``--data-dir`` holds either Sample Game 3 (metadata XML, raw
tracking text, event JSON) or Sample Game 1 or 2 (event CSV and one tracking CSV per team); the
layout is recognised from the file names. Reference labels are derived from the event data, the
tracking is degraded to resemble automatic tracking, the pipeline is run without Smart Tagging
(the samples have none), and the official ``possession_metrics.evaluate`` gives spellF1 and
macroF1.

Each configuration is scored on four views: clean or degraded ball track, and two readings of
what a bare recovery is worth. A change is worth adopting only if it helps on all of them.
A folder with a SkillCorner open match (tracking JSON lines, match JSON, phases of play) is
recognised as well, with SkillCorner's per-frame possession as the labels.
A folder with a DFL open match (match-information and position XML) is recognised too; its
labels are the provider's per-frame ball status and possession, so there is a single reading.
Sample Games 1 and 2 have no ``CARRY`` events, so their "controlled" reading also accepts a
lost ball or a ball played out as proof of control (on Sample Game 3, where carries exist, that
substitute reproduces the carry-based labels far better than passes and shots alone).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from possession.config import Config
from possession.io.config_file import load_config
from possession.pipeline import run_pipeline
from possession.structures import ABSTAIN, DEAD, LabelArray, Tracking
from possession.validation.degrade import DegradationConfig, degrade_tracking
from possession.validation.dfl import build_dfl_tracking, parse_match_information
from possession.validation.metrica import (
    CONTROLLED_ACTIONS,
    CONTROLLED_ACTIONS_WITHOUT_CARRIES,
    MatchEvent,
    build_metrica_tracking,
    home_team_id,
    labels_from_events,
    parse_events,
)
from possession.validation.metrica_csv import build_metrica_csv_tracking, parse_events_csv
from possession.validation.skillcorner import build_skillcorner_tracking

LOGGER = logging.getLogger("validate")
# The samples have no speed channel; speeds are derived the way a tracker would report them.
SPEED_SMOOTHING_FRAMES = 5
MAX_PLAYER_SPEED_MPS = 12.0
# The CSV samples do not state the pitch size; the provider's documentation gives 105 x 68 m,
# which is also what the metadata of Sample Game 3 says.
CSV_PITCH_LENGTH_M = 105.0
CSV_PITCH_WIDTH_M = 68.0
METADATA_SUFFIX = "_metadata.xml"
CSV_EVENTS_SUFFIX = "_RawEventsData.csv"
CARRY = "CARRY"
DFL_INFORMATION_PATTERN = "DFL_02_01_matchinformation_*.xml"
DFL_POSITIONS_PATTERN = "DFL_04_03_positions_raw_observed_*.xml"
SKILLCORNER_TRACKING_SUFFIX = "_tracking_extrapolated.jsonl"
# The challenge runs at 25 fps; matches recorded at another rate are resampled to it.
TARGET_FPS = 25.0
# Empty frames laid between the halves of a DFL match (each half is numbered on its own):
# one minute at 25 fps, long enough for the pipeline to see a break.
DFL_BREAK_FRAMES = 1500
# A harsher damage profile, for clean tracks that never lose the ball. The provided tracking
# has no ball on 41% of graded frames (DATA.md); losing the ball during half of the stoppages
# and on 30% of live frames brings these matches to about that share.
SPARSE_LIVE_GAP_SHARE = 0.30
SPARSE_STOPPAGE_LOSS_PROBABILITY = 0.5


def only_file(data_dir: Path, suffix: str) -> Path | None:
    """Return the single file in ``data_dir`` matching the glob ``suffix``, or None if absent."""
    found = sorted(data_dir.glob(suffix))
    if len(found) > 1:
        raise ValueError(f"{data_dir} holds more than one {suffix}; use one folder per match")
    return found[0] if found else None


def metrica_references(events: list[MatchEvent], n_frames: int) -> dict[str, LabelArray]:
    """Both readings of the event-based labels (see ``labels_from_events``)."""
    has_carries = any(event.kind == CARRY for event in events)
    return {
        "controlled": labels_from_events(
            events,
            n_frames,
            controlled_only=True,
            controlled_actions=(
                CONTROLLED_ACTIONS if has_carries else CONTROLLED_ACTIONS_WITHOUT_CARRIES
            ),
        ),
        "any-recovery": labels_from_events(events, n_frames, controlled_only=False),
    }


def load_match(data_dir: Path) -> tuple[Tracking, dict[str, LabelArray]]:
    """Read the tracking and the reference labels of the one match stored in ``data_dir``."""
    skillcorner_path = only_file(data_dir, f"*{SKILLCORNER_TRACKING_SUFFIX}")
    if skillcorner_path is not None:
        stem = skillcorner_path.name.removesuffix(SKILLCORNER_TRACKING_SUFFIX)
        with skillcorner_path.open(encoding="utf-8") as tracking_file:
            tracking, labels = build_skillcorner_tracking(
                tracking_file,
                (data_dir / f"{stem}_match.json").read_text(encoding="utf-8"),
                (data_dir / f"{stem}_phases_of_play.csv").read_text(encoding="utf-8"),
                target_fps=TARGET_FPS,
                speed_smoothing_frames=SPEED_SMOOTHING_FRAMES,
                max_speed_mps=MAX_PLAYER_SPEED_MPS,
            )
        return tracking, {"provider": labels}
    information_path = only_file(data_dir, DFL_INFORMATION_PATTERN)
    if information_path is not None:
        positions_path = only_file(data_dir, DFL_POSITIONS_PATTERN)
        if positions_path is None:
            raise FileNotFoundError(f"no DFL position file in {data_dir}")
        tracking, labels = build_dfl_tracking(
            str(positions_path),
            parse_match_information(information_path.read_text(encoding="utf-8")),
            break_frames=DFL_BREAK_FRAMES,
            speed_smoothing_frames=SPEED_SMOOTHING_FRAMES,
            max_speed_mps=MAX_PLAYER_SPEED_MPS,
        )
        return tracking, {"provider": labels}
    metadata_path = only_file(data_dir, f"*{METADATA_SUFFIX}")
    if metadata_path is not None:
        stem = metadata_path.name.removesuffix(METADATA_SUFFIX)
        metadata = metadata_path.read_text(encoding="utf-8")
        tracking = build_metrica_tracking(
            metadata,
            (data_dir / f"{stem}_tracking.txt").read_text(encoding="utf-8"),
            SPEED_SMOOTHING_FRAMES,
            MAX_PLAYER_SPEED_MPS,
        )
        events = parse_events(
            (data_dir / f"{stem}_events.json").read_text(encoding="utf-8"), home_team_id(metadata)
        )
        return tracking, metrica_references(events, tracking.n_frames)
    events_path = only_file(data_dir, f"*{CSV_EVENTS_SUFFIX}")
    if events_path is None:
        raise FileNotFoundError(f"no validation match found in {data_dir}")
    stem = events_path.name.removesuffix(CSV_EVENTS_SUFFIX)
    tracking = build_metrica_csv_tracking(
        (data_dir / f"{stem}_RawTrackingData_Home_Team.csv").read_text(encoding="utf-8"),
        (data_dir / f"{stem}_RawTrackingData_Away_Team.csv").read_text(encoding="utf-8"),
        pitch_length_m=CSV_PITCH_LENGTH_M,
        pitch_width_m=CSV_PITCH_WIDTH_M,
        speed_smoothing_frames=SPEED_SMOOTHING_FRAMES,
        max_speed_mps=MAX_PLAYER_SPEED_MPS,
    )
    events = parse_events_csv(events_path.read_text(encoding="utf-8"))
    return tracking, metrica_references(events, tracking.n_frames)


def with_value(config: Config, assignment: str) -> Config:
    """Apply one ``section.field=value`` assignment (value parsed as a YAML scalar)."""
    path, _, raw = assignment.partition("=")
    section, _, field = path.partition(".")
    current = getattr(config, section)
    target = type(getattr(current, field))
    value = yaml.safe_load(raw)
    changes: dict[str, Any] = {field: value if target is bool else target(value)}
    if field == "dead_enter_cost":
        changes["dead_exit_cost"] = changes[field]
    return dataclasses.replace(config, **{section: dataclasses.replace(current, **changes)})


def score(metrics: Any, reference: LabelArray, predicted: LabelArray) -> str:
    """spellF1 / macroF1 / possession counts from the official scorer."""
    result = metrics.evaluate(reference, predicted)
    return (
        f"spell {result['spell_f1_mean']:.3f} macro {result['macro_f1']:.3f}"
        f" n {result['n_pred_spells']:3d}/{result['n_gt_spells']:3d}"
    )


def main() -> None:
    """Run the requested configurations and log one line per configuration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.FIELD=VALUE")
    parser.add_argument(
        "--sweep",
        nargs="*",
        default=[],
        metavar="SECTION.FIELD=V1,V2",
        help="parameters to vary one at a time around the (overridden) configuration",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="also score a harsher damage profile with the ball missing on about 40%% of frames",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("possession").setLevel(logging.WARNING)

    sys.path.insert(0, str(args.official_repo / "evaluation"))
    metrics = importlib.import_module("possession_metrics")

    clean, references = load_match(args.data_dir)
    stoppages = next(iter(references.values()))  # every reading has the same stoppages
    dead = stoppages == DEAD
    in_match = stoppages != ABSTAIN
    LOGGER.info(
        "reference: %d frames in play or stopped, dead share %.3f, "
        "ball missing agrees with dead on %.3f of them",
        int(in_match.sum()),
        dead[in_match].mean(),
        (np.isnan(clean.ball_xy[:, 0]) == dead)[in_match].mean(),
    )
    views: dict[str, Tracking] = {"clean": clean}
    for seed in args.seeds:
        views[f"degraded{seed}"] = degrade_tracking(clean, stoppages, DegradationConfig(seed=seed))
    for seed in args.seeds if args.sparse else []:
        views[f"sparse{seed}"] = degrade_tracking(
            clean,
            stoppages,
            DegradationConfig(
                seed=seed,
                live_gap_share=SPARSE_LIVE_GAP_SHARE,
                stoppage_loss_probability=SPARSE_STOPPAGE_LOSS_PROBABILITY,
            ),
        )
    for name, view in views.items():
        LOGGER.info(
            "%s: ball tracked on %.3f of match frames",
            name,
            np.isfinite(view.ball_xy[in_match, 0]).mean(),
        )

    base = load_config(args.config)
    for assignment in args.set:
        base = with_value(base, assignment)
    runs: list[tuple[str, Config]] = [("base", base)]
    for sweep in args.sweep:
        path, _, values = sweep.partition("=")
        runs += [
            (f"{path}={value}", with_value(base, f"{path}={value}")) for value in values.split(",")
        ]
    for tag, config in runs:
        cells = []
        damaged: dict[tuple[str, str], list[float]] = {}
        for view_name, view in views.items():
            predicted = run_pipeline(view, [], config).labels
            for reference_name, reference in references.items():
                if view_name == "clean":
                    cells.append(f"clean/{reference_name} {score(metrics, reference, predicted)}")
                else:
                    kind = view_name.rstrip("0123456789")
                    result = metrics.evaluate(reference, predicted)
                    damaged.setdefault((kind, reference_name), []).append(result["spell_f1_mean"])
        means = " ".join(
            f"{kind}/{name} spell {np.mean(values):.3f} (min {np.min(values):.3f})"
            for (kind, name), values in damaged.items()
        )
        LOGGER.info("%-44s %s | %s", tag, " | ".join(cells), means)


if __name__ == "__main__":
    main()
