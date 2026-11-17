"""Label-free sensitivity analysis: how the output changes when one parameter moves.

    python scripts/sensitivity.py --config config/default.yaml

For every listed parameter the pipeline is re-run with the value scaled down and up, and the
ground-truth-free diagnostics are printed side by side: number and median length of
possessions and stoppages, dead-ball share, agreement of the tracking-only decode with Smart
Tagging team votes, and the alignment of decoded restarts with tagged set pieces. This is the
evidence behind the values in ``config/default.yaml``; it never reads a ground truth.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path

import numpy as np
import yaml

from possession.config import Config, config_fingerprint
from possession.consistency import boundary_event_alignment
from possession.diagnostics import independent_checks, segment_summary
from possession.io.atd import parse_team_names
from possession.io.cache import load_tracking_cached
from possession.io.config_file import load_config
from possession.io.smart_tagging import read_smart_tagging
from possession.pipeline import run_pipeline
from possession.structures import TaggedPhase, Tracking

LOGGER = logging.getLogger("sensitivity")

# (section, field, values). Values bracket the default on both sides.
SWEEPS: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    ("decoding", "switch_cost", (1.0, 2.0, 3.0, 4.0, 6.0, 9.0)),
    ("decoding", "dead_enter_cost", (6.0, 8.0, 10.0, 12.0, 16.0)),
    ("liveness", "emission_temperature", (0.1, 0.25, 0.5)),
    ("liveness", "probability_floor", (0.0001, 0.02, 0.05, 0.1)),
    ("liveness", "seed_activity_split_mps", (1.2, 1.5, 1.8)),
    ("ownership", "reach_radius_m", (3.0, 4.0, 6.0)),
    ("ownership", "ball_speed_scale_mps", (5.0, 8.0, 12.0)),
    ("ownership", "control_llr_per_frame", (0.3, 0.4, 0.5)),
    ("activity", "smoothing_frames", (5, 13, 25, 50)),
    ("activity", "lookahead_frames", (25, 50, 75, 100, 150)),
    ("touch_evidence", "llr_per_touch", (0.0, 1.0, 2.0, 3.0, 5.0)),
    ("touch_evidence", "touch_radius_m", (1.5, 2.5, 4.0)),
    ("learned_ownership", "llr_scale", (0.05, 0.1, 0.2, 0.35, 0.6)),
    ("touch_snap", "window_before_frames", (19, 38, 57)),
    ("touch_snap", "window_after_frames", (19, 38, 57)),
    ("touch_snap", "touch_radius_m", (1.5, 2.5, 4.0)),
    ("decoding", "transition_discount", (0.5, 1.0, 1.5, 2.0)),
    ("dead_edges", "start_window_before_frames", (38, 75, 112)),
    ("dead_edges", "end_window_before_frames", (38, 75, 112)),
    ("dead_edges", "end_window_after_frames", (38, 75, 112)),
)


def with_value(config: Config, section: str, field: str, value: object) -> Config:
    """Copy of the configuration with one field replaced (dead costs move together)."""
    current = getattr(config, section)
    target = type(getattr(current, field))
    changes: dict[str, object] = {field: value if target is bool else target(value)}
    if field == "dead_enter_cost":
        changes["dead_exit_cost"] = changes[field]
    return dataclasses.replace(config, **{section: dataclasses.replace(current, **changes)})


def with_overrides(config: Config, assignments: list[str]) -> Config:
    """Apply ``section.field=value`` assignments (values parsed as YAML scalars)."""
    for assignment in assignments:
        path, _, raw = assignment.partition("=")
        section, _, field = path.partition(".")
        config = with_value(config, section, field, yaml.safe_load(raw))
    return config


def describe(tracking: Tracking, phases: list[TaggedPhase], config: Config) -> str:
    """One line of label-free diagnostics for a configuration."""
    result = run_pipeline(tracking, phases, config)
    window = np.zeros(tracking.n_frames, dtype=bool)
    window[result.structure.first_kickoff_frame : result.structure.last_tagged_frame] = True
    final = segment_summary(result.decoded[window & ~result.structure.forced_dead], tracking.fps)
    checks = independent_checks(result, phases, config, tracking.fps)
    sanity = boundary_event_alignment(result.decoded, result.events, config.report)
    possessions, stoppages, tags = final["possessions"], final["stoppages"], checks["set_pieces"]
    turnovers = checks["transitions"]
    return (
        f"poss {possessions['n']:3d} med {possessions['median_s']:5.1f} s"
        f" | stop {stoppages['n']:3d} med {stoppages['median_s']:5.1f} s"
        f" | D {final['label_share']['D']:.3f}"
        f" | INDEPENDENT votes(track-only) {checks['tag_vote_agreement_tracking_only']}"
        f" votes(final) {checks['tag_vote_agreement_final']}"
        f" restart@tag 1s {tags.get('within_tight')} 3s {tags.get('within_loose')}"
        f" iqr {tags.get('spread_s')} s"
        f" change@transition 1s {turnovers.get('within_tight')} 3s {turnovers.get('within_loose')}"
        f" iqr {turnovers.get('spread_s')} s"
        f" | SANITY exit {sanity['stoppage_starts_at_ball_exit']}"
        f" kick {sanity['restarts_at_kick_or_reappearance']}"
        f" kink {sanity['switches_at_trajectory_kink']}"
    )


def main() -> None:
    """Run the sweeps and log one line per configuration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="restrict to these parameters, written section.field",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="overrides applied to the configuration before sweeping, e.g. to switch a stage on",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("possession").setLevel(logging.WARNING)

    config = with_overrides(load_config(args.config), args.set)
    data_dir = args.data_dir or Path(os.environ[config.data.data_dir_env])
    metadata_path = data_dir / config.data.tracking_metadata
    tracking = load_tracking_cached(
        metadata_path,
        data_dir / config.data.tracking_raw,
        args.cache_dir,
        config_fingerprint(config, "data"),
    )
    home_name, away_name = parse_team_names(metadata_path.read_text(encoding="utf-8"))
    phases = read_smart_tagging(
        data_dir / config.data.smart_tagging, tracking.fps, home_name, away_name
    )
    LOGGER.info("  default  %s", describe(tracking, phases, config))
    for section, field, values in SWEEPS:
        if args.only is not None and f"{section}.{field}" not in args.only:
            continue
        LOGGER.info("%s.%s (default %s)", section, field, getattr(getattr(config, section), field))
        for value in values:
            line = describe(tracking, phases, with_value(config, section, field, value))
            LOGGER.info("  %8g  %s", value, line)


if __name__ == "__main__":
    main()
