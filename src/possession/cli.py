"""Command line: ``python -m possession run --config config/default.yaml --out out.csv``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from possession.config import Config, config_fingerprint
from possession.diagnostics import diagnostics_report
from possession.io.atd import parse_team_names
from possession.io.cache import load_tracking_cached
from possession.io.config_file import load_config
from possession.io.smart_tagging import read_smart_tagging
from possession.pipeline import run_pipeline
from possession.submission import run_official_validator, write_submission

LOGGER = logging.getLogger("possession")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``possession`` command."""
    parser = argparse.ArgumentParser(prog="possession", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="infer possession and write the submission file")
    run.add_argument("--config", type=Path, required=True, help="YAML configuration")
    run.add_argument("--out", type=Path, required=True, help="submission CSV to write")
    run.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="data bundle root (default: the environment variable named in the config)",
    )
    run.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="where parsed inputs are cached (git-ignored)",
    )
    run.add_argument("--no-cache", action="store_true", help="parse the inputs from scratch")
    run.add_argument(
        "--validator",
        type=Path,
        default=None,
        help="path to the official evaluation/validate_submission.py; run on the output if given",
    )
    run.add_argument("--report", type=Path, default=None, help="write diagnostics JSON here")
    run.add_argument("--verbose", action="store_true", help="log at DEBUG level")
    return parser


def resolve_data_dir(config: Config, override: Path | None) -> Path:
    """Return the data bundle root: the flag, else the configured environment variable."""
    if override is not None:
        return override
    from_env = os.environ.get(config.data.data_dir_env)
    if not from_env:
        raise SystemExit(f"no data directory: pass --data-dir or set {config.data.data_dir_env}")
    return Path(from_env)


def run(args: argparse.Namespace) -> int:
    """Execute the ``run`` command; returns the process exit code."""
    config = load_config(args.config)
    data_dir = resolve_data_dir(config, args.data_dir)
    metadata_path = data_dir / config.data.tracking_metadata
    raw_path = data_dir / config.data.tracking_raw
    LOGGER.info("configuration fingerprint %s", config_fingerprint(config))

    tracking = load_tracking_cached(
        metadata_path,
        raw_path,
        None if args.no_cache else args.cache_dir,
        config_fingerprint(config, "data"),
    )
    home_name, away_name = parse_team_names(metadata_path.read_text(encoding="utf-8"))
    phases = read_smart_tagging(
        data_dir / config.data.smart_tagging, tracking.fps, home_name, away_name
    )
    LOGGER.info("loaded %d frames, %d tagged phases", tracking.n_frames, len(phases))

    result = run_pipeline(tracking, phases, config)
    digest = write_submission(args.out, result.labels)
    LOGGER.info("wrote %s", args.out)
    LOGGER.info("sha256 %s", digest)

    report = diagnostics_report(result, phases, config, tracking.fps)
    report["sha256"] = digest
    report["config_fingerprint"] = config_fingerprint(config)
    LOGGER.info("diagnostics\n%s", json.dumps(report, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.validator is not None:
        passed, output = run_official_validator(args.validator, args.out)
        LOGGER.info("official validator:\n%s", output.strip())
        if not passed:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return run(args)
