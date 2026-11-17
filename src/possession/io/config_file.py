"""Loading of the YAML configuration file."""

from __future__ import annotations

from pathlib import Path

import yaml

from possession.config import Config, config_from_mapping


def load_config(path: Path) -> Config:
    """Read, type-check and validate a configuration file."""
    return config_from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
