"""Locations of repository files used by the tests."""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPOSITORY / "config" / "default.yaml"
