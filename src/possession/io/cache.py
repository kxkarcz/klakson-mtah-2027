"""On-disk cache of the parsed tracking.

Parsing the raw text is the slowest deterministic step and its result depends only on the
input bytes, so it is stored as ``.npz`` under a key derived from those bytes and from the
data section of the configuration. The cache directory is git-ignored: its content would
reconstruct the licensed tracking.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from possession.io.atd import read_tracking
from possession.structures import Tracking

_CHUNK_BYTES = 1 << 20


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def tracking_cache_key(metadata_path: Path, raw_path: Path, data_fingerprint: str) -> str:
    """Cache key: changes whenever either input file or the data configuration changes."""
    joined = "|".join((file_sha256(metadata_path), file_sha256(raw_path), data_fingerprint))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()[:20]


def load_tracking_cached(
    metadata_path: Path, raw_path: Path, cache_dir: Path | None, data_fingerprint: str
) -> Tracking:
    """Parse the tracking, reusing a cached copy when the inputs are unchanged."""
    if cache_dir is None:
        return read_tracking(metadata_path, raw_path)
    key = tracking_cache_key(metadata_path, raw_path, data_fingerprint)
    cache_file = cache_dir / f"tracking_{key}.npz"
    if cache_file.is_file():
        with np.load(cache_file) as stored:
            return Tracking(
                player_xy=stored["player_xy"],
                player_speed=stored["player_speed"],
                slot_team=stored["slot_team"],
                ball_xy=stored["ball_xy"],
                fps=float(stored["fps"]),
                pitch_length=float(stored["pitch_length"]),
                pitch_width=float(stored["pitch_width"]),
            )
    tracking = read_tracking(metadata_path, raw_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_file,
        player_xy=tracking.player_xy,
        player_speed=tracking.player_speed,
        slot_team=tracking.slot_team,
        ball_xy=tracking.ball_xy,
        fps=tracking.fps,
        pitch_length=tracking.pitch_length,
        pitch_width=tracking.pitch_width,
    )
    return tracking
