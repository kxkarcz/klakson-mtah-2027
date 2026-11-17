"""Submission file: formatting, writing and the official validation."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from possession.structures import LABEL_NAMES, LabelArray

HEADER = "frame,label"


def format_submission(labels: LabelArray) -> str:
    """Render labels as the ``frame,label`` CSV the scorer expects, LF line endings.

    The frame column repeats the row number on purpose: the validator uses it to turn an
    off-by-one into a hard error.
    """
    if labels.size and (labels.min() < 0 or labels.max() >= len(LABEL_NAMES)):
        raise ValueError("label codes outside the H/A/D/X alphabet")
    names = np.array(LABEL_NAMES)[labels]
    rows = "\n".join(f"{frame},{name}" for frame, name in enumerate(names))
    return f"{HEADER}\n{rows}\n" if labels.size else f"{HEADER}\n"


def write_submission(path: Path, labels: LabelArray) -> str:
    """Write the CSV byte-for-byte reproducibly and return its SHA-256.

    Written in binary mode so the platform never rewrites line endings; the same labels give
    the same bytes on Windows and Linux.
    """
    payload = format_submission(labels).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def run_official_validator(validator_script: Path, submission: Path) -> tuple[bool, str]:
    """Run the organisers' ``validate_submission.py`` on the file; ``(passed, output)``.

    The official script is executed as-is rather than re-implemented, so "valid" here means
    exactly what it will mean at scoring time.
    """
    completed = subprocess.run(
        [sys.executable, str(validator_script), str(submission)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0, completed.stdout + completed.stderr
