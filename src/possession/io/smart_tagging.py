"""Reader for the SportsCode Smart Tagging export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from possession.structures import AWAY, HOME, TaggedPhase


def seconds_to_frame(seconds: float, fps: float) -> int:
    """Video frame of a tag timestamp (seconds from the first frame of the tactical video)."""
    return round(seconds * fps)


def parse_smart_tagging(
    xml_text: str, fps: float, home_team_name: str, away_team_name: str
) -> list[TaggedPhase]:
    """Parse instances into frame-indexed phases, sorted by start.

    Codes and tag values are matched case-insensitively because the export mixes cases, and a
    silent non-match would simply drop evidence. A team value that is neither club (the provider
    writes "N/A") becomes ``None`` rather than a guess.
    """
    sides = {home_team_name.strip().lower(): HOME, away_team_name.strip().lower(): AWAY}
    phases: list[TaggedPhase] = []
    for instance in ET.fromstring(xml_text).iter("instance"):
        start, end, code = (instance.findtext(tag) for tag in ("start", "end", "code"))
        if start is None or end is None or code is None:
            raise ValueError("Smart Tagging instance without start, end or code")
        tags = {
            (label.findtext("group") or "").strip().lower(): (label.findtext("text") or "").strip()
            for label in instance.findall("label")
        }
        set_piece_type = tags.get("type")
        phases.append(
            TaggedPhase(
                code=code.strip().upper(),
                start_frame=seconds_to_frame(float(start), fps),
                end_frame=seconds_to_frame(float(end), fps),
                team=sides.get(tags.get("team", "").lower()),
                set_piece_type=set_piece_type.lower() if set_piece_type else None,
            )
        )
    return sorted(phases, key=lambda phase: (phase.start_frame, phase.end_frame, phase.code))


def read_smart_tagging(
    path: Path, fps: float, home_team_name: str, away_team_name: str
) -> list[TaggedPhase]:
    """Read and parse the Smart Tagging file from disk."""
    return parse_smart_tagging(
        path.read_text(encoding="utf-8"), fps, home_team_name, away_team_name
    )
