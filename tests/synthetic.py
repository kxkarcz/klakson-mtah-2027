"""Builders of small synthetic inputs. No test reads the real match data."""

from __future__ import annotations

import numpy as np

from possession.structures import AWAY, HOME, Tracking

FPS = 25.0
LENGTH, WIDTH = 100.0, 60.0

METADATA_XML = """<?xml version="1.0"?>
<main>
  <Metadata>
    <GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>
    <Sessions><Session id="s1">
      <FieldSize><Width>100</Width><Height>60</Height></FieldSize>
    </Session></Sessions>
    <Teams>
      <Team id="T_home"><Name>Home FC</Name></Team>
      <Team id="T_away"><Name>Away United</Name></Team>
    </Teams>
    <Players>
      <Player id="P1" teamId="T_home"><Name>Track 1</Name></Player>
      <Player id="P2" teamId="T_away"><Name>Track 2</Name></Player>
      <Player id="P3" teamId="T_home"><Name>Track 3</Name></Player>
    </Players>
  </Metadata>
</main>
"""


def make_tracking(
    n_frames: int,
    home_xy: np.ndarray,
    away_xy: np.ndarray,
    ball_xy: np.ndarray,
    speed: float | np.ndarray = 2.0,
) -> Tracking:
    """Tracking with fixed formations: ``home_xy``/``away_xy`` are ``(n_players, 2)`` or
    ``(n_frames, n_players, 2)``; ``speed`` is a scalar or a per-frame array."""

    def expand(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return np.broadcast_to(xy, (n_frames, *xy.shape[-2:])).copy()

    home, away = expand(home_xy), expand(away_xy)
    player_xy = np.concatenate((home, away), axis=1)
    per_frame = np.broadcast_to(np.asarray(speed, dtype=np.float64), (n_frames,))
    return Tracking(
        player_xy=player_xy,
        player_speed=np.repeat(per_frame[:, None], player_xy.shape[1], axis=1),
        slot_team=np.array([HOME] * home.shape[1] + [AWAY] * away.shape[1], dtype=np.int8),
        ball_xy=np.asarray(ball_xy, dtype=np.float64),
        fps=FPS,
        pitch_length=LENGTH,
        pitch_width=WIDTH,
    )


def grid_team(x_centre: float, n_players: int = 11) -> np.ndarray:
    """A deterministic block of players around ``x_centre``."""
    rows = np.arange(n_players)
    return np.column_stack((x_centre + (rows % 3 - 1) * 8.0, 8.0 + rows * 4.0))
