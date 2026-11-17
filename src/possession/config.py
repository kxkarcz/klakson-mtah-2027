"""Typed, validated configuration loaded from YAML.

Every tunable number of the pipeline lives in ``config/default.yaml`` with its unit and its
football rationale; this module only gives those numbers types and refuses anything unexpected,
so a typo in the YAML fails loudly instead of silently running with a default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from dataclasses import dataclass
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


@dataclass(frozen=True)
class DataConfig:
    """Where the provided assets live, relative to the data directory."""

    data_dir_env: str
    tracking_metadata: str
    tracking_raw: str
    smart_tagging: str


@dataclass(frozen=True)
class BallConfig:
    """Cleaning of the ball track and the kinematic thresholds derived from it."""

    max_interpolation_gap_frames: int
    speed_smoothing_frames: int
    out_of_bounds_margin_m: float
    stationary_speed_mps: float
    fast_speed_mps: float
    short_gap_max_frames: int
    medium_gap_max_frames: int
    max_plausible_speed_mps: float


@dataclass(frozen=True)
class ActivityConfig:
    """Collective movement intensity used as a live/dead cue."""

    smoothing_frames: int
    min_players: int
    lookahead_enabled: bool
    lookahead_frames: int


@dataclass(frozen=True)
class KickoffConfig:
    """Detection of the kick-off line-up and of the halves."""

    halfway_tolerance_m: float
    min_players_per_team: int
    min_formation_frames: int
    tag_margin_frames: int
    halftime_smoothing_frames: int


@dataclass(frozen=True)
class LivenessConfig:
    """Unsupervised live/dead model (two-state HMM fitted by EM)."""

    em_iterations: int
    initial_dead_probability: float
    expected_live_spell_frames: float
    expected_dead_spell_frames: float
    probability_floor: float
    emission_temperature: float
    seed_activity_split_mps: float
    min_activity_std_mps: float


@dataclass(frozen=True)
class OwnershipConfig:
    """Team-in-control evidence from ball-player geometry."""

    control_radius_m: float
    reach_radius_m: float
    distance_scale_m: float
    ball_speed_scale_mps: float
    control_llr_per_frame: float
    duel_margin_m: float


@dataclass(frozen=True)
class CollectiveConfig:
    """Ball-free ownership cue learnt from pseudo-labels."""

    enabled: bool
    velocity_lag_frames: int
    min_players_per_team: int
    pseudo_label_min_llr: float
    ridge_penalty: float
    fit_iterations: int
    max_llr_per_frame: float
    temperature: float


@dataclass(frozen=True)
class TaggingConfig:
    """Smart Tagging as soft evidence."""

    enabled: bool
    possession_codes: tuple[str, ...]
    opponent_codes: tuple[str, ...]
    transition_codes: tuple[str, ...]
    set_piece_code: str
    kickoff_start_type: str
    edge_trim_frames: int
    temperature: float
    max_llr_per_frame: float
    min_calibration_frames: int


@dataclass(frozen=True)
class DecodingConfig:
    """Transition costs and minimum durations of the joint H/A/D decoder."""

    switch_cost: float
    dead_enter_cost: float
    dead_exit_cost: float
    min_possession_frames: int
    min_dead_frames: int
    transition_discount_enabled: bool
    transition_discount: float
    transition_window_before_frames: int
    transition_window_after_frames: int


@dataclass(frozen=True)
class AbstentionConfig:
    """Where the submission writes X instead of a label."""

    outside_match: bool
    post_match_margin_frames: int
    in_play_enabled: bool
    in_play_min_gap_frames: int
    in_play_boundary_guard_frames: int


@dataclass(frozen=True)
class EventsConfig:
    """Detection of discrete ball events (exits, restarts, touches)."""

    exit_margin_m: float
    line_proximity_m: float
    heading_lag_frames: int
    min_rest_frames: int
    kick_speed_mps: float
    kick_window_frames: int
    kink_lag_frames: int
    kink_min_velocity_change_mps: float
    kink_min_separation_frames: int


@dataclass(frozen=True)
class DeadEdgeConfig:
    """Snapping of stoppage edges to ball events."""

    enabled: bool
    start_window_before_frames: int
    start_window_after_frames: int
    end_window_before_frames: int
    end_window_after_frames: int
    min_remaining_frames: int
    use_deceleration: bool
    deceleration_lag_frames: int


@dataclass(frozen=True)
class TouchSnapConfig:
    """Snapping of changes of possession to the touch that made them."""

    enabled: bool
    window_before_frames: int
    window_after_frames: int
    touch_radius_m: float
    min_remaining_frames: int


@dataclass(frozen=True)
class TouchEvidenceConfig:
    """Ownership evidence from the touches themselves (who was at the ball when it turned)."""

    enabled: bool
    llr_per_touch: float
    touch_radius_m: float
    duel_margin_m: float


@dataclass(frozen=True)
class LearnedOwnershipConfig:
    """Supervised ownership model: feature definitions and the trained weights."""

    enabled: bool
    llr_scale: float
    radii_m: tuple[float, ...]
    banded_radii_m: tuple[float, ...]
    speed_band_edges_mps: tuple[float, ...]
    side_scale_m: float
    side_reach_radii_m: tuple[float, ...]
    second_scale_m: float
    support_radius_m: float
    support_scale_players: float
    support_clip: float
    context_radius_m: float
    context_windows_frames: tuple[int, ...]
    last_touch_decay_frames: tuple[float, ...]
    next_touch_decay_frames: tuple[float, ...]
    repeated_touch_decay_frames: float
    weights: tuple[float, ...]


def ownership_feature_names(config: LearnedOwnershipConfig) -> tuple[str, ...]:
    """Names of the learned model's feature columns, in the order of its weights."""
    names = [f"alone_within_{radius}" for radius in config.radii_m]
    bands = len(config.speed_band_edges_mps) + 1
    names += [
        f"alone_within_{radius}_speed_band_{band}"
        for radius in config.banded_radii_m
        for band in range(bands)
    ]
    names += [f"closer_side_reach_{radius}" for radius in config.side_reach_radii_m]
    names += ["second_nearest", "support"]
    for base in ("alone", "closer_side"):
        for window in config.context_windows_frames:
            names += [f"{base}_past_{window}", f"{base}_next_{window}"]
    names += [f"last_touch_decay_{decay}" for decay in config.last_touch_decay_frames]
    names += ["last_touch"]
    names += [f"next_touch_decay_{decay}" for decay in config.next_touch_decay_frames]
    names += ["touch_now", "repeated_touch"]
    return tuple(names)


@dataclass(frozen=True)
class LearnedLivenessConfig:
    """Supervised live/dead model: feature definitions and the trained weights."""

    enabled: bool
    llr_scale: float
    activity_cap_mps: float
    activity_scale_mps: float
    activity_windows_frames: tuple[tuple[int, int], ...]
    visibility_windows_frames: tuple[tuple[int, int], ...]
    gap_cap_frames: int
    ball_speed_cap_mps: float
    ball_speed_scale_mps: float
    resting_speed_mps: float
    edge_cap_m: float
    edge_scale_m: float
    near_line_m: float
    nearest_cap_m: float
    nearest_scale_m: float
    nearest_window_frames: int
    sprint_speed_mps: float
    speed_spread_cap_mps: float
    team_gap_default_m: float
    team_gap_cap_m: float
    team_gap_scale_m: float
    layout_scale_m: float
    event_latest_decays_frames: tuple[float, ...]
    event_next_decays_frames: tuple[float, ...]
    weights: tuple[float, ...]


def liveness_feature_count(config: LearnedLivenessConfig) -> int:
    """Feature columns of the learned live/dead model, intercept included."""
    ball_classes = 7  # dead_ball.N_BALL_CLASSES; a test keeps the two in step
    return (
        1
        + ball_classes
        + 2
        + len(config.activity_windows_frames)
        + len(config.visibility_windows_frames)
        + 3
        + 2
        + 2
        + 2
        + 2
        + 1
        + 3
        + 3 * (len(config.event_latest_decays_frames) + len(config.event_next_decays_frames))
    )


@dataclass(frozen=True)
class ReportConfig:
    """Tolerances of the label-free diagnostics (they never affect the labels)."""

    restart_tight_tolerance_frames: int
    restart_loose_tolerance_frames: int
    boundary_event_tolerance_frames: int
    switch_kink_tolerance_frames: int


@dataclass(frozen=True)
class Config:
    """The full pipeline configuration."""

    data: DataConfig
    ball: BallConfig
    activity: ActivityConfig
    kickoff: KickoffConfig
    liveness: LivenessConfig
    ownership: OwnershipConfig
    collective: CollectiveConfig
    tagging: TaggingConfig
    decoding: DecodingConfig
    abstention: AbstentionConfig
    events: EventsConfig
    dead_edges: DeadEdgeConfig
    touch_snap: TouchSnapConfig
    touch_evidence: TouchEvidenceConfig
    learned_ownership: LearnedOwnershipConfig
    learned_liveness: LearnedLivenessConfig
    report: ReportConfig


class ConfigError(ValueError):
    """Raised when the YAML does not describe a valid configuration."""


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Check one YAML value against its declared type, converting sequences to tuples."""
    origin = get_origin(annotation)
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _build(annotation, value, path)
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        inner = get_args(annotation)[0]
        return tuple(_coerce(item, inner, f"{path}[{i}]") for i, item in enumerate(value))
    if origin in (Union, types.UnionType):
        raise ConfigError(f"{path}: union types are not supported in the configuration")
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if annotation in (bool, str):
        if not isinstance(value, annotation):
            raise ConfigError(f"{path}: expected {annotation.__name__}, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported field type {annotation!r}")


def _build(cls: type[T], mapping: Any, path: str) -> T:
    """Instantiate a config dataclass from a mapping with exactly its fields."""
    if not isinstance(mapping, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping")
    hints = get_type_hints(cls)
    names = [field.name for field in dataclasses.fields(cls)]  # type: ignore[arg-type]
    missing = sorted(set(names) - set(mapping))
    unknown = sorted(set(mapping) - set(names))
    if missing or unknown:
        raise ConfigError(f"{path or 'config'}: missing keys {missing}, unknown keys {unknown}")
    return cls(**{name: _coerce(mapping[name], hints[name], f"{path}.{name}") for name in names})


def _validate(config: Config) -> None:
    """Reject values that are well-typed but meaningless."""
    checks = {
        "ball.max_interpolation_gap_frames >= 0": config.ball.max_interpolation_gap_frames >= 0,
        "ball.speed_smoothing_frames >= 1": config.ball.speed_smoothing_frames >= 1,
        "ball gap classes ordered": 0
        < config.ball.short_gap_max_frames
        < config.ball.medium_gap_max_frames,
        "ball speeds ordered": 0
        < config.ball.stationary_speed_mps
        < config.ball.fast_speed_mps
        < config.ball.max_plausible_speed_mps,
        "liveness.min_activity_std_mps > 0": config.liveness.min_activity_std_mps > 0,
        "activity.smoothing_frames >= 1": config.activity.smoothing_frames >= 1,
        "activity.lookahead_frames >= 1": config.activity.lookahead_frames >= 1,
        "liveness.em_iterations >= 0": config.liveness.em_iterations >= 0,
        "0 < liveness.initial_dead_probability < 1": 0
        < config.liveness.initial_dead_probability
        < 1,
        "liveness spell lengths > 1": min(
            config.liveness.expected_live_spell_frames,
            config.liveness.expected_dead_spell_frames,
        )
        > 1,
        "liveness.emission_temperature > 0": config.liveness.emission_temperature > 0,
        "0 < liveness.probability_floor < 0.1": 0 < config.liveness.probability_floor < 0.1,
        "ownership radii ordered": 0
        < config.ownership.control_radius_m
        < config.ownership.reach_radius_m,
        "ownership scales > 0": min(
            config.ownership.distance_scale_m,
            config.ownership.ball_speed_scale_mps,
            config.ownership.control_llr_per_frame,
        )
        > 0,
        "decoding costs >= 0": min(
            config.decoding.switch_cost,
            config.decoding.dead_enter_cost,
            config.decoding.dead_exit_cost,
        )
        >= 0,
        "decoding minimum durations >= 1": min(
            config.decoding.min_possession_frames, config.decoding.min_dead_frames
        )
        >= 1,
        "events frame counts >= 1": min(
            config.events.heading_lag_frames,
            config.events.min_rest_frames,
            config.events.kick_window_frames,
            config.events.kink_lag_frames,
            config.events.kink_min_separation_frames,
        )
        >= 1,
        "events.exit_margin_m >= 0": config.events.exit_margin_m >= 0,
        "0 <= decoding.transition_discount <= switch_cost": 0
        <= config.decoding.transition_discount
        <= config.decoding.switch_cost,
        "decoding transition windows >= 0": min(
            config.decoding.transition_window_before_frames,
            config.decoding.transition_window_after_frames,
        )
        >= 0,
        "touch_snap windows >= 0": min(
            config.touch_snap.window_before_frames, config.touch_snap.window_after_frames
        )
        >= 0,
        "touch_snap.touch_radius_m > 0": config.touch_snap.touch_radius_m > 0,
        "touch_snap.min_remaining_frames >= 1": config.touch_snap.min_remaining_frames >= 1,
        "touch_evidence.llr_per_touch >= 0": config.touch_evidence.llr_per_touch >= 0,
        "touch_evidence.touch_radius_m > 0": config.touch_evidence.touch_radius_m > 0,
        "touch_evidence.duel_margin_m >= 0": config.touch_evidence.duel_margin_m >= 0,
        "learned_ownership.llr_scale > 0": config.learned_ownership.llr_scale > 0,
        "learned_liveness.llr_scale > 0": config.learned_liveness.llr_scale > 0,
        "learned_liveness has one weight per feature": len(config.learned_liveness.weights)
        == liveness_feature_count(config.learned_liveness),
        "learned_liveness windows are non-empty": all(
            stop > start
            for start, stop in (
                *config.learned_liveness.activity_windows_frames,
                *config.learned_liveness.visibility_windows_frames,
            )
        ),
        "learned_liveness scales, caps and decays > 0": min(
            config.learned_liveness.activity_cap_mps,
            config.learned_liveness.activity_scale_mps,
            config.learned_liveness.gap_cap_frames,
            config.learned_liveness.ball_speed_cap_mps,
            config.learned_liveness.ball_speed_scale_mps,
            config.learned_liveness.resting_speed_mps,
            config.learned_liveness.edge_cap_m,
            config.learned_liveness.edge_scale_m,
            config.learned_liveness.near_line_m,
            config.learned_liveness.nearest_cap_m,
            config.learned_liveness.nearest_scale_m,
            config.learned_liveness.nearest_window_frames,
            config.learned_liveness.sprint_speed_mps,
            config.learned_liveness.speed_spread_cap_mps,
            config.learned_liveness.team_gap_default_m,
            config.learned_liveness.team_gap_cap_m,
            config.learned_liveness.team_gap_scale_m,
            config.learned_liveness.layout_scale_m,
            *config.learned_liveness.event_latest_decays_frames,
            *config.learned_liveness.event_next_decays_frames,
        )
        > 0,
        "learned_ownership has one weight per feature": len(config.learned_ownership.weights)
        == len(ownership_feature_names(config.learned_ownership)),
        "learned_ownership radii, scales and decays > 0": min(
            *config.learned_ownership.radii_m,
            *config.learned_ownership.banded_radii_m,
            *config.learned_ownership.side_reach_radii_m,
            *config.learned_ownership.last_touch_decay_frames,
            *config.learned_ownership.next_touch_decay_frames,
            config.learned_ownership.side_scale_m,
            config.learned_ownership.second_scale_m,
            config.learned_ownership.support_radius_m,
            config.learned_ownership.support_scale_players,
            config.learned_ownership.support_clip,
            config.learned_ownership.context_radius_m,
            config.learned_ownership.repeated_touch_decay_frames,
        )
        > 0,
        "learned_ownership.side_reach_radii_m is not empty": len(
            config.learned_ownership.side_reach_radii_m
        )
        >= 1,
        "learned_ownership.context_windows_frames >= 1": min(
            config.learned_ownership.context_windows_frames, default=1
        )
        >= 1,
        "learned_ownership.speed_band_edges_mps increase": list(
            config.learned_ownership.speed_band_edges_mps
        )
        == sorted(set(config.learned_ownership.speed_band_edges_mps)),
        "dead_edges windows >= 0": min(
            config.dead_edges.start_window_before_frames,
            config.dead_edges.start_window_after_frames,
            config.dead_edges.end_window_before_frames,
            config.dead_edges.end_window_after_frames,
        )
        >= 0,
        "dead_edges frame counts >= 1": min(
            config.dead_edges.min_remaining_frames, config.dead_edges.deceleration_lag_frames
        )
        >= 1,
        "events thresholds > 0": min(
            config.events.line_proximity_m,
            config.events.kick_speed_mps,
            config.events.kink_min_velocity_change_mps,
        )
        > 0,
        "report tolerances ordered": 0
        < config.report.restart_tight_tolerance_frames
        <= config.report.restart_loose_tolerance_frames,
        "report tolerances >= 1": min(
            config.report.boundary_event_tolerance_frames,
            config.report.switch_kink_tolerance_frames,
        )
        >= 1,
        "tagging.temperature > 0": config.tagging.temperature > 0,
        "collective.temperature > 0": config.collective.temperature > 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ConfigError(f"invalid configuration: {failed}")


def config_from_mapping(mapping: Any) -> Config:
    """Build and validate a ``Config`` from parsed YAML."""
    config = _build(Config, mapping, "")
    _validate(config)
    return config


def config_fingerprint(config: Config, *sections: str) -> str:
    """Stable short hash of the named sections (all of them when none is named).

    Used as a cache key: a cached intermediate result is only valid for the exact parameters
    that produced it, and hashing only the relevant sections keeps unrelated edits from
    invalidating expensive stages.
    """
    full = dataclasses.asdict(config)
    chosen = {name: full[name] for name in (sections or tuple(full))}
    payload = json.dumps(chosen, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
