"""The full inference pipeline as one pure function of the parsed inputs."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np

from possession.abstention import apply_abstention
from possession.config import Config
from possession.dead_ball import (
    DEAD_BALL,
    LIVE,
    LivenessModel,
    ball_observation_class,
    fit_liveness,
    liveness_log_emissions,
    states_are_identified,
)
from possession.decoding import decode_possession
from possession.features.ball_events import BallEvents, detect_ball_events, restart_candidates
from possession.features.collective import collective_activity, team_shape
from possession.features.geometry import ball_proximity
from possession.features.kickoff import MatchStructure, match_structure
from possession.features.shape_cue import (
    ShapeModel,
    fit_shape_model,
    home_attack_sign,
    shape_features,
    shape_log_odds,
)
from possession.learned_liveness import learned_liveness_emissions, liveness_features
from possession.learned_ownership import learned_log_odds, ownership_features
from possession.ownership import control_log_odds, ownership_log_emissions, touch_log_odds
from possession.preprocessing.ball import BallTrack, clean_ball
from possession.preprocessing.teams import assign_slots_to_teams
from possession.refinement.dead_ball_edges import EdgeRefinement, refine_dead_ball_edges
from possession.refinement.touch_snap import TouchSnap, snap_switches_to_touches
from possession.structures import (
    AWAY,
    DEAD,
    HOME,
    N_STATES,
    BoolArray,
    FloatArray,
    LabelArray,
    TaggedPhase,
    Tracking,
)
from possession.tagging_prior import (
    TaggingCalibration,
    calibrate_votes,
    phase_votes,
    tagging_log_odds,
    transition_switch_discount,
)

LOGGER = logging.getLogger(__name__)

# Log-score that makes a state unreachable without producing NaN in sums.
_IMPOSSIBLE = -1.0e9


@dataclass(frozen=True)
class PipelineResult:
    """Final labels plus everything the diagnostics report on."""

    labels: LabelArray  # with abstentions, the submission content
    decoded: LabelArray  # H/A/D on every frame, before abstention
    viterbi: LabelArray  # the decoder output before any boundary refinement
    edge_refinement: EdgeRefinement | None
    touch_snap: TouchSnap | None
    first_pass: LabelArray  # tracking-only decode used to produce pseudo-labels
    ball: BallTrack
    events: BallEvents
    structure: MatchStructure
    liveness: LivenessModel
    tagging: TaggingCalibration
    votes: FloatArray
    shape_model: ShapeModel | None


def joint_log_emissions(
    liveness_emissions: FloatArray, log_odds_home: FloatArray, forced_dead: BoolArray
) -> FloatArray:
    """Assemble ``(n_frames, 3)`` log-emissions over (HOME, AWAY, DEAD).

    A possession state needs the frame to look live *and* to favour that team; the dead state
    needs it to look dead and is indifferent to who stands next to the ball.
    """
    emissions = np.empty((liveness_emissions.shape[0], N_STATES))
    ownership = ownership_log_emissions(log_odds_home)
    live = np.where(forced_dead, _IMPOSSIBLE, liveness_emissions[:, LIVE])
    emissions[:, HOME] = live + ownership[:, 0]
    emissions[:, AWAY] = live + ownership[:, 1]
    emissions[:, DEAD] = liveness_emissions[:, DEAD_BALL]
    return emissions


def run_pipeline(tracking: Tracking, phases: list[TaggedPhase], config: Config) -> PipelineResult:
    """Infer the per-frame possession labels of one match."""
    masks = assign_slots_to_teams(tracking)
    ball = clean_ball(tracking, config.ball)
    ball_known = ~np.isnan(ball.xy[:, 0])
    events = detect_ball_events(ball, tracking.fps, config.events, config.ball)
    structure = match_structure(tracking, masks, phases, config.kickoff, config.tagging)

    activity = collective_activity(tracking, masks, config.activity)
    ball_class = ball_observation_class(ball, config.ball)
    liveness = fit_liveness(ball_class, activity, structure.in_match, config.liveness)
    if not states_are_identified(liveness):
        raise RuntimeError("liveness EM converged with the dead state faster than the live one")
    liveness_emissions = liveness_log_emissions(
        liveness, ball_class, activity, config.liveness.emission_temperature
    )
    if config.learned_liveness.enabled:
        liveness_emissions = learned_liveness_emissions(
            liveness_features(
                tracking,
                ball=ball,
                ball_class=ball_class,
                activity_now=collective_activity(
                    tracking, masks, dataclasses.replace(config.activity, lookahead_enabled=False)
                ),
                shapes=(
                    team_shape(
                        tracking.player_xy, masks.home, config.collective.min_players_per_team
                    ),
                    team_shape(
                        tracking.player_xy, masks.away, config.collective.min_players_per_team
                    ),
                ),
                events=events,
                config=config.learned_liveness,
            ),
            config.learned_liveness,
        )

    proximity = ball_proximity(tracking.player_xy, masks, ball.xy)
    if config.learned_ownership.enabled:
        control = learned_log_odds(
            ownership_features(
                tracking.player_xy,
                masks,
                ball,
                events.kink,
                config=config.learned_ownership,
                touch_config=config.touch_evidence,
            ),
            config.learned_ownership,
        )
    else:
        control = control_log_odds(proximity, ball.speed, config.ownership)
        if config.touch_evidence.enabled:
            control = control + touch_log_odds(events.kink, proximity, config.touch_evidence)
    first_pass = decode_possession(
        joint_log_emissions(liveness_emissions, control, structure.forced_dead), config.decoding
    )
    LOGGER.info("first pass decoded (tracking only)")

    # Pseudo-labels: frames where the ball is tracked and the tracking-only pass names a team.
    usable = ball.observed & (first_pass != DEAD) & structure.in_match
    votes = phase_votes(phases, tracking.n_frames, config.tagging)
    calibration = calibrate_votes(votes, first_pass, usable, config.tagging)
    log_odds = control.copy()
    if config.tagging.enabled:
        log_odds += tagging_log_odds(votes, calibration)

    shape_model: ShapeModel | None = None
    if config.collective.enabled:
        home_shape = team_shape(
            tracking.player_xy, masks.home, config.collective.min_players_per_team
        )
        away_shape = team_shape(
            tracking.player_xy, masks.away, config.collective.min_players_per_team
        )
        features = shape_features(
            home_shape,
            away_shape,
            home_attack_sign(home_shape, away_shape, structure.second_half_start),
            config.collective.velocity_lag_frames,
            tracking.fps,
        )
        confident = usable & (np.abs(control) >= config.collective.pseudo_label_min_llr)
        shape_model = fit_shape_model(features, first_pass, confident, config.collective)
        log_odds += shape_log_odds(shape_model, features, ball_known, config.collective)

    switch_discount = (
        transition_switch_discount(phases, tracking.n_frames, config.tagging, config.decoding)
        if config.decoding.transition_discount_enabled
        else None
    )
    decoded = decode_possession(
        joint_log_emissions(liveness_emissions, log_odds, structure.forced_dead),
        config.decoding,
        switch_discount=switch_discount,
    )
    viterbi_labels = decoded
    edge_refinement: EdgeRefinement | None = None
    if config.dead_edges.enabled:
        edge_refinement = refine_dead_ball_edges(
            decoded,
            events,
            restart_candidates(events, ball, config.events),
            activity=activity,
            forced_dead=structure.forced_dead,
            config=config.dead_edges,
        )
        decoded = edge_refinement.labels
    touch_snap: TouchSnap | None = None
    if config.touch_snap.enabled:
        touch_snap = snap_switches_to_touches(decoded, events.kink, proximity, config.touch_snap)
        decoded = touch_snap.labels
    labels = apply_abstention(
        decoded,
        ball_known,
        structure.first_kickoff_frame,
        structure.last_tagged_frame,
        config.abstention,
    )
    return PipelineResult(
        labels=labels,
        decoded=decoded,
        viterbi=viterbi_labels,
        edge_refinement=edge_refinement,
        touch_snap=touch_snap,
        first_pass=first_pass,
        ball=ball,
        events=events,
        structure=structure,
        liveness=liveness,
        tagging=calibration,
        votes=votes,
        shape_model=shape_model,
    )


__all__ = ["AWAY", "PipelineResult", "joint_log_emissions", "run_pipeline"]
