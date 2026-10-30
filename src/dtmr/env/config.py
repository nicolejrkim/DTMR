"""Runtime config dataclass for the Warp tracking environment (TrackingEnvConfig)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


# Environment fields read from the top level of the YAML. Every other field of the
# config below is a cost/regulariser setting and lives under `cost_config:`.
TOP_LEVEL_KEYS = frozenset({
    "motion_path", "motion_fps", "dt", "timestep", "action_scale", "dphi_exp_base",
    "kp", "kd", "inner_rate_pd",
    "model_dir", "model_file", "reset_ground", "joint_groups",
    "auto_joint_range", "joint_range_margin_abs", "joint_range_margin_ratio",
    "joint_range_round_to", "joint_range_min_width", "clip_joint_range_to_physical",
    "joint_range_physical_outset_ratio", "joint_range_physical_outset_abs",
})


@dataclass
class TrackingEnvConfig:
    """Runtime config for the Warp env: everything the env and its kernels read."""
    motion_path: str
    motion_fps: int = 50
    dt: float = 0.02
    timestep: float = 0.005
    action_scale: float = 1.0
    dphi_exp_base: float = 2.0
    kp: Sequence[float] = field(default_factory=tuple)
    kd: Sequence[float] = field(default_factory=tuple)
    # Tracking cost weights. Position terms use the smooth-abs2 robust norm below.
    w_pos_relcom: float = 30.0         # link positions relative to the keypoint centroid
    w_com: float = 10.0                # keypoint centroid position
    w_com_z_scale: float = 1.0         # multiplier on the vertical component of the w_com and w_pelvis errors
    w_pelvis: float = 0.0              # pelvis (root link) position; additive to w_com
    w_pos_relpelvis: float = 0.0       # link positions relative to the pelvis; additive to w_pos_relcom
    w_orient_pure: float = 10.0        # link orientations
    w_orient_offset: float = 0.0       # combined relcom-position and orientation residual via offset points
    w_energy: float = 0.1
    w_joint_vel: float = 0.0           # joint-velocity magnitude penalty
    # Reference joint-velocity tracking: 0.5 * ||qvel[6:] - ref_jvel * rate||^2, where the
    # reference velocity (the reference's qvel, lerped between frames) is rescaled by the
    # sampled playback rate dphi / dphi_star, so velocity and position targets agree when
    # the optimiser slows or speeds the clip. Anchors motion inside the smooth-abs2 dead
    # zone. Zeros (a magnitude penalty, with a warning) when the reference has no qvel.
    w_joint_vel_ref: float = 0.0
    w_self_collision: float = 0.0
    w_dphi: float = 1.0                # playback-rate regulariser
    # Robust-norm shape parameters (see _smooth_abs2 in the kernels).
    smooth_abs2_p: float = 0.2
    smooth_abs2_q: float = 4.0
    smooth_abs_p: float = 0.1
    cosh_p: float = 1.0
    self_collision_threshold: float = 0.03
    # Regularisers in the style of IsaacLab's locomotion rewards.
    w_action_rate: float = 0.0
    w_joint_limit: float = 0.0
    joint_limit_margin: float = 0.1
    # False: the joint-limit penalty measures against the commanded range (the output of
    # auto_joint_range); True: against the physical range of the MJCF.
    joint_limit_use_physical: bool = False
    # Undesired contacts: counts, per world and control step, the contacts in which a body
    # outside the allow-list takes part (contact_dist below the threshold; the world body
    # is always allowed). A count, not a penetration depth, so weights are small.
    w_undesired_contacts: float = 0.0
    # True: only floor contacts of non-allowed bodies count; body-body contacts are left
    # to w_self_collision. False: any contact counts.
    undesired_floor_only: bool = False
    undesired_contact_dist_threshold: float = 0.0
    undesired_allowed_bodies: Sequence[str] = ()
    # Ground clearance of links that must never take a landing (knees, shanks). The scene
    # collides feet only, so a knee dropping to the floor makes no contact; this charges
    # the squared shortfall below knee_ground_clearance instead. The shanks' collision
    # capsules are self-collision proxies that would intersect the floor in stance, so
    # enabling shin-floor collision is not an alternative.
    w_knee_ground: float = 0.0
    knee_ground_clearance: float = 0.12
    knee_ground_bodies: Sequence[str] = ()
    # Reference-relative mode: the threshold is the reference knee height at the current
    # phase minus knee_ground_margin, so a clip that legitimately puts a knee low (deep
    # squats, kneeling) is only charged for going lower than its source.
    knee_ground_ref_relative: bool = False
    knee_ground_margin: float = 0.0
    # Phase-acceleration penalty on d^2 phi / dt^2, the counterpart of w_dphi.
    w_dphi_accel: float = 0.0
    # Flight-phase playback-rate pinning: while the reference is airborne, penalise
    # w_flight_dphi * (dphi / dphi_star - 1)^2, since ballistic flight time is fixed by
    # physics; stance frames keep full timing freedom. Airborne means every
    # flight_feet_links link in the reference is above the clip's minimum foot height
    # plus flight_clearance for at least flight_min_frames; links absent from the
    # reference make the term inert.
    w_flight_dphi: float = 0.0
    flight_clearance: float = 0.10
    flight_min_frames: int = 3
    flight_feet_links: Sequence[str] = ()
    # Foot slip: squared horizontal velocity of the foot_bodies links while in contact
    # with the world (contact_dist below foot_slip_dist_threshold).
    w_foot_slip: float = 0.0
    foot_slip_dist_threshold: float = 0.0
    foot_bodies: Sequence[str] = ()
    # Foot impact: squared downward vertical velocity of a foot while in contact (stomping
    # at touchdown). Same contact gating as foot slip.
    w_foot_impact: float = 0.0
    # Foot yaw rate: squared world-frame yaw angular velocity of a planted foot, which a
    # pure pivot produces while barely moving the slip term. Same contact gating.
    w_foot_yaw_rate: float = 0.0
    # Joint jerk: (qvel - prev_qvel)^2 over the actuated joints, the realised-side
    # counterpart of w_action_rate.
    w_joint_jerk: float = 0.0
    # State-Lipschitz regulariser, after "Learning Smooth Humanoid Locomotion through
    # Lipschitz-Constrained Policies" (arXiv:2410.11825), by finite differences over
    # consecutive control steps:
    #   cost_lipschitz = ||a_t - a_{t-1}||^2 / (||q_t - q_{t-1}||^2 + lipschitz_eps)
    # with a the normalised joint-target action and q the actuated joint positions. Unlike
    # w_action_rate it charges only action changes the robot's motion does not explain.
    w_lipschitz: float = 0.0
    # Denominator floor of the Lipschitz ratio (rad^2 summed over the actuated joints):
    # guards the divide when the robot is nearly static and sets the crossover scale.
    lipschitz_eps: float = 1e-2
    # Early-horizon weighting of the per-step tracking cost:
    #   w(h) = 1 + (early_horizon_alpha - 1) * exp(-h / early_horizon_tau)
    # over horizon steps h; alpha 1.0 = off.
    early_horizon_alpha: float = 1.0
    early_horizon_tau: float = 1.0
    # Commanded joint range from the reference: when on, joint_range is clipped to a band
    # around the reference trajectory (margin_abs + margin_ratio * span, rounded to
    # round_to, at least min_width wide), keeping the samples in sensible target ranges.
    auto_joint_range: bool = False
    joint_range_margin_abs: float = 0.05
    joint_range_margin_ratio: float = 0.05
    joint_range_round_to: float = 0.01
    joint_range_min_width: float = 0.05
    # True: the reference-derived range is clipped to the physical range of the MJCF.
    # False: it follows the reference plus margin even past the hardware limits, so the
    # PD target may point outside them where the motion demands it. The joint-limit
    # penalty measures against the true limits either way.
    clip_joint_range_to_physical: bool = True
    # When clipping to the physical range, extend the clamp outward by this fraction of
    # each joint's span plus this many radians, so a position-PD target can sit past a
    # mechanical stop and hold torque against it (the sim still clamps the joint).
    joint_range_physical_outset_ratio: float = 0.0
    joint_range_physical_outset_abs: float = 0.0
    # Model directory under src/dtmr/models/ and the scene file inside it. Like every
    # body-name list in this config, these come from the robot descriptor
    # (src/dtmr/robots/<robot>.yaml) selected by `robot:` in the YAML.
    model_dir: str = ""
    # Named joint groups of the robot: {group: [glob patterns on joint names]}, e.g.
    # {"arms": ["*shoulder*", "*elbow*", "*wrist*"]}. Used wherever a knob addresses a
    # body part (see `TrackingEnv.joint_indices`). Comes from the robot descriptor.
    joint_groups: dict = field(default_factory=dict)
    model_file: str = "scene_mjx_torque_tracking.xml"
    # True: the PD law is recomputed at every physics substep from the current qpos and
    # qvel. False: the torque is computed once per control tick and held over the
    # substeps, which leaves a harmonic at half the control rate in the joint velocities.
    inner_rate_pd: bool = False
    # Action space: action[:nu] are normalized joint-position targets,
    # torque = kp*(target-q) - kd*qd, clipped to ctrlrange.
    # Per-link multipliers on cost_pos_relcom, cost_orient_pure and cost_orient_offset
    # (unlisted links: 1.0). The centroid target itself is never reweighted, since it is
    # the plain mean of the link positions.
    cost_link_scale: dict = field(default_factory=dict)
    # Per-link multipliers on the smooth-abs2 shape parameters of those three terms:
    #   _smooth_abs2(x, p, q) = (|x|^q + p^q)^(1/q) - p
    # p is the dead-zone width, q the transition sharpness; a named override multiplies
    # the global smooth_abs2_p / smooth_abs2_q for that link only.
    cost_link_smooth_abs2_p_scale: dict = field(default_factory=dict)
    cost_link_smooth_abs2_q_scale: dict = field(default_factory=dict)
    # Synthetic head keypoint: when set (an offset in the parent link's frame), a tracked
    # link named "head" is appended, computed from the parent's pose on both sides. It
    # enters cost_pos_relcom only and is excluded from the centroid; weight it with
    # cost_link_scale: {head: X}. None = off.
    head_keypoint_offset: Optional[Sequence[float]] = None
    head_keypoint_parent: str = ""
    # Spawn grounding: the robot resets to the reference's first frame, and a frame that
    # sits below the floor makes the contact solver eject it on step 1. When True, the
    # reset qpos alone is lifted until its lowest collision surface rests on the floor;
    # the reference, and so every tracking target, is untouched. A no-op when the first
    # frame does not penetrate.
    reset_ground: bool = True
