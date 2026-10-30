"""Warp @wp.kernel definitions for the tracking environment (ctrl, contacts, cost,
phase). Warp resolves a kernel's called @wp.func helpers against THIS module's
globals at kernel-registration time, so the funcs are imported by name below
BEFORE any kernel is defined. (A star import would import nothing here: every
func name starts with '_' and is skipped by `from .funcs import *`.)"""
import warp as wp

from .funcs import (
    NORM_EPS,
    _rotate_vec_wxyz,
    _smooth_abs2,
    _smooth_abs,
    _cosh_term,
    _slerp_wxyz,
)

# Floor on (playback rate / nominal rate) inside the phase-regularity cost: rates slower
# than this are penalised as if they were exactly this slow.
MIN_RATE_RATIO = wp.constant(1.0 / 16.0)


@wp.kernel
def _broadcast_state_kernel(
    src_qpos: wp.array(dtype=float),
    src_qvel: wp.array(dtype=float),
    dst_qpos: wp.array2d(dtype=float),
    dst_qvel: wp.array2d(dtype=float),
    nq: int,
    nv: int,
):
    wid = wp.tid()
    for i in range(nq):
        dst_qpos[wid, i] = src_qpos[i]
    for i in range(nv):
        dst_qvel[wid, i] = src_qvel[i]


@wp.kernel
def _compute_ctrl_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    actions_buf: wp.array3d(dtype=float),
    ctrl_out: wp.array2d(dtype=float),
    step_t: int,
    nu: int,
    action_scale: float,
    act_qpos_adr: wp.array(dtype=wp.int32),
    act_dof_adr: wp.array(dtype=wp.int32),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    joint_range_lo: wp.array(dtype=float),
    joint_range_hi: wp.array(dtype=float),
    torque_range_lo: wp.array(dtype=float),
    torque_range_hi: wp.array(dtype=float),
):
    """PD torque controller matching BaseEnv.act2tau.

    action ∈ [-1, 1] → normalized joint target → torque = kp*(q_tgt - q) - kd*qdot.
    """
    wid = wp.tid()
    for i in range(nu):
        action_val = actions_buf[wid, step_t, i]
        act_norm = (action_val * action_scale + 1.0) / 2.0
        target = joint_range_lo[i] + act_norm * (joint_range_hi[i] - joint_range_lo[i])
        target = wp.clamp(target, joint_range_lo[i], joint_range_hi[i])
        cur_q = qpos[wid, act_qpos_adr[i]]
        cur_qd = qvel[wid, act_dof_adr[i]]
        tau = kp[i] * (target - cur_q) - kd[i] * cur_qd
        ctrl_out[wid, i] = wp.clamp(tau, torque_range_lo[i], torque_range_hi[i])


@wp.kernel
def _self_collision_kernel(
    contact_geom: wp.array(dtype=wp.vec2i),       # (naconmax,)
    contact_dist: wp.array(dtype=float),          # (naconmax,)
    contact_worldid: wp.array(dtype=wp.int32),    # (naconmax,)
    nacon: wp.array(dtype=wp.int32),              # (1,) active contact count
    is_robot_geom: wp.array(dtype=wp.int32),      # (ngeom,) 1 if bodyid != 0
    geom_bodyid: wp.array(dtype=wp.int32),        # (ngeom,)
    threshold: float,
    self_coll_out: wp.array(dtype=float),         # (nworld,) pre-zeroed
):
    """Thread-per-contact. For each self-contact (both geoms robot, different
    bodies), atomically add 0.5 * pen^2 into the world's slot, i.e.
    `quadratic_norm(penetration) = 0.5 * ||pen||^2`."""
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    pair = contact_geom[cid]
    g1 = pair[0]
    g2 = pair[1]
    if g1 < 0 or g2 < 0:
        return
    if is_robot_geom[g1] == 0 or is_robot_geom[g2] == 0:
        return
    if geom_bodyid[g1] == geom_bodyid[g2]:
        return
    pen = threshold - contact_dist[cid]
    if pen <= 0.0:
        return
    wid = contact_worldid[cid]
    wp.atomic_add(self_coll_out, wid, 0.5 * pen * pen)


@wp.kernel
def _undesired_contacts_kernel(
    contact_geom: wp.array(dtype=wp.vec2i),       # (naconmax,)
    contact_dist: wp.array(dtype=float),          # (naconmax,)
    contact_worldid: wp.array(dtype=wp.int32),    # (naconmax,)
    nacon: wp.array(dtype=wp.int32),              # (1,)
    geom_bodyid: wp.array(dtype=wp.int32),        # (ngeom,)
    body_is_allowed: wp.array(dtype=wp.int32),    # (nbody,) 1 if allowed (incl. world), else 0
    dist_threshold: float,                        # count if contact_dist < this
    floor_only: int,                              # 1 = count ONLY floor contacts (self-contacts ignored)
    undesired_out: wp.array(dtype=float),         # (nworld,) pre-zeroed
):
    """Port of IsaacLab `mdp.undesired_contacts`: thread per active contact,
    +1.0 to the world's slot if at least one involved body is NOT in the
    allow list and the contact is actually touching (contact_dist below
    threshold). We use penetration/distance as the trigger instead of the
    contact-force magnitude IsaacLab uses, since MuJoCo Warp exposes
    contact.dist directly in the graph-capture-friendly path; the two are
    monotonically related for any contact with active force."""
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    if contact_dist[cid] >= dist_threshold:
        return
    pair = contact_geom[cid]
    g1 = pair[0]
    g2 = pair[1]
    if g1 < 0 or g2 < 0:
        return
    b1 = geom_bodyid[g1]
    b2 = geom_bodyid[g2]
    if floor_only == 1:
        # "Nothing but the allowed bodies may touch the FLOOR": exactly one
        # side must be the worldbody, and the other side decides. Self-contacts
        # are ignored here — w_self_collision owns those.
        if b1 != 0 and b2 != 0:
            return
        other = b1
        if b1 == 0:
            other = b2
        if body_is_allowed[other] == 1:
            return
    else:
        # Both allowed? Not undesired.
        if body_is_allowed[b1] == 1 and body_is_allowed[b2] == 1:
            return
    wid = contact_worldid[cid]
    wp.atomic_add(undesired_out, wid, 1.0)


@wp.kernel
def _foot_slip_kernel(
    contact_geom: wp.array(dtype=wp.vec2i),       # (naconmax,)
    contact_dist: wp.array(dtype=float),          # (naconmax,)
    contact_worldid: wp.array(dtype=wp.int32),    # (naconmax,)
    nacon: wp.array(dtype=wp.int32),              # (1,)
    geom_bodyid: wp.array(dtype=wp.int32),        # (ngeom,)
    body_is_foot: wp.array(dtype=wp.int32),       # (nbody,) 1 if foot link
    cvel: wp.array2d(dtype=wp.spatial_vector),    # (nworld, nbody)
    dist_threshold: float,                        # qualify when contact_dist < this
    foot_slip_out: wp.array(dtype=float),         # (nworld,) pre-zeroed
):
    """Per-contact foot-slip penalty: when a foot link contacts the
    worldbody, atomic-add the squared horizontal velocity of the foot's CoM
    (cvel[3:5] in world frame) to the world's slot. Targets the failure
    where MPC slides feet through the ground because there is no
    friction-cone awareness in the cost. We use foot CoM velocity rather
    than the contact-point velocity since the angular contribution is
    small for the small ankle_roll body relative to its translation."""
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    if contact_dist[cid] >= dist_threshold:
        return
    pair = contact_geom[cid]
    g1 = pair[0]
    g2 = pair[1]
    if g1 < 0 or g2 < 0:
        return
    b1 = geom_bodyid[g1]
    b2 = geom_bodyid[g2]
    foot_b = int(-1)
    if body_is_foot[b1] == 1 and b2 == 0:
        foot_b = b1
    elif body_is_foot[b2] == 1 and b1 == 0:
        foot_b = b2
    if foot_b < 0:
        return
    wid = contact_worldid[cid]
    cv = cvel[wid, foot_b]
    vx = cv[3]
    vy = cv[4]
    slip_sq = vx * vx + vy * vy
    wp.atomic_add(foot_slip_out, wid, slip_sq)


@wp.kernel
def _foot_impact_kernel(
    contact_geom: wp.array(dtype=wp.vec2i),       # (naconmax,)
    contact_dist: wp.array(dtype=float),          # (naconmax,)
    contact_worldid: wp.array(dtype=wp.int32),    # (naconmax,)
    nacon: wp.array(dtype=wp.int32),              # (1,)
    geom_bodyid: wp.array(dtype=wp.int32),        # (ngeom,)
    body_is_foot: wp.array(dtype=wp.int32),       # (nbody,)
    cvel: wp.array2d(dtype=wp.spatial_vector),    # (nworld, nbody)
    dist_threshold: float,
    foot_impact_out: wp.array(dtype=float),       # (nworld,) pre-zeroed
):
    """Per-contact foot-impact penalty: when a foot link contacts the
    worldbody, atomic-add the squared *downward* component of foot-CoM
    vertical velocity (max(0, -vz))^2 to the world's slot. Approximates
    touchdown impact velocity — the only time vz<0 while in-contact is
    at the no-contact → contact transition; afterwards the foot is
    either pinned (vz≈0) or lifting off (vz>0, contributes 0)."""
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    if contact_dist[cid] >= dist_threshold:
        return
    pair = contact_geom[cid]
    g1 = pair[0]
    g2 = pair[1]
    if g1 < 0 or g2 < 0:
        return
    b1 = geom_bodyid[g1]
    b2 = geom_bodyid[g2]
    foot_b = int(-1)
    if body_is_foot[b1] == 1 and b2 == 0:
        foot_b = b1
    elif body_is_foot[b2] == 1 and b1 == 0:
        foot_b = b2
    if foot_b < 0:
        return
    wid = contact_worldid[cid]
    cv = cvel[wid, foot_b]
    vz = cv[5]
    vz_down = wp.max(0.0, -vz)
    impact_sq = vz_down * vz_down
    wp.atomic_add(foot_impact_out, wid, impact_sq)


@wp.kernel
def _foot_yaw_rate_kernel(
    contact_geom: wp.array(dtype=wp.vec2i),       # (naconmax,)
    contact_dist: wp.array(dtype=float),          # (naconmax,)
    contact_worldid: wp.array(dtype=wp.int32),    # (naconmax,)
    nacon: wp.array(dtype=wp.int32),              # (1,)
    geom_bodyid: wp.array(dtype=wp.int32),        # (ngeom,)
    body_is_foot: wp.array(dtype=wp.int32),       # (nbody,) 1 if foot link
    cvel: wp.array2d(dtype=wp.spatial_vector),    # (nworld, nbody)
    dist_threshold: float,                        # qualify when contact_dist < this
    foot_yaw_rate_out: wp.array(dtype=float),     # (nworld,) pre-zeroed
):
    """Per-contact foot yaw-rate penalty: when a foot link contacts the
    worldbody, atomic-add the squared world-frame yaw angular velocity of the
    foot (cvel[2] = wz) to the world's slot. Targets the planted-foot yaw
    pivot artifact — distinct from foot-slip (linear vx,vy) which a pure yaw
    pivot about the foot barely moves. cvel is [wx, wy, wz, vx, vy, vz], so
    cv[2] is the yaw component. Mirrors _foot_slip_kernel's contact gating."""
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    if contact_dist[cid] >= dist_threshold:
        return
    pair = contact_geom[cid]
    g1 = pair[0]
    g2 = pair[1]
    if g1 < 0 or g2 < 0:
        return
    b1 = geom_bodyid[g1]
    b2 = geom_bodyid[g2]
    foot_b = int(-1)
    if body_is_foot[b1] == 1 and b2 == 0:
        foot_b = b1
    elif body_is_foot[b2] == 1 and b1 == 0:
        foot_b = b2
    if foot_b < 0:
        return
    wid = contact_worldid[cid]
    cv = cvel[wid, foot_b]
    wz = cv[2]
    wp.atomic_add(foot_yaw_rate_out, wid, wz * wz)


@wp.kernel
def _cost_extras_kernel(
    # State reads.
    qpos: wp.array2d(dtype=float),             # (nworld, nq)
    actions_buf: wp.array3d(dtype=float),      # (nworld, H, n_action_dim)
    prev_action: wp.array2d(dtype=float),      # (nworld, nu)  mutated in-place
    dphi_buf: wp.array(dtype=float),           # (nworld,)
    prev_dphi_buf: wp.array(dtype=float),      # (nworld,)  mutated in-place
    qvel: wp.array2d(dtype=float),             # (nworld, nv)
    prev_qvel_buf: wp.array2d(dtype=float),    # (nworld, nv)  mutated in-place
    prev_qpos_buf: wp.array2d(dtype=float),    # (nworld, nu)  mutated in-place
    # Static tables.
    act_qpos_adr: wp.array(dtype=wp.int32),    # (nu,)
    joint_limit_lo: wp.array(dtype=float),     # (nu,) lower bound
    joint_limit_hi: wp.array(dtype=float),     # (nu,) upper bound
    # Output: atomically add to step_rewards[wid, step_t] (negative of cost).
    step_rewards: wp.array2d(dtype=float),     # (nworld, H)
    # Weights.
    w_action_rate: float,
    w_joint_limit: float,
    w_dphi_accel: float,
    w_joint_jerk: float,
    w_lipschitz: float,
    lipschitz_eps: float,
    joint_limit_margin: float,
    # Shapes / constants.
    nu: int,
    nv: int,
    step_t: int,
):
    """isaaclab-style regularizers: action-rate, joint-jerk, phase-accel, joint-limit.

    Runs after `_cost_tracking_kernel` and atomically adjusts the same
    `step_rewards[wid, step_t]` slot. Every weight defaults to 0.0 so the
    kernel is a no-op when none of the regularizers are enabled. Each term
    contributes positively to cost (i.e. lowers step_rewards = -total_cost).
    """
    wid = wp.tid()

    # Action-rate regularizer (on pre-PD joint-target actions in actions_buf).
    # Update prev_action[wid, i] ← actions_buf[wid, step_t, i] after reading.
    cost_action_rate = float(0.0)
    for i in range(nu):
        a_now = actions_buf[wid, step_t, i]
        d_a = a_now - prev_action[wid, i]
        cost_action_rate = cost_action_rate + d_a * d_a
        prev_action[wid, i] = a_now

    # Phase-acceleration regularizer (d^2 phi / dt^2). prev_dphi is zero on
    # rollout start so the first-step diff is the small dphi itself
    # (negligible vs typical action-rate scale). Always update the buffer
    # in place; the weight gating is applied in the assembly below.
    dphi_now = dphi_buf[wid]
    d2phi = dphi_now - prev_dphi_buf[wid]
    cost_dphi_accel = d2phi * d2phi
    prev_dphi_buf[wid] = dphi_now

    # Joint-jerk regularizer: finite-diff (qvel - prev_qvel)^2 over actuated
    # dofs (skipping the 6 floating-base dofs at indices 0..5). Mirrors the
    # prev_action lifecycle — first-step diff is qvel itself (small, since
    # the rollout starts near rest), in-place buffer update each step.
    cost_joint_jerk = float(0.0)
    for j in range(6, nv):
        v_now = qvel[wid, j]
        dv = v_now - prev_qvel_buf[wid, j]
        cost_joint_jerk = cost_joint_jerk + dv * dv
        prev_qvel_buf[wid, j] = v_now

    # Joint-limit soft barrier with inside-margin.
    cost_joint_limit = float(0.0)
    if w_joint_limit > 0.0:
        for i in range(nu):
            qi = qpos[wid, act_qpos_adr[i]]
            lo = joint_limit_lo[i] + joint_limit_margin
            hi = joint_limit_hi[i] - joint_limit_margin
            under = lo - qi
            over = qi - hi
            if under < 0.0:
                under = 0.0
            if over < 0.0:
                over = 0.0
            cost_joint_limit = cost_joint_limit + under * under + over * over

    # State-Lipschitz regularizer: penalize the local Lipschitz constant of the
    # realized controller = ||da||^2 / (||dq||^2 + eps). The numerator is
    # exactly cost_action_rate (||a_t - a_{t-1}||^2), already computed above; the
    # denominator is the squared change in the actuated joint positions over the
    # same step. Small action changes during large motions are cheap; action
    # wiggle while nearly static is costly. prev_qpos_buf mirrors prev_action's
    # lifecycle (zero at rollout start), updated in place only when active. Note
    # prev_action was already overwritten above, so we reuse cost_action_rate
    # rather than re-reading it.
    cost_lipschitz = float(0.0)
    if w_lipschitz > 0.0:
        d_q_sq = float(0.0)
        for i in range(nu):
            q_now = qpos[wid, act_qpos_adr[i]]
            d_q = q_now - prev_qpos_buf[wid, i]
            d_q_sq = d_q_sq + d_q * d_q
            prev_qpos_buf[wid, i] = q_now
        cost_lipschitz = cost_action_rate / (d_q_sq + lipschitz_eps)

    # Assemble the regularizer contribution (all terms add positively to cost).
    total_extras = (
        w_action_rate * cost_action_rate
        + w_joint_limit * cost_joint_limit
        + w_dphi_accel * cost_dphi_accel
        + w_joint_jerk * cost_joint_jerk
        + w_lipschitz * cost_lipschitz
    )
    # step_rewards holds `-total_cost`; subtract extras cost (add its negation).
    step_rewards[wid, step_t] = step_rewards[wid, step_t] - total_extras


@wp.kernel
def _cost_tracking_kernel(
    # mjw.Data fields read directly (mirror reference's _reward_vtrack_kernel pattern).
    xpos: wp.array2d(dtype=wp.vec3),        # (nworld, nbody)
    xquat: wp.array2d(dtype=wp.quat),       # (nworld, nbody) wxyz
    ctrl: wp.array2d(dtype=float),          # (nworld, nu)
    qvel: wp.array2d(dtype=float),          # (nworld, nv)
    body_indices: wp.array(dtype=wp.int32), # (n_links,)
    link_scale: wp.array(dtype=float),      # (n_links,) per-link cost weight (default 1.0)
    # Per-link lever-arm offset in the body frame; zeros for real links.
    # Synthetic links (k >= n_links_real) track xpos[bk] + R(xquat[bk])·offset.
    link_pos_offset: wp.array2d(dtype=float),  # (n_links, 3)
    dphi_buf: wp.array(dtype=float),        # (nworld,)
    # Reference frame (shared across worlds but varies by step via frame_idx)
    ref_pos: wp.array3d(dtype=float),       # (T, n_links, 3)
    ref_quat: wp.array3d(dtype=float),      # (T, n_links, 4)  wxyz
    ref_jvel: wp.array2d(dtype=float),      # (T, nu) joint vel, rad/s at nominal playback
    frame_idx_buf: wp.array(dtype=wp.int32),  # (nworld,)
    frame_alpha_buf: wp.array(dtype=float),   # (nworld,) lerp/slerp fraction in [0, 1]
    T_minus_1: int,
    # Orientation offsets (4, 3) in local frame
    orient_offsets: wp.array2d(dtype=float),
    # Pre-accumulated self-collision cost per world (from _self_collision_kernel)
    self_coll_buf: wp.array(dtype=float),   # (nworld,)
    # Pre-accumulated undesired-contacts count per world (from _undesired_contacts_kernel)
    undesired_buf: wp.array(dtype=float),   # (nworld,)
    # Pre-accumulated foot-slip squared horizontal velocity (from _foot_slip_kernel)
    foot_slip_buf: wp.array(dtype=float),   # (nworld,)
    # Pre-accumulated foot-impact squared downward velocity (from _foot_impact_kernel)
    foot_impact_buf: wp.array(dtype=float), # (nworld,)
    # Pre-accumulated foot yaw-rate squared (from _foot_yaw_rate_kernel)
    foot_yaw_rate_buf: wp.array(dtype=float), # (nworld,)
    # Per-step scratch output
    step_rewards: wp.array2d(dtype=float),
    # Cost weights
    w_pos_relcom: float,
    w_com: float,
    w_com_z_scale: float,
    w_pelvis: float,
    w_pos_relpelvis: float,
    w_orient_pure: float,
    w_orient_offset: float,
    w_energy: float,
    w_joint_vel: float,
    w_joint_vel_ref: float,
    w_self_collision: float,
    w_undesired_contacts: float,
    w_foot_slip: float,
    w_foot_impact: float,
    w_foot_yaw_rate: float,
    w_dphi: float,
    # Ground-clearance penalty for links that must not touch down (knees/shanks).
    # Kinematic, not contact-based: the tracking model collides feet only, so a
    # knee landing is invisible to physics; this charges for the link dropping
    # below `knee_ground_clearance` regardless of collision geometry.
    knee_ground_bodies: wp.array(dtype=wp.int32),  # (n_knee,) body ids, empty = off
    knee_ground_ref_z: wp.array2d(dtype=float),    # (T, n_knee) reference height, phase-indexed
    w_knee_ground: float,
    knee_ground_clearance: float,
    knee_ground_margin: float,     # allowed dip below the reference (m)
    knee_ground_ref_relative: int, # 1 = threshold follows the reference knee
    # Per-link robust-norm shape parameters (length n_links). Each link's
    # smooth_abs2 contributions read its own (p, q) — see
    # cost_link_smooth_abs2_p_scale / _q_scale in the env config.
    smooth_abs2_p_link: wp.array(dtype=float),
    smooth_abs2_q_link: wp.array(dtype=float),
    smooth_abs_p: float,
    cosh_p: float,
    # Per-actuator torque-range span (hi - lo); (nu,)
    torque_range_span: wp.array(dtype=float),
    # Shapes / constants
    n_links: int,
    # Real (body-backed) link count; links k >= n_links_real are synthetic
    # keypoints that contribute to cost_pos_relcom only — excluded from the
    # COM mean and from both orient terms.
    n_links_real: int,
    n_offsets: int,
    nu: int,
    nv: int,
    dphi_star: float,
    early_horizon_alpha: float,
    early_horizon_tau: float,
    step_t: int,
):
    wid = wp.tid()
    fidx = frame_idx_buf[wid]
    alpha = frame_alpha_buf[wid]
    fidx_n = fidx + 1
    if fidx_n > T_minus_1:
        fidx_n = T_minus_1
    one_minus_alpha = 1.0 - alpha

    # COM (mean of link positions) — reference and current
    ref_com_x = float(0.0); ref_com_y = float(0.0); ref_com_z = float(0.0)
    cur_com_x = float(0.0); cur_com_y = float(0.0); cur_com_z = float(0.0)
    inv_n = 1.0 / float(n_links_real)
    for k in range(n_links_real):
        bk = body_indices[k]
        p = xpos[wid, bk]
        rpx = one_minus_alpha * ref_pos[fidx, k, 0] + alpha * ref_pos[fidx_n, k, 0]
        rpy = one_minus_alpha * ref_pos[fidx, k, 1] + alpha * ref_pos[fidx_n, k, 1]
        rpz = one_minus_alpha * ref_pos[fidx, k, 2] + alpha * ref_pos[fidx_n, k, 2]
        ref_com_x = ref_com_x + rpx
        ref_com_y = ref_com_y + rpy
        ref_com_z = ref_com_z + rpz
        cur_com_x = cur_com_x + p[0]
        cur_com_y = cur_com_y + p[1]
        cur_com_z = cur_com_z + p[2]
    ref_com_x = ref_com_x * inv_n; ref_com_y = ref_com_y * inv_n; ref_com_z = ref_com_z * inv_n
    cur_com_x = cur_com_x * inv_n; cur_com_y = cur_com_y * inv_n; cur_com_z = cur_com_z * inv_n

    # Pelvis (root link, index 0) world position — reference (lerped) and current.
    # Shared by cost_pos_relpelvis (pose relative to the pelvis/root frame) and
    # cost_pelvis (absolute pelvis position). MuJoCo places the floating base
    # first, so body_indices[0] / ref_pos[:, 0] are the pelvis.
    p_pel = xpos[wid, body_indices[0]]
    cur_pel_x = p_pel[0]; cur_pel_y = p_pel[1]; cur_pel_z = p_pel[2]
    ref_pel_x = one_minus_alpha * ref_pos[fidx, 0, 0] + alpha * ref_pos[fidx_n, 0, 0]
    ref_pel_y = one_minus_alpha * ref_pos[fidx, 0, 1] + alpha * ref_pos[fidx_n, 0, 1]
    ref_pel_z = one_minus_alpha * ref_pos[fidx, 0, 2] + alpha * ref_pos[fidx_n, 0, 2]

    cost_pos_relcom = float(0.0)
    cost_pos_relpelvis = float(0.0)
    cost_orient_pure = float(0.0)
    cost_orient_offset = float(0.0)
    for k in range(n_links):
        bk = body_indices[k]
        ws = link_scale[k]
        sp = smooth_abs2_p_link[k]
        sq = smooth_abs2_q_link[k]
        p = xpos[wid, bk]
        q = xquat[wid, bk]
        px = p[0]
        py = p[1]
        pz = p[2]
        # Synthetic keypoints (k >= n_links_real) live at a body-frame
        # lever-arm off their parent body.
        if k >= n_links_real:
            r_off = _rotate_vec_wxyz(
                q[0], q[1], q[2], q[3],
                link_pos_offset[k, 0], link_pos_offset[k, 1], link_pos_offset[k, 2],
            )
            px = px + r_off[0]
            py = py + r_off[1]
            pz = pz + r_off[2]
        # Lerp the reference link position between adjacent frames so the
        # cost gradient stays smooth across phi advances that don't land on
        # an integer frame boundary.
        rpx = one_minus_alpha * ref_pos[fidx, k, 0] + alpha * ref_pos[fidx_n, k, 0]
        rpy = one_minus_alpha * ref_pos[fidx, k, 1] + alpha * ref_pos[fidx_n, k, 1]
        rpz = one_minus_alpha * ref_pos[fidx, k, 2] + alpha * ref_pos[fidx_n, k, 2]
        # COM-relative position error, summed via smooth_abs2 per-element.
        dx = (px - cur_com_x) - (rpx - ref_com_x)
        dy = (py - cur_com_y) - (rpy - ref_com_y)
        dz = (pz - cur_com_z) - (rpz - ref_com_z)
        cost_pos_relcom = cost_pos_relcom + ws * (
            _smooth_abs2(dx, sp, sq)
            + _smooth_abs2(dy, sp, sq)
            + _smooth_abs2(dz, sp, sq)
        )
        # Pelvis-relative position error: identical to cost_pos_relcom but anchored
        # to the pelvis/root instead of the keypoint centroid.
        pdx = (px - cur_pel_x) - (rpx - ref_pel_x)
        pdy = (py - cur_pel_y) - (rpy - ref_pel_y)
        pdz = (pz - cur_pel_z) - (rpz - ref_pel_z)
        cost_pos_relpelvis = cost_pos_relpelvis + ws * (
            _smooth_abs2(pdx, sp, sq)
            + _smooth_abs2(pdy, sp, sq)
            + _smooth_abs2(pdz, sp, sq)
        )

        # Orientation via rotated offset vectors, summed via smooth_abs2
        # per-element. Synthetic keypoints share their parent's quaternion, so
        # orient terms on them would just double the parent's — skip them.
        if k < n_links_real:
            cq0 = q[0]; cq1 = q[1]
            cq2 = q[2]; cq3 = q[3]
            rq = _slerp_wxyz(
                ref_quat[fidx, k, 0], ref_quat[fidx, k, 1],
                ref_quat[fidx, k, 2], ref_quat[fidx, k, 3],
                ref_quat[fidx_n, k, 0], ref_quat[fidx_n, k, 1],
                ref_quat[fidx_n, k, 2], ref_quat[fidx_n, k, 3],
                alpha,
            )
            rq0 = rq[0]; rq1 = rq[1]; rq2 = rq[2]; rq3 = rq[3]
            cn = wp.sqrt(cq0 * cq0 + cq1 * cq1 + cq2 * cq2 + cq3 * cq3) + NORM_EPS
            cq0 = cq0 / cn; cq1 = cq1 / cn; cq2 = cq2 / cn; cq3 = cq3 / cn
            for j in range(n_offsets):
                ox = orient_offsets[j, 0]; oy = orient_offsets[j, 1]; oz = orient_offsets[j, 2]
                cr = _rotate_vec_wxyz(cq0, cq1, cq2, cq3, ox, oy, oz)
                rr = _rotate_vec_wxyz(rq0, rq1, rq2, rq3, ox, oy, oz)
                # cost_orient_pure: rotated-offset error only (orientation, no position).
                ex = cr[0] - rr[0]; ey = cr[1] - rr[1]; ez = cr[2] - rr[2]
                cost_orient_pure = cost_orient_pure + ws * (
                    _smooth_abs2(ex, sp, sq)
                    + _smooth_abs2(ey, sp, sq)
                    + _smooth_abs2(ez, sp, sq)
                )
                # cost_orient_offset: per-link offset point in COM-relative coords.
                # err = (link_pos + R*offset - com)_cur - same_for_ref
                # i.e. smooth_abs2 on (cur_offset_relcom - ref_offset_relcom).
                cox = (p[0] + cr[0] - cur_com_x) - (rpx + rr[0] - ref_com_x)
                coy = (p[1] + cr[1] - cur_com_y) - (rpy + rr[1] - ref_com_y)
                coz = (p[2] + cr[2] - cur_com_z) - (rpz + rr[2] - ref_com_z)
                cost_orient_offset = cost_orient_offset + ws * (
                    _smooth_abs2(cox, sp, sq)
                    + _smooth_abs2(coy, sp, sq)
                    + _smooth_abs2(coz, sp, sq)
                )

    # COM position error via per-element smooth_abs, summed over the 3
    # components.
    dcx = cur_com_x - ref_com_x
    dcy = cur_com_y - ref_com_y
    dcz = cur_com_z - ref_com_z
    cost_com = (
        _smooth_abs(dcx, smooth_abs_p)
        + _smooth_abs(dcy, smooth_abs_p)
        + w_com_z_scale * _smooth_abs(dcz, smooth_abs_p)
    )

    # Pelvis position error — a pelvis-anchored analog of cost_com that tracks the
    # root link's absolute world position directly (vs cost_com's keypoint
    # centroid). Reuses the pelvis position computed above. Same per-element
    # smooth_abs and w_com_z_scale z-axis treatment as cost_com. Additive to cost_com.
    dpx = cur_pel_x - ref_pel_x
    dpy = cur_pel_y - ref_pel_y
    dpz = cur_pel_z - ref_pel_z
    cost_pelvis = (
        _smooth_abs(dpx, smooth_abs_p)
        + _smooth_abs(dpy, smooth_abs_p)
        + w_com_z_scale * _smooth_abs(dpz, smooth_abs_p)
    )

    # Control energy via cosh_norm on per-DOF normalized torque.
    cost_energy = float(0.0)
    for i in range(nu):
        c = ctrl[wid, i] / torque_range_span[i]
        cost_energy = cost_energy + _cosh_term(c, cosh_p)

    # Joint-velocity penalty: 0.5 * ||qvel[6:]||^2 (matches quadratic_norm).
    cost_joint_vel = float(0.0)
    for i in range(6, nv):
        v = qvel[wid, i]
        cost_joint_vel = cost_joint_vel + v * v
    cost_joint_vel = 0.5 * cost_joint_vel

    # Reference joint-velocity tracking: 0.5 * ||qvel[6:] - vref*rate||^2.
    # vref is lerped between reference frames like ref_pos; rate = dphi/dphi_star
    # rescales the nominal-playback velocity to the sampled TMR phase speed.
    cost_joint_vel_ref = float(0.0)
    if w_joint_vel_ref > 0.0:
        rate = dphi_buf[wid] / dphi_star
        for i in range(nu):
            vref = one_minus_alpha * ref_jvel[fidx, i] + alpha * ref_jvel[fidx_n, i]
            dv = qvel[wid, 6 + i] - vref * rate
            cost_joint_vel_ref = cost_joint_vel_ref + dv * dv
        cost_joint_vel_ref = 0.5 * cost_joint_vel_ref

    # Phase regularity (log-ratio squared).
    dphi_val = dphi_buf[wid]
    ratio = wp.min(wp.max(dphi_val / dphi_star, MIN_RATE_RATIO), 1.0) + NORM_EPS
    lr = wp.log(ratio)
    cost_dphi = lr * lr

    # Self-collision already accumulated into self_coll_buf[wid] by
    # _self_collision_kernel (per-contact atomic add).
    cost_self = self_coll_buf[wid]
    # Undesired-contacts count already accumulated into undesired_buf[wid] by
    # _undesired_contacts_kernel (per-contact atomic +1).
    cost_undesired = undesired_buf[wid]
    # Foot-slip squared horizontal velocity already accumulated into
    # foot_slip_buf[wid] by _foot_slip_kernel (per foot-vs-ground contact).
    cost_foot_slip = foot_slip_buf[wid]
    # Foot-impact squared downward velocity at touchdown already accumulated
    # into foot_impact_buf[wid] by _foot_impact_kernel.
    cost_foot_impact = foot_impact_buf[wid]
    # Foot yaw-rate squared in contact already accumulated into
    # foot_yaw_rate_buf[wid] by _foot_yaw_rate_kernel.
    cost_foot_yaw_rate = foot_yaw_rate_buf[wid]

    # Squared shortfall below the clearance, summed over the guarded links.
    cost_knee_ground = float(0.0)
    if w_knee_ground > 0.0:
        f0 = frame_idx_buf[wid]
        f1 = wp.min(f0 + 1, T_minus_1)
        a_k = frame_alpha_buf[wid]
        for ki in range(knee_ground_bodies.shape[0]):
            bk_knee = knee_ground_bodies[ki]
            thr = knee_ground_clearance
            if knee_ground_ref_relative == 1:
                # the reference's own knee height at this phase, minus the
                # allowed margin: charges only for going LOWER than the source
                ref_z = (1.0 - a_k) * knee_ground_ref_z[f0, ki] + a_k * knee_ground_ref_z[f1, ki]
                # Capped at the absolute clearance: the guard is about the FLOOR,
                # so it must go silent when the reference knee is airborne —
                # uncapped, phase mismatch during flight turns this into a
                # jump-height-matching term (measured: +30 cm apex overshoot).
                thr = wp.min(ref_z - knee_ground_margin, knee_ground_clearance)
            gap = thr - xpos[wid, bk_knee][2]
            if gap > 0.0:
                cost_knee_ground += gap * gap

    total_cost = (
        w_pos_relcom * cost_pos_relcom
        + w_com * cost_com
        + w_pelvis * cost_pelvis
        + w_pos_relpelvis * cost_pos_relpelvis
        + w_orient_pure * cost_orient_pure
        + w_orient_offset * cost_orient_offset
        + w_energy * cost_energy
        + w_joint_vel * cost_joint_vel
        + w_joint_vel_ref * cost_joint_vel_ref
        + w_self_collision * cost_self
        + w_undesired_contacts * cost_undesired
        + w_foot_slip * cost_foot_slip
        + w_foot_impact * cost_foot_impact
        + w_foot_yaw_rate * cost_foot_yaw_rate
        + w_dphi * cost_dphi
        + w_knee_ground * cost_knee_ground
    )
    # Early-horizon weight: alpha=1 (default) leaves eh_w==1 (no-op).
    eh_w = 1.0 + (early_horizon_alpha - 1.0) * wp.exp(
        -float(step_t) / early_horizon_tau
    )
    step_rewards[wid, step_t] = -total_cost * eh_w


@wp.kernel
def _mean_reward_kernel(
    step_rewards: wp.array2d(dtype=float),
    mean_rewards: wp.array(dtype=float),
    hsample: int,
):
    wid = wp.tid()
    total = float(0.0)
    for t in range(hsample):
        total = total + step_rewards[wid, t]
    mean_rewards[wid] = total / float(hsample)


@wp.kernel
def _phase_update_kernel(
    rdphi_buf: wp.array(dtype=float),          # (nworld,) — last action dim
    phi_buf: wp.array(dtype=float),            # (nworld,)
    dphi_buf_out: wp.array(dtype=float),       # (nworld,)
    frame_idx_out: wp.array(dtype=wp.int32),   # (nworld,)
    frame_alpha_out: wp.array(dtype=float),    # (nworld,) fraction in [0, 1]
    dt: float,
    motion_dur: float,
    dphi_exp_base: float,
    T_minus_1: int,
):
    """Mirrors env.act2dphi_new3: dphi = n**rdphi * dt / motion_dur (no upper clip).

    Also emits the fractional alpha within the [fidx, fidx+1] interval so the
    cost kernels can lerp/slerp between adjacent reference frames instead of
    snapping to the floored frame.
    """
    wid = wp.tid()
    rdphi = rdphi_buf[wid]
    dphi = wp.pow(dphi_exp_base, rdphi) * dt / motion_dur
    dphi = wp.max(dphi, 0.0)
    phi_new = phi_buf[wid] + dphi
    if phi_new > 1.0:
        phi_new = 1.0
    phi_buf[wid] = phi_new
    dphi_buf_out[wid] = dphi
    f_cont = phi_new * float(T_minus_1 + 1)
    fidx = int(wp.floor(f_cont))
    alpha = f_cont - float(fidx)
    if fidx < 0:
        fidx = 0
        alpha = 0.0
    if fidx > T_minus_1:
        fidx = T_minus_1
        alpha = 0.0
    frame_idx_out[wid] = fidx
    frame_alpha_out[wid] = alpha



@wp.kernel
def _flight_dphi_cost_kernel(
    ref_airborne: wp.array(dtype=float),       # (T,) 1.0 = reference airborne at frame
    frame_idx: wp.array(dtype=wp.int32),       # (nworld,) phase-warped frame
    dphi_buf: wp.array(dtype=float),           # (nworld,) phase step this tick
    step_rewards: wp.array2d(dtype=float),     # (nworld, H)
    w_flight_dphi: float,
    dphi_star: float,
    step_t: int,
):
    """Flight-phase playback-rate pin.

    While the REFERENCE frame being tracked is airborne, charge
    w_flight_dphi * (dphi/dphi_star - 1)^2 so the optimizer cannot slow down
    (or rush) the reference through a ballistic window it must fly in real
    time. Launched after the tracking + extras kernels, so it subtracts from
    the already-written reward slot; one thread per world, no atomics needed.
    """
    wid = wp.tid()
    if ref_airborne[frame_idx[wid]] > 0.5:
        rate_err = dphi_buf[wid] / dphi_star - 1.0
        step_rewards[wid, step_t] = (
            step_rewards[wid, step_t] - w_flight_dphi * rate_err * rate_err
        )
