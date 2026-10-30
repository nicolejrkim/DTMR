"""TrackingEnvWarp: the warp-backed parallel tracking environment.

Owns the MuJoCo + MuJoCo-Warp model, batched mjw.Data, and all kernel launchers.
Standalone (no base-class env); the PD control law is a
Warp kernel (see kernels.py)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import mujoco
import mujoco_warp as mjw
import warp as wp

from .np_helpers import _compute_joint_range_from_qpos
from dtmr.utils import get_model_path
from dtmr.utils import load_link_poses_npz, load_reference_state

from .config import TrackingEnvConfig
from .np_helpers import (
    _quat_rotate_np,
)
from .kernels import (
    _broadcast_state_kernel,
    _compute_ctrl_kernel,
    _self_collision_kernel,
    _undesired_contacts_kernel,
    _foot_slip_kernel,
    _foot_impact_kernel,
    _foot_yaw_rate_kernel,
    _cost_extras_kernel,
    _cost_tracking_kernel,
    _phase_update_kernel,
    _flight_dphi_cost_kernel,
)


# Lever arm of the four body-frame offset points the orientation cost compares (metres).
ORIENT_LEVER_ARM_M = 0.1
# Contact and constraint capacity per world of a data buffer made without explicit sizes
# (the single execution world). The controller allocates its rollout worlds larger.
EXEC_NCONMAX = 60
EXEC_NJMAX = 240


class TrackingEnvWarp:
    """Warp-backed parallel environment for temporal-optimization motion tracking.

    Robot-generic: the robot comes from its descriptor (model, gains, body
    lists, joint groups). Owns `mj_model` / `mjw_model`,
    makes batched `mjw.Data` via `make_data(nworld)`, exposes kernel launchers
    for broadcast / control / cost. The outer controller drives the diffusion
    loop and the per-step phase update.
    """

    def __init__(self, config: TrackingEnvConfig):
        """Build the env in ordered setup stages (see the _setup_* helpers).

        Order matters: later stages read attributes set by earlier ones (e.g.
        _setup_joint_range needs the reference state from _setup_reference and
        nu from _setup_model; _setup_link_scales needs n_links from
        _setup_tracked_links). Do not reorder the calls below.
        """
        self.config = config
        self._setup_model(config)
        self._setup_actuators(config)
        self._setup_contact_masks(config)
        self._setup_reference(config)
        self._setup_joint_range(config)
        self._setup_orient_offsets()
        self._setup_cost_weights(config)
        self._setup_phase(config)
        self._setup_flight_mask(config)
        self._setup_tracked_links(config)
        self._setup_link_scales(config)

    def _setup_model(self, config: TrackingEnvConfig) -> None:
        """Load the MJCF + MuJoCo-Warp model; resolve dims, timestep, control mode."""
        if not config.model_dir:
            raise ValueError(
                "model_dir is not set: select a robot with `robot: <name>` in the config "
                "(see src/dtmr/robots/) or give model_dir and model_file explicitly"
            )
        model_path = get_model_path(config.model_dir, config.model_file)
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_model.opt.timestep = config.timestep
        self.mjw_model = mjw.put_model(self.mj_model)
        try:
            self.mjw_model.opt.ls_parallel = True
        except Exception:
            pass

        self.dt = config.dt
        self.timestep = config.timestep
        self.n_frames = max(1, int(round(config.dt / config.timestep)))
        self.action_scale = config.action_scale
        self.inner_rate_pd = bool(config.inner_rate_pd)
        self.nq = int(self.mj_model.nq)
        self.nv = int(self.mj_model.nv)
        self.nu = int(self.mj_model.nu)

    def _setup_actuators(self, config: TrackingEnvConfig) -> None:
        """Actuator adr lookups, kp/kd, and torque/physical joint ranges."""
        # Actuator → qpos/qvel adr lookups.
        act_jid = self.mj_model.actuator_trnid[:, 0]
        self.actuator_joint_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in act_jid
        ]
        self.joint_groups = {str(k): list(v) for k, v in dict(config.joint_groups).items()}
        self.act_qpos_adr_np = np.array(self.mj_model.jnt_qposadr[act_jid], dtype=np.int32)
        self.act_dof_adr_np = np.array(self.mj_model.jnt_dofadr[act_jid], dtype=np.int32)
        joint_range_np = np.array(self.mj_model.jnt_range[act_jid], dtype=np.float32)
        torque_range_np = np.array(self.mj_model.actuator_ctrlrange, dtype=np.float32)

        kp_np = np.asarray(config.kp, dtype=np.float32)
        kd_np = np.asarray(config.kd, dtype=np.float32)
        if kp_np.size != self.nu:
            raise ValueError(f"kp has {kp_np.size} entries but model has nu={self.nu}")
        if kd_np.size != self.nu:
            raise ValueError(f"kd has {kd_np.size} entries but model has nu={self.nu}")

        self.kp_wp = wp.array(kp_np, dtype=float)
        self.kd_wp = wp.array(kd_np, dtype=float)
        self.act_qpos_adr_wp = wp.array(self.act_qpos_adr_np, dtype=wp.int32)
        self.act_dof_adr_wp = wp.array(self.act_dof_adr_np, dtype=wp.int32)
        # Physical range from the MJCF. Always kept so we can still clip to it
        # when auto_joint_range narrows the command range.
        self.physical_joint_range = joint_range_np.copy()
        self.torque_range_lo_wp = wp.array(torque_range_np[:, 0].copy(), dtype=float)
        self.torque_range_hi_wp = wp.array(torque_range_np[:, 1].copy(), dtype=float)
        # Per-actuator torque span (hi - lo), used to normalize control:
        # `ctrl_normalized = ctrl / (torque_hi - torque_lo)`.
        torque_span_np = np.maximum(
            torque_range_np[:, 1] - torque_range_np[:, 0], 1.0,
        ).astype(np.float32)
        self.torque_range_span_wp = wp.array(torque_span_np, dtype=float)
        # Raw MJCF joint range; consumed by _setup_joint_range (which may narrow it).
        self._mjcf_joint_range_np = joint_range_np

    def _setup_contact_masks(self, config: TrackingEnvConfig) -> None:
        """Geom/body masks for self-collision, undesired-contact, and foot kernels."""
        # Self-collision masking lookups (bodyid 0 == worldbody/floor in MJCF).
        geom_bodyid_np = np.asarray(self.mj_model.geom_bodyid, dtype=np.int32)
        self.geom_bodyid_wp = wp.array(geom_bodyid_np, dtype=wp.int32)
        self.is_robot_geom_wp = wp.array(
            (geom_bodyid_np != 0).astype(np.int32), dtype=wp.int32,
        )
        self.ngeom = int(self.mj_model.ngeom)

        # Undesired-contacts body allow-list. World body (id 0) is always
        # allowed — we're not penalizing the floor. All named bodies in
        # `config.undesired_allowed_bodies` are resolved to body ids; unknown
        # names are dropped with a warning so stale YAMLs don't crash the run.
        nbody = int(self.mj_model.nbody)
        allowed_np = np.zeros(nbody, dtype=np.int32)
        allowed_np[0] = 1  # worldbody
        unknown: list[str] = []
        for name in config.undesired_allowed_bodies:
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, str(name))
            if bid < 0:
                unknown.append(str(name))
            else:
                allowed_np[bid] = 1
        if unknown:
            print(f"[warp] undesired_allowed_bodies: unknown body names ignored: {unknown}")
        self.body_is_allowed_wp = wp.array(allowed_np, dtype=wp.int32)

        # Foot-body indicator for the foot-slip kernel. Mirrors body_is_allowed
        # but only marks the named foot links — the kernel also requires the
        # other contact body to be the worldbody (id 0), so this flag is
        # specifically "is this body a foot we want to penalize sliding for".
        is_foot_np = np.zeros(nbody, dtype=np.int32)
        unknown_feet: list[str] = []
        for name in config.foot_bodies:
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, str(name))
            if bid <= 0:
                unknown_feet.append(str(name))
            else:
                is_foot_np[bid] = 1
        if unknown_feet:
            print(f"[warp] foot_bodies: unknown body names ignored: {unknown_feet}")
        self.body_is_foot_wp = wp.array(is_foot_np, dtype=wp.int32)

    def _setup_reference(self, config: TrackingEnvConfig) -> None:
        """Load reference motion + optional state ref; pick the reset pose."""
        # Reference motion: link_pos / link_quat / link_names.
        self._load_motion(config.motion_path)

        # Optional reference qpos/qvel (for controller warm-start) — inferred
        # from the reference npz itself (or, for the older layout, the CSV next to it)
        # the sibling state CSV.
        self._state_ref = self._maybe_load_state_ref(config.motion_path)

        # Reset state: prefer ref[0]
        # so tracking cost isn't dominated by a several-metre root offset on
        # startup. Falls back to the "knees_bent" keyframe if no state ref.
        if self._state_ref is not None:
            self._reset_qpos_np = np.asarray(self._state_ref["qpos"][0], dtype=np.float32)
            self._reset_qvel_np = np.asarray(self._state_ref["qvel"][0], dtype=np.float32)
            if config.reset_ground:
                self._ground_reset_state()
        else:
            try:
                self._reset_qpos_np = np.array(
                    self.mj_model.keyframe("knees_bent").qpos, dtype=np.float32,
                )
            except Exception:
                self._reset_qpos_np = np.array(self.mj_model.qpos0, dtype=np.float32)
            self._reset_qvel_np = np.zeros(self.nv, dtype=np.float32)

    def _ground_reset_state(self) -> None:
        """Lift the RESET pose out of the floor (reference left untouched).

        Measured against this env's own MuJoCo model, so the geometry is exactly
        what the simulator will collide. Only ever lifts; a non-penetrating
        first frame is left as it is.
        """
        try:
            import mujoco
            from dtmr.data.ground_spawn import (
                geom_lowpoints, _lowest_z,
            )
            d = mujoco.MjData(self.mj_model)
            d.qpos[:] = self._reset_qpos_np[: self.mj_model.nq]
            mujoco.mj_forward(self.mj_model, d)
            z = _lowest_z(d, geom_lowpoints(self.mj_model))
            if z < 0.0:
                self._reset_qpos_np[2] += float(-z)
                print(f"[reset-ground] spawn pose was {(-z)*1000:.1f} mm inside the "
                      f"floor - lifted reset qpos by that amount "
                      f"(reference unchanged)")
        except Exception as exc:  # never block a run on the grounding check
            print(f"[reset-ground] skipped ({type(exc).__name__}: {exc})")

    def _setup_joint_range(self, config: TrackingEnvConfig) -> None:
        """Commanded joint range: physical, or a tight band around the reference."""
        joint_range_np = self._mjcf_joint_range_np
        # Commanded joint range: either the physical range or a tight band
        # around the reference qpos (auto_joint_range).
        if config.auto_joint_range:
            if self._state_ref is None or "qpos" not in self._state_ref:
                raise ValueError(
                    "auto_joint_range=True requires a reference state trajectory (qpos)."
                )
            ref_joint_q = np.asarray(
                self._state_ref["qpos"][:, 7:7 + self.nu], dtype=np.float32,
            )
            joint_range_np = _compute_joint_range_from_qpos(
                ref_joint_q,
                margin_abs=float(config.joint_range_margin_abs),
                margin_ratio=float(config.joint_range_margin_ratio),
                round_to=float(config.joint_range_round_to),
                min_width=float(config.joint_range_min_width),
                # Pass physical_range only when clipping is requested; otherwise
                # let the commanded range follow the reference ± margin even
                # outside the HW limits.
                physical_range=(
                    self.physical_joint_range
                    if config.clip_joint_range_to_physical
                    else None
                ),
                # Extend the physical clamp outward so the PD target can sit past the stop
                # (static holding torque); ratio*span + abs, 0 = clamp to hard limits.
                physical_outset_ratio=float(config.joint_range_physical_outset_ratio),
                physical_outset_abs=float(config.joint_range_physical_outset_abs),
            ).astype(np.float32)

        self.joint_range_lo_wp = wp.array(joint_range_np[:, 0].copy(), dtype=float)
        self.joint_range_hi_wp = wp.array(joint_range_np[:, 1].copy(), dtype=float)
        # Host-side (nu, 2) copy for controllers that need joint limits in numpy.
        self.joint_range = joint_range_np.copy()

    def _setup_orient_offsets(self) -> None:
        """Four ±x/±y lever-arm offsets (10 cm) for the orientation cost terms."""
        d = ORIENT_LEVER_ARM_M
        offsets = np.array(
            [[d, 0.0, 0.0], [-d, 0.0, 0.0], [0.0, d, 0.0], [0.0, -d, 0.0]],
            dtype=np.float32,
        )
        self.orient_offsets_wp = wp.array(offsets, dtype=float)
        self.n_offsets = int(offsets.shape[0])

    def _setup_cost_weights(self, config: TrackingEnvConfig) -> None:
        """Copy all cost weights, robust-norm shape params, and BM stds off config."""
        # Cost weights.
        self.w_pos_relcom = float(config.w_pos_relcom)
        self.w_com = float(config.w_com)
        self.w_com_z_scale = float(config.w_com_z_scale)
        self.w_pelvis = float(config.w_pelvis)
        self.w_pos_relpelvis = float(config.w_pos_relpelvis)
        self.w_orient_pure = float(config.w_orient_pure)
        self.w_orient_offset = float(config.w_orient_offset)
        self.w_energy = float(config.w_energy)
        self.w_joint_vel = float(config.w_joint_vel)
        self.w_joint_vel_ref = float(config.w_joint_vel_ref)
        self.w_self_collision = float(config.w_self_collision)
        self.w_dphi = float(config.w_dphi)
        # Regularizer weights.
        self.w_action_rate = float(config.w_action_rate)
        self.w_joint_limit = float(config.w_joint_limit)
        self.w_dphi_accel = float(config.w_dphi_accel)
        self.w_foot_slip = float(config.w_foot_slip)
        self.foot_slip_dist_threshold = float(config.foot_slip_dist_threshold)
        self.w_foot_impact = float(config.w_foot_impact)
        self.w_foot_yaw_rate = float(config.w_foot_yaw_rate)
        self.w_joint_jerk = float(config.w_joint_jerk)
        self.w_lipschitz = float(config.w_lipschitz)
        self.lipschitz_eps = float(config.lipschitz_eps)
        self.early_horizon_alpha = float(config.early_horizon_alpha)
        self.early_horizon_tau = max(float(config.early_horizon_tau), 1e-6)
        # Robust-norm shape params.
        self.smooth_abs2_p = float(config.smooth_abs2_p)
        self.smooth_abs2_q = float(config.smooth_abs2_q)
        self.smooth_abs_p = float(config.smooth_abs_p)
        self.cosh_p = float(config.cosh_p)
        self.self_collision_threshold = float(config.self_collision_threshold)
        self.joint_limit_margin = float(config.joint_limit_margin)
        self.joint_limit_use_physical = bool(config.joint_limit_use_physical)
        self.w_knee_ground = float(config.w_knee_ground)
        self.knee_ground_clearance = float(config.knee_ground_clearance)
        knee_ids = []
        unknown_knee = []
        for name in (config.knee_ground_bodies or ()):
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                unknown_knee.append(name)
            else:
                knee_ids.append(bid)
        if unknown_knee:
            print(f"[warp] knee_ground_bodies: unknown body names ignored: {unknown_knee}")
        self.knee_ground_bodies_wp = wp.array(np.asarray(knee_ids, dtype=np.int32), dtype=wp.int32)
        self.knee_ground_ref_relative = int(bool(config.knee_ground_ref_relative))
        self.knee_ground_margin = float(config.knee_ground_margin)
        # Reference knee height per (frame, guarded link), for the ref-relative
        # threshold. Built after the reference motion is loaded (see below).
        if self.w_knee_ground > 0.0:
            print(f"[warp] knee-ground clearance penalty: w={self.w_knee_ground} "
                  f"clearance={self.knee_ground_clearance:.3f} m over {len(knee_ids)} links")
        self.w_undesired_contacts = float(config.w_undesired_contacts)
        self.undesired_floor_only = int(bool(config.undesired_floor_only))
        self.undesired_contact_dist_threshold = float(config.undesired_contact_dist_threshold)

    def _setup_phase(self, config: TrackingEnvConfig) -> None:
        """Phase constants (T, motion duration, dphi base) + reference body velocities."""
        # Phase constants.
        self.T = int(self._ref_pos_np.shape[0])
        self.motion_dur = (self.T - 1) * (1.0 / config.motion_fps)
        self.dphi_exp_base = float(config.dphi_exp_base)

        # Reference joint velocities for w_joint_vel_ref, frame-aligned with
        # ref_pos. Source: the sibling state CSV's qvel columns (rad/s at
        # nominal playback; the cost kernel rescales by dphi/dphi_star).
        # Zeros when the CSV is missing — the term then degrades to a
        # magnitude penalty, so warn if the weight is on.
        ref_jvel_np = np.zeros((self.T, self.nu), dtype=np.float32)
        if self._state_ref is not None and "qvel" in self._state_ref:
            src = np.asarray(
                self._state_ref["qvel"][:, 6:6 + self.nu], dtype=np.float32,
            )
            n = min(self.T, src.shape[0])
            ref_jvel_np[:n] = src[:n]
            if n < self.T:
                ref_jvel_np[n:] = src[n - 1]
        elif self.w_joint_vel_ref > 0.0:
            print("[env] w_joint_vel_ref > 0 but no state CSV qvel — "
                  "tracking zeros (magnitude penalty).")
        self.ref_jvel_wp = wp.array(ref_jvel_np, dtype=float)

    def _setup_flight_mask(self, config: TrackingEnvConfig) -> None:
        """Reference airborne mask for w_flight_dphi (see config docstring).

        airborne[t] = every flight_feet_links link is above clip-min foot
        height + flight_clearance at frame t, with runs shorter than
        flight_min_frames removed. Inert (all zeros + note) when the weight
        is 0 or the feet links are not in the reference.
        """
        self.w_flight_dphi = float(config.w_flight_dphi)
        air = np.zeros(self.T, dtype=np.float32)
        self._flight_mask_active = False
        if self.w_flight_dphi > 0.0:
            feet = [str(n) for n in config.flight_feet_links]
            if not feet:
                raise ValueError(
                    "w_flight_dphi > 0 needs flight_feet_links (the robot descriptor normally "
                    "provides it)"
                )
            missing = [n for n in feet if n not in self._link_names]
            if missing:
                print(f"[env] w_flight_dphi={self.w_flight_dphi} but reference has no "
                      f"{missing} links -> humanoid flight pin inert")
            else:
                idx = [self._link_names.index(n) for n in feet]
                fz = self._ref_pos_np[:, idx, 2].min(axis=1)
                mask = fz > (float(fz.min()) + float(config.flight_clearance))
                # drop short runs (estimation jitter)
                min_run = int(config.flight_min_frames)
                t = 0
                while t < self.T:
                    if mask[t]:
                        u = t
                        while u < self.T and mask[u]:
                            u += 1
                        if u - t < min_run:
                            mask[t:u] = False
                        t = u
                    else:
                        t += 1
                air = mask.astype(np.float32)
                n_runs = int(np.sum(np.diff(np.concatenate([[0.0], air])) > 0))
                self._flight_mask_active = True
                print(f"[env] w_flight_dphi={self.w_flight_dphi}: {int(air.sum())}/{self.T} "
                      f"reference frames airborne in {n_runs} flight(s) "
                      f"(clearance {float(config.flight_clearance):.2f} m, "
                      f"min_run {min_run}) -> rate pinned to nominal there")
        self.ref_airborne_wp = wp.array(air, dtype=float)

    def _setup_tracked_links(self, config: TrackingEnvConfig) -> None:
        """Resolve tracked-link body indices + optional synthetic head keypoint,
        upload reference arrays to GPU, and build the physical joint-range buffers."""
        # Body indices for the reference links (after model load).
        self.body_indices_np = self._resolve_body_indices(self._link_names)
        self.n_links_real = int(self.body_indices_np.shape[0])

        # Optional synthetic "head" keypoint — appended LAST so kernels can
        # mask it by loop bounds (COM + orient loops run n_links_real, the
        # pos_relcom loop runs n_links). Host-side arrays (_link_names,
        # _ref_pos_np, body_indices_np) stay 30-wide: DTW / phi-tracking /
        # sample-log consumers gather raw xpos[body_indices] and must not see
        # the lever-arm entry.
        track_pos = self._ref_pos_np
        track_quat = self._ref_quat_np
        track_indices = self.body_indices_np
        track_names = list(self._link_names)
        link_pos_offset_np = np.zeros((self.n_links_real, 3), dtype=np.float32)
        if config.head_keypoint_offset is not None:
            off = np.asarray(config.head_keypoint_offset, dtype=np.float32).reshape(3)
            if "head" in track_names:
                raise ValueError(
                    "head_keypoint_offset set but the reference already has a 'head' link"
                )
            if str(config.head_keypoint_parent) not in track_names:
                raise ValueError(
                    f"head_keypoint_offset is set but head_keypoint_parent "
                    f"{config.head_keypoint_parent!r} is not a link of the reference"
                )
            parent_k = track_names.index(str(config.head_keypoint_parent))
            parent_bid = int(self.body_indices_np[parent_k])
            head_pos = self._ref_pos_np[:, parent_k] + _quat_rotate_np(
                self._ref_quat_np[:, parent_k], off
            )
            track_pos = np.concatenate([track_pos, head_pos[:, None, :]], axis=1)
            # Head ref quat = parent's; never read (orient terms skip
            # k >= n_links_real) but keeps ref_quat rectangular.
            track_quat = np.concatenate(
                [track_quat, self._ref_quat_np[:, parent_k : parent_k + 1]], axis=1
            )
            track_indices = np.concatenate(
                [track_indices, np.array([parent_bid], dtype=np.int32)]
            )
            track_names.append("head")
            link_pos_offset_np = np.concatenate([link_pos_offset_np, off[None]], axis=0)
        self._track_link_names = tuple(track_names)
        self.n_links = len(track_names)
        self.body_indices_wp = wp.array(track_indices.astype(np.int32), dtype=wp.int32)
        self.link_pos_offset_wp = wp.array(link_pos_offset_np, dtype=float)

        # Reference arrays on GPU (with the synthetic head row if present).
        # Reference height of each guarded knee link, phase-indexed (T, n_knee).
        _knee_names = tuple(config.knee_ground_bodies or ())
        _knee_ref = np.zeros((self.T, max(len(_knee_names), 1)), dtype=np.float32)
        for _ki, _n in enumerate(_knee_names):
            if _n in self._link_names:
                _knee_ref[:, _ki] = self._ref_pos_np[:, list(self._link_names).index(_n), 2]
            else:
                # not in the reference link set: fall back to the absolute clearance
                _knee_ref[:, _ki] = self.knee_ground_clearance
                print(f"[warp] knee_ground: {_n} absent from the reference links; "
                      f"using the absolute clearance for it")
        self.knee_ground_ref_z_wp = wp.array(np.ascontiguousarray(_knee_ref), dtype=float)
        if self.w_knee_ground > 0.0 and self.knee_ground_ref_relative:
            print(f"[warp] knee-ground penalty is REFERENCE-RELATIVE "
                  f"(margin {self.knee_ground_margin*100:.1f} cm); reference knee min "
                  f"{_knee_ref.min():.3f} m")
        self.ref_pos_wp = wp.array(np.ascontiguousarray(track_pos), dtype=float)
        self.ref_quat_wp = wp.array(np.ascontiguousarray(track_quat), dtype=float)
        # Joint-limit penalty range source (commanded by default). A per-axis
        # (nu, 2) array is kept on device for convenience; swap to physical at
        # kernel-launch time via `joint_limit_use_physical`.
        self.physical_joint_range_lo_wp = wp.array(
            self.physical_joint_range[:, 0].copy(), dtype=float,
        )
        self.physical_joint_range_hi_wp = wp.array(
            self.physical_joint_range[:, 1].copy(), dtype=float,
        )

    def _setup_link_scales(self, config: TrackingEnvConfig) -> None:
        """Per-link cost-weight scale and per-link smooth_abs2 shape (p, q) tables."""
        # Per-link cost weights — defaults to 1.0 for every tracked link, with
        # named overrides from config.cost_link_scale. Unknown link names log
        # a warning so stale YAMLs don't crash the run.
        link_scale_np = np.ones(self.n_links, dtype=np.float32)
        unknown: list[str] = []
        for name, scale in (config.cost_link_scale or {}).items():
            try:
                idx = self._track_link_names.index(str(name))
            except ValueError:
                unknown.append(str(name))
                continue
            link_scale_np[idx] = float(scale)
        if unknown:
            print(f"[warp] cost_link_scale: unknown link names ignored: {unknown}")
        self.link_scale_wp = wp.array(link_scale_np, dtype=float)

        # Per-link smooth_abs2 shape parameters — global (p, q) by default,
        # multiplied per link by config.cost_link_smooth_abs2_p_scale and
        # ..._q_scale. Use this to give specific links (e.g. ankles) a wider
        # dead-zone or sharper transition than the rest of the body.
        smooth_abs2_p_link_np = np.full(self.n_links, self.smooth_abs2_p, dtype=np.float32)
        smooth_abs2_q_link_np = np.full(self.n_links, self.smooth_abs2_q, dtype=np.float32)
        for label, scale_dict, arr in (
            ("smooth_abs2_p", config.cost_link_smooth_abs2_p_scale, smooth_abs2_p_link_np),
            ("smooth_abs2_q", config.cost_link_smooth_abs2_q_scale, smooth_abs2_q_link_np),
        ):
            unknown_shape: list[str] = []
            for name, mult in (scale_dict or {}).items():
                try:
                    idx = self._track_link_names.index(str(name))
                except ValueError:
                    unknown_shape.append(str(name))
                    continue
                arr[idx] = arr[idx] * float(mult)
            if unknown_shape:
                print(f"[warp] cost_link_{label}_scale: unknown link names ignored: {unknown_shape}")
        self.smooth_abs2_p_link_wp = wp.array(smooth_abs2_p_link_np, dtype=float)
        self.smooth_abs2_q_link_wp = wp.array(smooth_abs2_q_link_np, dtype=float)


    # ----- Loading / resolution -----

    def _load_motion(self, motion_path: str) -> None:
        """Load the link world poses (pos, rot wxyz, link_names) of the reference npz."""
        pos, quat, names, _name_to_idx, _time_idx = load_link_poses_npz(motion_path)
        self._ref_pos_np = np.asarray(pos, dtype=np.float32)
        self._ref_quat_np = np.asarray(quat, dtype=np.float32)
        self._link_names = tuple(str(n) for n in names.tolist())

    def _maybe_load_state_ref(self, motion_path: str) -> Optional[dict]:
        """Reference qpos/qvel: from the reference npz (keys qpos, qvel), or for the
        older two-file layout from the state CSV next to it.

        Returns None if there is no state (controllers fall back to zero init).
        """
        if motion_path is None:
            return None
        return load_reference_state(motion_path, self.nq, self.nv)

    def _resolve_body_indices(self, names: Sequence[str]) -> np.ndarray:
        """Map reference link names to MuJoCo body indices.

        Every name must be a body of the tracking model: an unknown name would
        otherwise track against the world origin.
        """
        idxs = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, n)
                for n in names]
        missing = [n for n, bid in zip(names, idxs) if bid <= 0]
        if missing:
            raise ValueError(
                f"reference links not in the tracking model: {missing} "
                f"(is the reference for this robot?)"
            )
        return np.array(idxs, dtype=np.int32)

    # ----- Data construction -----

    def joint_indices(self, *groups: str) -> np.ndarray:
        """Action indices of the joints in the named groups of the robot descriptor.

        Groups are lists of case-insensitive glob patterns on joint names, so the
        same knob addresses the right joints on every robot.
        """
        import fnmatch

        picked = set()
        for group in groups:
            if group not in self.joint_groups:
                raise KeyError(
                    f"joint group {group!r} is not defined for this robot "
                    f"(known: {sorted(self.joint_groups)}); add it under joint_groups in "
                    f"the robot descriptor"
                )
            patterns = [str(p).lower() for p in self.joint_groups[group]]
            picked.update(
                i for i, name in enumerate(self.actuator_joint_names)
                if any(fnmatch.fnmatchcase(name.lower(), p) for p in patterns)
            )
        return np.array(sorted(picked), dtype=np.int64)

    def make_data(self, nworld: int = 1, nconmax: int = EXEC_NCONMAX,
                  njmax: int = EXEC_NJMAX) -> mjw.Data:
        return mjw.make_data(self.mj_model, nworld=nworld, nconmax=nconmax, njmax=njmax)

    def reset_data(self, d: mjw.Data) -> None:
        mjw.reset_data(self.mjw_model, d)

    def broadcast_reset_state(self, d: mjw.Data) -> None:
        """Set all worlds' qpos/qvel to the (single-world) reset state, then forward."""
        src_q = wp.array(self._reset_qpos_np, dtype=float)
        src_v = wp.array(self._reset_qvel_np, dtype=float)
        wp.launch(
            _broadcast_state_kernel, dim=d.nworld,
            inputs=[src_q, src_v, d.qpos, d.qvel, self.nq, self.nv],
        )
        mjw.forward(self.mjw_model, d)

    def broadcast_state(self, d: mjw.Data, qpos_np: np.ndarray, qvel_np: np.ndarray) -> None:
        src_q = wp.array(qpos_np.astype(np.float32), dtype=float)
        src_v = wp.array(qvel_np.astype(np.float32), dtype=float)
        wp.launch(
            _broadcast_state_kernel, dim=d.nworld,
            inputs=[src_q, src_v, d.qpos, d.qvel, self.nq, self.nv],
        )

    # ----- Kernel launchers -----

    def launch_ctrl(self, d: mjw.Data, actions_buf_wp: wp.array, step_t: int) -> None:
        wp.launch(
            _compute_ctrl_kernel, dim=d.nworld,
            inputs=[
                d.qpos, d.qvel, actions_buf_wp, d.ctrl, step_t,
                self.nu, self.action_scale,
                self.act_qpos_adr_wp, self.act_dof_adr_wp,
                self.kp_wp, self.kd_wp,
                self.joint_range_lo_wp, self.joint_range_hi_wp,
                self.torque_range_lo_wp, self.torque_range_hi_wp,
            ],
        )

    def launch_phase_update(
        self,
        rdphi_wp: wp.array,
        phi_wp: wp.array,
        dphi_out_wp: wp.array,
        frame_idx_out_wp: wp.array,
        frame_alpha_out_wp: wp.array,
    ) -> None:
        wp.launch(
            _phase_update_kernel, dim=rdphi_wp.shape[0],
            inputs=[rdphi_wp, phi_wp, dphi_out_wp, frame_idx_out_wp,
                    frame_alpha_out_wp,
                    self.dt, self.motion_dur, self.dphi_exp_base, self.T - 1],
        )

    def launch_cost(
        self,
        d: mjw.Data,
        dphi_wp: wp.array,
        frame_idx_wp: wp.array,
        frame_alpha_wp: wp.array,
        step_rewards_wp: wp.array,
        step_t: int,
        self_coll_buf_wp: Optional[wp.array] = None,
        undesired_buf_wp: Optional[wp.array] = None,
        foot_slip_buf_wp: Optional[wp.array] = None,
        foot_impact_buf_wp: Optional[wp.array] = None,
        foot_yaw_rate_buf_wp: Optional[wp.array] = None,
        actions_buf_wp: Optional[wp.array] = None,
        prev_action_wp: Optional[wp.array] = None,
        prev_dphi_wp: Optional[wp.array] = None,
        prev_qvel_wp: Optional[wp.array] = None,
        prev_qpos_wp: Optional[wp.array] = None,
    ) -> None:
        """Launch the tracking cost for one step.

        `self_coll_buf_wp` is a scratch `wp.zeros(nworld, dtype=float)` owned by
        the controller. It is zeroed here, populated by
        `_self_collision_kernel` when `w_self_collision > 0`, then consumed by
        `_cost_tracking_kernel`. Callers that previously passed nothing keep
        working: we auto-allocate a zero-buffer if None is given, so no self-
        collision cost is applied. The same lifecycle applies to
        `undesired_buf_wp` and `foot_slip_buf_wp`.

        When `actions_buf_wp` + `prev_action_wp` are supplied, the BM-aligned
        extras kernel runs after the main tracking kernel to apply any enabled
        exp-form tracking rewards, the action-rate regularizer (needs
        `prev_action_wp`), the joint-limit soft barrier, and the
        phase-acceleration regularizer (needs `prev_dphi_wp`). If
        `prev_dphi_wp` is omitted, an auto-allocated zeros buffer is used so
        the term is a per-step diff against zero each call.
        """
        dphi_star = self.dt / self.motion_dur
        if self_coll_buf_wp is None:
            self_coll_buf_wp = wp.zeros(d.nworld, dtype=float)
        else:
            self_coll_buf_wp.zero_()
        if undesired_buf_wp is None:
            undesired_buf_wp = wp.zeros(d.nworld, dtype=float)
        else:
            undesired_buf_wp.zero_()
        if foot_slip_buf_wp is None:
            foot_slip_buf_wp = wp.zeros(d.nworld, dtype=float)
        else:
            foot_slip_buf_wp.zero_()
        if foot_impact_buf_wp is None:
            foot_impact_buf_wp = wp.zeros(d.nworld, dtype=float)
        else:
            foot_impact_buf_wp.zero_()
        if foot_yaw_rate_buf_wp is None:
            foot_yaw_rate_buf_wp = wp.zeros(d.nworld, dtype=float)
        else:
            foot_yaw_rate_buf_wp.zero_()

        naconmax = int(d.contact.dist.shape[0])
        if self.w_self_collision > 0.0:
            wp.launch(
                _self_collision_kernel, dim=naconmax,
                inputs=[
                    d.contact.geom, d.contact.dist, d.contact.worldid,
                    d.nacon, self.is_robot_geom_wp, self.geom_bodyid_wp,
                    self.self_collision_threshold, self_coll_buf_wp,
                ],
            )
        if self.w_undesired_contacts > 0.0:
            wp.launch(
                _undesired_contacts_kernel, dim=naconmax,
                inputs=[
                    d.contact.geom, d.contact.dist, d.contact.worldid,
                    d.nacon, self.geom_bodyid_wp, self.body_is_allowed_wp,
                    self.undesired_contact_dist_threshold,
                    self.undesired_floor_only, undesired_buf_wp,
                ],
            )
        if self.w_foot_slip > 0.0:
            wp.launch(
                _foot_slip_kernel, dim=naconmax,
                inputs=[
                    d.contact.geom, d.contact.dist, d.contact.worldid,
                    d.nacon, self.geom_bodyid_wp, self.body_is_foot_wp,
                    d.cvel, self.foot_slip_dist_threshold, foot_slip_buf_wp,
                ],
            )
        if self.w_foot_impact > 0.0:
            wp.launch(
                _foot_impact_kernel, dim=naconmax,
                inputs=[
                    d.contact.geom, d.contact.dist, d.contact.worldid,
                    d.nacon, self.geom_bodyid_wp, self.body_is_foot_wp,
                    d.cvel, self.foot_slip_dist_threshold, foot_impact_buf_wp,
                ],
            )
        if self.w_foot_yaw_rate > 0.0:
            wp.launch(
                _foot_yaw_rate_kernel, dim=naconmax,
                inputs=[
                    d.contact.geom, d.contact.dist, d.contact.worldid,
                    d.nacon, self.geom_bodyid_wp, self.body_is_foot_wp,
                    d.cvel, self.foot_slip_dist_threshold, foot_yaw_rate_buf_wp,
                ],
            )

        wp.launch(
            _cost_tracking_kernel, dim=d.nworld,
            inputs=[
                d.xpos, d.xquat, d.ctrl, d.qvel,
                self.body_indices_wp, self.link_scale_wp,
                self.link_pos_offset_wp, dphi_wp,
                self.ref_pos_wp, self.ref_quat_wp, self.ref_jvel_wp,
                frame_idx_wp, frame_alpha_wp, self.T - 1,
                self.orient_offsets_wp, self_coll_buf_wp, undesired_buf_wp,
                foot_slip_buf_wp, foot_impact_buf_wp, foot_yaw_rate_buf_wp,
                step_rewards_wp,
                self.w_pos_relcom, self.w_com, self.w_com_z_scale, self.w_pelvis,
                self.w_pos_relpelvis,
                self.w_orient_pure, self.w_orient_offset,
                self.w_energy, self.w_joint_vel, self.w_joint_vel_ref,
                self.w_self_collision,
                self.w_undesired_contacts, self.w_foot_slip, self.w_foot_impact,
                self.w_foot_yaw_rate,
                self.w_dphi,
                self.knee_ground_bodies_wp,
                self.knee_ground_ref_z_wp,
                self.w_knee_ground,
                self.knee_ground_clearance,
                self.knee_ground_margin,
                self.knee_ground_ref_relative,
                self.smooth_abs2_p_link_wp, self.smooth_abs2_q_link_wp,
                self.smooth_abs_p, self.cosh_p,
                self.torque_range_span_wp,
                self.n_links, self.n_links_real, self.n_offsets, self.nu, self.nv,
                dphi_star,
                self.early_horizon_alpha, self.early_horizon_tau,
                step_t,
            ],
        )

        if actions_buf_wp is not None and prev_action_wp is not None:
            joint_lo = (
                self.physical_joint_range_lo_wp
                if self.joint_limit_use_physical
                else self.joint_range_lo_wp
            )
            joint_hi = (
                self.physical_joint_range_hi_wp
                if self.joint_limit_use_physical
                else self.joint_range_hi_wp
            )
            if prev_dphi_wp is None:
                prev_dphi_wp = wp.zeros(d.nworld, dtype=float)
            if prev_qvel_wp is None:
                prev_qvel_wp = wp.zeros((d.nworld, self.nv), dtype=float)
            if prev_qpos_wp is None:
                prev_qpos_wp = wp.zeros((d.nworld, self.nu), dtype=float)
            wp.launch(
                _cost_extras_kernel, dim=d.nworld,
                inputs=[
                    d.qpos,
                    actions_buf_wp, prev_action_wp,
                    dphi_wp, prev_dphi_wp,
                    d.qvel, prev_qvel_wp,
                    prev_qpos_wp,
                    self.act_qpos_adr_wp,
                    joint_lo, joint_hi,
                    step_rewards_wp,
                    self.w_action_rate, self.w_joint_limit,
                    self.w_dphi_accel, self.w_joint_jerk,
                    self.w_lipschitz, self.lipschitz_eps,
                    self.joint_limit_margin,
                    self.nu, self.nv, step_t,
                ],
            )

        if self.w_flight_dphi > 0.0 and self._flight_mask_active:
            wp.launch(
                _flight_dphi_cost_kernel, dim=d.nworld,
                inputs=[
                    self.ref_airborne_wp, frame_idx_wp, dphi_wp, step_rewards_wp,
                    self.w_flight_dphi, dphi_star, step_t,
                ],
            )

    def step_physics(
        self,
        d: mjw.Data,
        actions_buf_wp: Optional[wp.array] = None,
        step_t: int = 0,
    ) -> None:
        """Advance physics for one control tick (n_frames sub-steps).

        If `self.inner_rate_pd` is True and `actions_buf_wp` is provided, the
        PD law is recomputed against the current qpos/qvel at every substep
        (inner-rate PD at 1/timestep Hz). Otherwise, the torque written by the
        last `launch_ctrl` call is held constant for all substeps (zero-order hold).
        """
        if self.inner_rate_pd and actions_buf_wp is not None:
            for _ in range(self.n_frames):
                self.launch_ctrl(d, actions_buf_wp, step_t)
                mjw.step(self.mjw_model, d)
        else:
            for _ in range(self.n_frames):
                mjw.step(self.mjw_model, d)
