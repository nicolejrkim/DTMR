"""Tracker: DIAL-MPC + phase-as-action TMR controller on the
MuJoCo-Warp batched rollout. The heart of the MPC loop (reverse-diffusion,
graph/eager rollout, MPPI update, run()). The per-step search schedule lives in
schedule.py and the run summaries / diagnostic logs in run_log.py, both mixed in."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import warp as wp
from tqdm import tqdm

from dtmr.env import TrackingEnvWarp
from dtmr.env.kernels import _broadcast_state_kernel, _mean_reward_kernel

from .config import TrackerConfig
from .run_log import RunLogMixin
from .schedule import ScheduleMixin
from .splines import _build_interp_matrix_np
from .mppi_kernels import (
    _perturb_kernel,
    _node2u_batch_kernel,
    _copy_actions_kernel,
    _extract_rdphi_kernel,
    _broadcast_phi_kernel,
    _softmax_weights_kernel,
    _weighted_sum_Y_kernel,
)


# Contact and constraint capacity per rollout world of the planning buffers.
PLAN_NCONMAX = 120
PLAN_NJMAX = 480


class Tracker(ScheduleMixin, RunLogMixin):
    """DIAL-MPC + TMR controller using mujoco-warp batched rollouts.

    Action layout: Ũ = [U; φ] where U ∈ R^{nu} are normalized joint actions in
    [-1, 1] and the trailing dim is rdphi (phase command), also in [-1, 1].
    `env.launch_ctrl` consumes only the first `nu` columns; the last column is
    routed into `_phase_update_kernel` each tick.
    """

    def __init__(
        self,
        env: TrackingEnvWarp,
        args: TrackerConfig,
        nconmax: int = PLAN_NCONMAX,
        njmax: int = PLAN_NJMAX,
        use_graph: bool = True,
    ) -> None:
        self.env = env
        self.args = args
        self.ctrl_dt = float(env.dt)

        self.Hnode = args.Hnode
        self.Hsample = args.Hsample

        self.n_action_dim = env.nu + 1  # nu joints + rdphi

        # Spline matrices (node ↔ u). node_dt = Hsample*dt / Hnode.
        end = self.ctrl_dt * self.Hsample
        self.step_us = np.linspace(0.0, end, self.Hsample + 1, dtype=np.float32)
        self.step_nodes = np.linspace(0.0, end, self.Hnode + 1, dtype=np.float32)
        self.node2u_mat = _build_interp_matrix_np(self.step_nodes, self.step_us)
        self.u2node_mat = _build_interp_matrix_np(self.step_us, self.step_nodes)
        self.node2u_mat_wp = wp.array(self.node2u_mat, dtype=float)

        # Sigma schedule: shape (Hnode+1, C).
        # Each column can have an independent per-dim scale; the trailing dphi
        # column uses sigma_scale_phi when set (>= 0), otherwise inherits
        # sigma_scale so existing configs are unaffected.
        idx = np.arange(self.Hnode + 1, dtype=np.float32)
        horizon_envelope = (args.horizon_diffuse_factor ** idx[::-1]).astype(np.float32)
        phi_scale = args.sigma_scale_phi if args.sigma_scale_phi >= 0.0 else args.sigma_scale
        per_dim = np.full(self.n_action_dim, args.sigma_scale, dtype=np.float32)
        per_dim[-1] = phi_scale  # trailing dphi dimension
        _am = float(args.sigma_arm_mult)
        if _am != 1.0:
            arm_idx = env.joint_indices("arms")
            per_dim[arm_idx] *= _am
            print(f"[sigma] arm-channel noise x{_am:g} ({len(arm_idx)} arm joints)")
        self.sigma_control = np.outer(horizon_envelope, per_dim)  # (Hnode+1, C)
        # Decoupled dphi horizon envelope: when set (>= 0) the trailing column
        # uses its own per-node factor instead of horizon_diffuse_factor.
        if args.horizon_diffuse_factor_phi >= 0.0:
            envelope_phi = (args.horizon_diffuse_factor_phi
                            ** idx[::-1]).astype(np.float32)
            self.sigma_control[:, -1] = envelope_phi * phi_scale
        # Flat-envelope variant (all nodes = 1.0): used for the first
        # `sigma_flat_steps` MPC steps so node 0 (the executed action) is
        # annealed with full sigma like every other node — lets MPPI discover
        # start corrections (gravity pre-loads) it otherwise can't reach.
        self.sigma_control_flat = np.outer(
            np.ones_like(horizon_envelope), per_dim)
        self._sigma_base = self.sigma_control
        self.traj_diffuse_factor = args.traj_diffuse_factor
        # Decoupled dphi anneal: the trailing column may carry its own decay
        # factor and may opt out of the per-step start_exp band shift; with the
        # inherit defaults `_sigma_iter` stays on the single-expression path.
        self.traj_diffuse_factor_phi = (
            args.traj_diffuse_factor_phi
            if args.traj_diffuse_factor_phi >= 0.0
            else args.traj_diffuse_factor)
        self.phi_share_band = bool(args.phi_share_band)
        self._phi_sched_decoupled = (
            self.traj_diffuse_factor_phi != self.traj_diffuse_factor
            or not self.phi_share_band)
        # Warm-start tail fill for the phase column (see config): True carries the
        # last phase node forward, False resets it to the reference rate.
        self.phi_tail_carry = bool(args.phi_tail_carry)

        # Batched mjw.Data with nworld = Nsample + 1 (the +1 holds Ybar itself
        # so `rew_ybar` is available to the softmax kernel). Reference pattern:
        # no reset_data / no forward on plan_data — the graph's broadcast kernel
        # seeds qpos/qvel at the top of every launch, and mjw.step handles
        # the forward pass internally at each tick.
        self.nworld_plan = args.Nsample + 1
        self.plan_data = env.make_data(nworld=self.nworld_plan,
                                       nconmax=nconmax, njmax=njmax)

        # Elite truncation: keep only the top `n_elite` highest-reward rows in the
        # softmax (truncated MPPI). elite_frac >= 1.0 → n_elite = nworld, which the
        # kernel/eager guards treat as a pure no-op (current behavior, bit-for-bit).
        # Otherwise rank over the Nsample real samples; the +1 Ybar row stays eligible.
        if args.elite_frac >= 1.0:
            self.n_elite = self.nworld_plan
        else:
            self.n_elite = max(1, int(round(args.elite_frac * args.Nsample)))

        NW = self.nworld_plan
        H = self.Hsample
        Hn = self.Hnode
        C = self.n_action_dim

        # Persistent MPPI buffers (stable addresses for graph capture).
        self.Ybar_wp = wp.zeros((Hn + 1, C), dtype=float)
        self.Y0s_wp = wp.zeros((NW, Hn + 1, C), dtype=float)
        self.uss_wp = wp.zeros((NW, H + 1, C), dtype=float)
        self.actions_buf_wp = wp.zeros((NW, H, C), dtype=float)
        self.step_rewards_wp = wp.zeros((NW, H), dtype=float)
        self.mean_rewards_wp = wp.zeros(NW, dtype=float)
        self.weights_wp = wp.zeros(NW, dtype=float)
        self.current_sigma_wp = wp.zeros((Hn + 1, C), dtype=float)
        self.seed_buf_wp = wp.zeros(1, dtype=wp.int32)

        # Per-world phase scratch. Link pose is read directly from d.xpos /
        # d.xquat inside the cost kernel — no separate gather buffer (mirror
        # reference pattern).
        self.phi_wp = wp.zeros(NW, dtype=float)
        self.dphi_wp = wp.zeros(NW, dtype=float)
        self.rdphi_wp = wp.zeros(NW, dtype=float)
        self.frame_idx_wp = wp.zeros(NW, dtype=wp.int32)
        self.frame_alpha_wp = wp.zeros(NW, dtype=float)
        # Self-collision scratch (zeroed per-step inside launch_cost).
        self.self_coll_buf_wp = wp.zeros(NW, dtype=float)
        # Undesired-contacts scratch (per-step count, populated by the
        # _undesired_contacts_kernel when w_undesired_contacts > 0).
        self.undesired_buf_wp = wp.zeros(NW, dtype=float)
        # Foot-slip scratch (per-step squared horizontal foot velocity,
        # populated by the _foot_slip_kernel when w_foot_slip > 0).
        self.foot_slip_buf_wp = wp.zeros(NW, dtype=float)
        # Foot-impact scratch (per-step squared downward foot velocity at
        # touchdown, populated by the _foot_impact_kernel when w_foot_impact > 0).
        self.foot_impact_buf_wp = wp.zeros(NW, dtype=float)
        # Foot yaw-rate scratch (per-step squared foot yaw angular velocity in
        # contact, populated by the _foot_yaw_rate_kernel when w_foot_yaw_rate > 0).
        self.foot_yaw_rate_buf_wp = wp.zeros(NW, dtype=float)
        # Previous action buffer for the action-rate regularizer (extras kernel).
        # Zeroed at the top of every rollout so the first-step diff is 0.
        self.prev_action_wp = wp.zeros((NW, env.nu), dtype=float)
        # Previous-step dphi buffer for the phase-acceleration regularizer.
        # Same lifecycle as prev_action_wp (zero at rollout start).
        self.prev_dphi_wp = wp.zeros(NW, dtype=float)
        # Previous-step qvel buffer for the joint-jerk regularizer.
        # Same lifecycle as prev_action_wp (zero at rollout start).
        self.prev_qvel_wp = wp.zeros((NW, env.nv), dtype=float)
        # Previous-step actuated-qpos buffer for the state-Lipschitz regularizer.
        # Same lifecycle as prev_action_wp (zero at rollout start).
        self.prev_qpos_wp = wp.zeros((NW, env.nu), dtype=float)

        # Staging buffers for per-step inputs that must be fed into the graph.
        self.staging_qpos_wp = wp.zeros(env.nq, dtype=float)
        self.staging_qvel_wp = wp.zeros(env.nv, dtype=float)
        self.staging_phi_wp = wp.zeros(1, dtype=float)

        # Attempt graph capture; fall back to the eager python loop on failure.
        self.use_graph = bool(use_graph)
        self._graph = None
        if self.use_graph:
            self._graph = self._capture_graph()
            if self._graph is None:
                print("Graph capture failed — falling back to eager mode.")
                self.use_graph = False
            else:
                self._prewarm_graph()

    # ----- Host-side helpers -----

    def _node2u_np(self, Y: np.ndarray) -> np.ndarray:
        return self.node2u_mat @ Y

    def _u2node_np(self, us: np.ndarray) -> np.ndarray:
        return self.u2node_mat @ us

    def _node2u_batch_np(self, Ys: np.ndarray) -> np.ndarray:
        return np.einsum("ij,bjk->bik", self.node2u_mat, Ys)


    def _build_ref_actions(self) -> Optional[np.ndarray]:
        """Convert env._state_ref["qpos"] joints into normalized actions.

        Per-joint normalization from
        `[joint_range_lo, joint_range_hi]` into `[-1, 1]`, divided by
        `action_scale`, clipped, with a trailing dphi=0 column appended.
        Returns None if the env has no reference qpos.
        """
        env = self.env
        state_ref = getattr(env, "_state_ref", None)
        if state_ref is None or "qpos" not in state_ref:
            return None
        ref_qpos = np.asarray(state_ref["qpos"], dtype=np.float32)
        if ref_qpos.shape[1] < 7 + env.nu:
            return None
        ref_joint_q = ref_qpos[:, 7:7 + env.nu]
        jr = np.asarray(env.joint_range, dtype=np.float32)  # (nu, 2)
        span = np.maximum(jr[:, 1] - jr[:, 0], 1e-8)
        ref_act_normalized = (ref_joint_q - jr[:, 0]) / span
        ref_joints = (ref_act_normalized * 2.0 - 1.0) / float(env.action_scale)
        ref_joints = np.clip(ref_joints, -1.0, 1.0).astype(np.float32)
        dphi_col = np.zeros((ref_joints.shape[0], 1), dtype=np.float32)
        return np.concatenate([ref_joints, dphi_col], axis=1)

    def _init_Y_from_ref(
        self, ref_actions: Optional[np.ndarray]
    ) -> np.ndarray:
        """Build initial Y (nodes) by sampling the first `Hsample+1` reference
        actions into spline u-space, then projecting back to node-space via
        `u2node`. Falls back to zero init if no reference is available.
        """
        H = self.Hsample
        Hn = self.Hnode
        C = self.n_action_dim
        n_horizon = H + 1

        if ref_actions is None:
            return np.zeros((Hn + 1, C), dtype=np.float32)

        n_avail = min(n_horizon, ref_actions.shape[0])
        if n_avail < n_horizon:
            pad = np.tile(ref_actions[-1:], (n_horizon - n_avail, 1))
            us_init = np.concatenate([ref_actions[:n_avail], pad], axis=0)
        else:
            us_init = ref_actions[:n_horizon]
        us_init = us_init.astype(np.float32)
        return self._u2node_np(us_init)

    def _shift_Y_with_ref_np(
        self,
        Y: np.ndarray,
        t: int,
        ref_actions: Optional[np.ndarray],
    ) -> np.ndarray:
        """Reference-aware shift: roll Y by one control step and fill the tail
        with `ref_actions[t + Hsample + 1]` (or the last reference action if
        past the end; zeros if no reference).
        """
        u = self._node2u_np(Y)
        prev_last_phi = u[-1, -1]  # previous plan's last phase node (pre-roll)
        u = np.roll(u, -1, axis=0)
        n_horizon = self.Hsample + 1
        if ref_actions is not None:
            next_idx = t + n_horizon
            if next_idx < ref_actions.shape[0]:
                u[-1] = ref_actions[next_idx]
            else:
                u[-1] = ref_actions[-1]
        else:
            u[-1] = np.zeros(self.n_action_dim, dtype=np.float32)
        # Opt-in: carry the previous last phase forward instead of snapping the
        # tail phase to the reference (rdphi=0). Joints keep the reference tail.
        if self.phi_tail_carry:
            u[-1, -1] = prev_last_phi
        return self._u2node_np(u)

    # ----- CUDA graph capture -----

    def _launch_plan_cost(self, t: int) -> None:
        """Launch the per-world cost for plan-rollout step `t`.

        Shared verbatim by the graph and eager rollout bodies (both use the
        controller-owned plan buffers). Issues only kernel launches — no host
        syncs — so it is safe inside `wp.ScopedCapture`. The single-world
        execution path in `run()` uses its own `single_*` buffers and does not
        call this.
        """
        self.env.launch_cost(
            self.plan_data, self.dphi_wp, self.frame_idx_wp, self.frame_alpha_wp,
            self.step_rewards_wp, step_t=t,
            self_coll_buf_wp=self.self_coll_buf_wp,
            undesired_buf_wp=self.undesired_buf_wp,
            foot_slip_buf_wp=self.foot_slip_buf_wp,
            foot_impact_buf_wp=self.foot_impact_buf_wp,
            foot_yaw_rate_buf_wp=self.foot_yaw_rate_buf_wp,
            actions_buf_wp=self.actions_buf_wp,
            prev_action_wp=self.prev_action_wp,
            prev_dphi_wp=self.prev_dphi_wp,
            prev_qvel_wp=self.prev_qvel_wp,
            prev_qpos_wp=self.prev_qpos_wp,
        )

    def _rollout_graph_body(self) -> None:
        """Record the kernel sequence for one reverse-diffusion iteration.

        Called twice: once inside `wp.ScopedCapture()` to build the graph, and
        not elsewhere (the eager path runs a python-loop variant so numpy
        host-side operations can stay intact).
        """
        d = self.plan_data
        env = self.env
        NW = self.nworld_plan
        H = self.Hsample
        C = self.n_action_dim

        # Perturb Ybar → Y0s (last world = Ybar itself).
        wp.launch(
            _perturb_kernel, dim=self.args.Nsample,
            inputs=[self.Ybar_wp, self.current_sigma_wp, self.Y0s_wp,
                    self.seed_buf_wp, self.Hnode + 1, C, self.args.Nsample,
                    float(self.args.noise_time_corr)],
        )

        # Node → u spline, per trajectory.
        wp.launch(
            _node2u_batch_kernel, dim=NW,
            inputs=[self.node2u_mat_wp, self.Y0s_wp, self.uss_wp,
                    H + 1, self.Hnode + 1, C],
        )

        # uss[:, :H, :] → actions_buf
        wp.launch(
            _copy_actions_kernel, dim=NW,
            inputs=[self.uss_wp, self.actions_buf_wp, H, C],
        )

        # Broadcast initial state (qpos, qvel, phi) to all worlds. Reference
        # does not explicitly call mjw.forward: the first mjw.step each tick
        # runs forward internally, so the kinematics fields d.xpos / d.xquat
        # are refreshed before the cost kernel reads them.
        wp.launch(
            _broadcast_state_kernel, dim=NW,
            inputs=[self.staging_qpos_wp, self.staging_qvel_wp,
                    d.qpos, d.qvel, env.nq, env.nv],
        )
        wp.launch(
            _broadcast_phi_kernel, dim=NW,
            inputs=[self.staging_phi_wp, self.phi_wp],
        )
        self.step_rewards_wp.zero_()
        self.prev_action_wp.zero_()
        self.prev_dphi_wp.zero_()
        self.prev_qvel_wp.zero_()
        self.prev_qpos_wp.zero_()

        # Horizon rollout.
        for t in range(H):
            wp.launch(
                _extract_rdphi_kernel, dim=NW,
                inputs=[self.actions_buf_wp, self.rdphi_wp, t, env.nu],
            )
            env.launch_phase_update(
                self.rdphi_wp, self.phi_wp, self.dphi_wp,
                self.frame_idx_wp, self.frame_alpha_wp,
            )
            env.launch_ctrl(d, self.actions_buf_wp, t)
            env.step_physics(d, self.actions_buf_wp, t)
            self._launch_plan_cost(t)

        # Aggregate + softmax + weighted Ybar update.
        wp.launch(
            _mean_reward_kernel, dim=NW,
            inputs=[self.step_rewards_wp, self.mean_rewards_wp, H],
        )
        wp.launch(
            _softmax_weights_kernel, dim=1,
            inputs=[self.mean_rewards_wp, self.weights_wp,
                    float(self.args.temp_sample), NW, int(self.n_elite)],
        )
        wp.launch(
            _weighted_sum_Y_kernel, dim=self.Hnode + 1,
            inputs=[self.weights_wp, self.Y0s_wp, self.Ybar_wp,
                    NW, C],
        )

    def _capture_graph(self):
        try:
            with wp.ScopedCapture() as capture:
                self._rollout_graph_body()
            print("CUDA graph captured successfully.")
            return capture.graph
        except Exception as e:
            print(f"CUDA graph capture failed: {e}")
            return None

    def _prewarm_graph(self) -> None:
        """Force eager `cudaGraphInstantiate` once to avoid the race that
        surfaces as CUDA error 801 when two threads hit `capture_launch`
        simultaneously on the first call.
        """
        try:
            wp.capture_launch(self._graph)
            wp.synchronize()
        except Exception:
            pass

    # ----- Eager rollout (--no_graph, or graph capture failed) -----

    def _rollout_batch_eager(
        self,
        qpos0: np.ndarray,
        qvel0: np.ndarray,
        phi0: float,
        actions_all: np.ndarray,
    ) -> np.ndarray:
        """Python-loop rollout. Returns the horizon-mean rewards, shape (NW,)."""
        d = self.plan_data
        env = self.env
        NW = self.nworld_plan
        H = self.Hsample

        env.broadcast_state(d, qpos0, qvel0)

        phi_host = np.full(NW, float(phi0), dtype=np.float32)
        wp.copy(self.phi_wp, wp.array(phi_host, dtype=float))

        wp.copy(self.actions_buf_wp,
                wp.array(actions_all.astype(np.float32), dtype=float))
        self.step_rewards_wp.zero_()
        self.prev_action_wp.zero_()
        self.prev_dphi_wp.zero_()
        self.prev_qvel_wp.zero_()
        self.prev_qpos_wp.zero_()

        for t in range(H):
            rdphi_host = actions_all[:, t, -1].astype(np.float32)
            wp.copy(self.rdphi_wp, wp.array(rdphi_host, dtype=float))
            env.launch_phase_update(
                self.rdphi_wp, self.phi_wp, self.dphi_wp,
                self.frame_idx_wp, self.frame_alpha_wp,
            )
            env.launch_ctrl(d, self.actions_buf_wp, t)
            env.step_physics(d, self.actions_buf_wp, t)
            self._launch_plan_cost(t)

        wp.launch(_mean_reward_kernel, dim=NW,
                  inputs=[self.step_rewards_wp, self.mean_rewards_wp, H])
        wp.synchronize()
        return self.mean_rewards_wp.numpy().copy()

    def _reverse_once_eager(
        self,
        qpos0: np.ndarray,
        qvel0: np.ndarray,
        phi0: float,
        Ybar: np.ndarray,
        noise_scale: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        N = self.args.Nsample
        H = self.Hsample
        C = self.n_action_dim

        eps = rng.standard_normal((N, self.Hnode + 1, C)).astype(np.float32)
        rho = float(self.args.noise_time_corr)
        if rho > 0.0:
            # AR(1) across nodes, unit marginal variance (numpy mirror of the
            # rho branch in _perturb_kernel).
            c = np.float32(np.sqrt(1.0 - rho * rho))
            for j in range(1, self.Hnode + 1):
                eps[:, j, :] = rho * eps[:, j - 1, :] + c * eps[:, j, :]
        Y0s = Ybar[None, :, :] + eps * noise_scale[None, :, :]
        Y0s[:, 0, :] = Ybar[0]
        Y0s = np.clip(Y0s, -1.0, 1.0)
        Y0s = np.concatenate([Y0s, Ybar[None, :, :]], axis=0)

        uss = self._node2u_batch_np(Y0s)
        uss = np.clip(uss, -1.0, 1.0)
        actions = uss[:, :H, :]

        rews = self._rollout_batch_eager(qpos0, qvel0, phi0, actions)
        finite = np.isfinite(rews)
        if not finite.all():
            if finite.any():
                rews = np.where(finite, rews, rews[finite].min() - 1.0)
            else:
                self._last_improvement = np.inf  # unknown — never early-terminate on it
                return Ybar
        rew_ybar = rews[-1]
        self._last_improvement = float(rews.max() - rew_ybar)
        std = np.sqrt(np.mean((rews - rews.mean()) ** 2) + 1e-6)
        logp = (rews - rew_ybar) / std / self.args.temp_sample
        # Elite truncation (truncated MPPI): zero-weight all but the top n_elite
        # rewards. Exact selection via np.partition; no-op when n_elite >= nworld.
        if self.n_elite < rews.shape[0]:
            thresh = np.partition(rews, -self.n_elite)[-self.n_elite]
            logp = np.where(rews >= thresh, logp, -np.inf)
        logp -= logp.max()
        weights = np.exp(logp)
        weights /= weights.sum()
        Ybar_new = np.einsum("n,nij->ij", weights, Y0s)
        return Ybar_new

    # ----- Graph rollout (CUDA-graph fast path) -----

    def _reverse_graph(
        self,
        qpos0: np.ndarray,
        qvel0: np.ndarray,
        phi0: float,
        Ybar: np.ndarray,
        rng: np.random.Generator,
        n_diffuse: int,
        start_exp: int = 0,
    ) -> np.ndarray:
        # Seed Ybar / qpos / qvel / phi0 into persistent buffers.
        wp.copy(self.Ybar_wp, wp.array(Ybar.astype(np.float32), dtype=float))
        wp.copy(self.staging_qpos_wp, wp.array(qpos0.astype(np.float32), dtype=float))
        wp.copy(self.staging_qvel_wp, wp.array(qvel0.astype(np.float32), dtype=float))
        wp.copy(self.staging_phi_wp,
                wp.array(np.array([float(phi0)], dtype=np.float32), dtype=float))

        # SPIDER-style early termination: reading rewards costs one host sync
        # per iteration, so the check only runs when the threshold is set.
        imp_thresh = float(self.args.improvement_threshold)
        imp_check = max(1, int(self.args.improvement_check_steps))
        _adapt = int(self.args.adapt_ladder)
        imp_hist = []
        self._cert_call = getattr(self, "_cert_call", -1) + 1
        # Closed-loop certainty anneal (config.sigma_cert): the per-iteration
        # ESS of the sample-reward distribution decides the next rung. Probe
        # mode logs the ESS along the normal schedule without acting on it.
        _cert = int(self.args.sigma_cert)
        _probe = int(self.args.sigma_cert_probe)
        _ctau = float(self.args.sigma_cert_temp)
        if _ctau <= 0.0:
            _ctau = float(self.args.temp_sample)
        _flat = float(self.args.sigma_cert_flat_ess)
        _skip = float(self.args.sigma_cert_skip_imp)
        if _cert:
            _e0 = int(self.args.sigma_cert_start_exp)
            if int(self.args.sigma_cert_entry_mode):
                # entry thermostat: use the reward-EMA-adapted entry rung
                # (maintained in main_loop); the cold-start/warmup window
                # (negative start_exp) still overrides toward hotter.
                _e0 = int(getattr(self, "_cert_entry", _e0))
            # cold-start / warmup windows may enter even hotter than the
            # certainty entry rung; keep the wider of the two.
            _e = min(start_exp, _e0) if start_exp < 0 else _e0
            _e_end = int(self.args.sigma_cert_end_exp)
            _cap = int(self.args.sigma_cert_max_iters)
        i = 0
        while True:
            if _cert:
                if i >= _cap or _e > _e_end:
                    break
                sigma_i = self._sigma_iter(0, _e).astype(np.float32)
            else:
                if i >= n_diffuse:
                    break
                sigma_i = self._sigma_iter(
                    i * float(getattr(self, "_adapt_stride", 1.0)),
                    start_exp).astype(np.float32)
            wp.copy(self.current_sigma_wp, wp.array(sigma_i, dtype=float))

            seed_i = int(rng.integers(0, 2 ** 31))
            wp.copy(self.seed_buf_wp,
                    wp.array(np.array([seed_i], dtype=np.int32), dtype=wp.int32))

            wp.capture_launch(self._graph)
            rews = (self.mean_rewards_wp.numpy()
                    if (_cert or _probe or imp_thresh > 0.0 or _adapt) else None)
            _ess = _imp_h = None
            if _cert or _probe:
                # world N (last) is the un-noised Ybar rollout = the incumbent
                _r = rews[:-1].astype(np.float64)
                _imp_h = float(np.nanmax(_r) - float(rews[-1]))
                _sd = max(float(np.nanstd(_r)), 1e-9)
                _p = np.exp((_r - np.nanmax(_r)) / (_ctau * _sd))
                _p = np.nan_to_num(_p)
                _p /= max(float(_p.sum()), 1e-12)
                _ess = 1.0 / (len(_p) * float((_p * _p).sum()))
                if getattr(self, "_cert_ess_log", None) is not None:
                    # also record the incumbent reward (its delta across
                    # iterations = realized gain of the anneal), improvement
                    # headroom (best sample vs incumbent) and p_beat, so probe
                    # runs can calibrate which statistic predicts "another
                    # iteration would help"; col 0 = reverse-call index so
                    # offline analysis can split MPC steps.
                    _inc = float(rews[-1])
                    self._cert_ess_log.append(
                        (float(getattr(self, "_cert_call", 0)),
                         float(_e if _cert else start_exp + i), float(_ess),
                         _imp_h, float((_r > _inc).mean()), _inc))
            i += 1
            if _adapt:
                # samples-only headroom (can go negative = no sample beat the
                # plan); feeds the next step's full-vs-tail choice and the
                # improvement_hist telemetry for threshold calibration.
                self._last_improvement = float(
                    np.nanmax(rews[:-1]) - float(rews[-1]))
                imp_hist.append(self._last_improvement)
            if imp_thresh > 0.0:
                imp_hist.append(float(np.nanmax(rews) - rews[-1]))
                if (len(imp_hist) >= imp_check
                        and all(v < imp_thresh for v in imp_hist[-imp_check:])):
                    break
            if _cert:
                if _ess > _flat:
                    break  # flat landscape: no discriminative signal left
                # destructive-hot scale (cloud far below the plan): fall
                # through 2 rungs; productive scale (headroom near 0):
                # descend one rung at a time
                _e += 2 if _imp_h < _skip else 1

        self._last_reverse_iters = i
        self._last_imp_hist = imp_hist
        wp.synchronize()
        return self.Ybar_wp.numpy().copy()

    def _reverse_eager(
        self,
        qpos0: np.ndarray,
        qvel0: np.ndarray,
        phi0: float,
        Ybar: np.ndarray,
        rng: np.random.Generator,
        n_diffuse: int,
        start_exp: int = 0,
    ) -> np.ndarray:
        if (int(self.args.sigma_cert)
                and not getattr(self, "_cert_eager_warned", False)):
            print("[sigma-cert] WARNING: eager path does not support "
                  "sigma_cert; running the open-loop anneal")
            self._cert_eager_warned = True
        imp_thresh = float(self.args.improvement_threshold)
        imp_check = max(1, int(self.args.improvement_check_steps))
        imp_hist = []
        self._last_reverse_iters = n_diffuse
        for i in range(n_diffuse):
            sigma_i = self._sigma_iter(
                i * float(getattr(self, "_adapt_stride", 1.0)), start_exp)
            Ybar = self._reverse_once_eager(qpos0, qvel0, phi0, Ybar, sigma_i, rng)
            if imp_thresh > 0.0:
                imp_hist.append(self._last_improvement)
                if (len(imp_hist) >= imp_check
                        and all(v < imp_thresh for v in imp_hist[-imp_check:])):
                    self._last_reverse_iters = i + 1
                    break
        self._last_imp_hist = imp_hist
        return Ybar

    def reverse(
        self,
        qpos0: np.ndarray,
        qvel0: np.ndarray,
        phi0: float,
        Ybar: np.ndarray,
        rng: np.random.Generator,
        n_diffuse: int,
        start_exp: int = 0,
    ) -> np.ndarray:
        if self.use_graph and self._graph is not None:
            return self._reverse_graph(qpos0, qvel0, phi0, Ybar, rng, n_diffuse, start_exp)
        return self._reverse_eager(qpos0, qvel0, phi0, Ybar, rng, n_diffuse, start_exp)

    # ----- Main loop -----

    def _settle(self, exec_d, Y, rng, ref_actions_all, single_action_buf):
        """Unlogged pre-roll at the first reference frame (settle_steps). Returns the
        plan, which the MPC and preload variants adjust."""
        env = self.env
        # Physics settle pre-roll: hold at the reference frame-0 pose for
        # `settle_steps` unlogged steps so the power-on transient (gravity-sag
        # catch, contact equilibration) completes before tracking/scoring.
        # settle_mpc=False: fixed PD hold on the frame-0 action.
        # settle_mpc=True:  run the optimizer with phi held at 0 — the sampler
        # finds the gravity pre-load offsets and they persist in the plan Y
        # (equilibrium-aware conveyor at t=0). phi/plan never advance: no
        # phase update, no plan shift, no logging.
        settle = int(self.args.settle_steps)
        if settle > 0 and ref_actions_all is not None:
            if bool(self.args.settle_mpc):
                for k in range(settle):
                    s_qpos = exec_d.qpos.numpy()[0].copy()
                    s_qvel = exec_d.qvel.numpy()[0].copy()
                    if k == 0:
                        s_nd, s_exp = self.args.Ndiffuse_init, 0
                    elif self.args.Ndiffuse_steady >= 0:
                        s_nd = self.args.Ndiffuse_steady
                        s_exp = max(0, self.args.Ndiffuse_init - s_nd)
                    else:
                        s_nd, s_exp = self.args.Ndiffuse, 0
                    Y = self.reverse(s_qpos, s_qvel, 0.0, Y, rng, s_nd, s_exp)
                    s_act = self._node2u_np(Y)[0].astype(np.float32)
                    wp.copy(single_action_buf,
                            wp.array(s_act.reshape(1, 1, -1), dtype=float))
                    env.launch_ctrl(exec_d, single_action_buf, 0)
                    env.step_physics(exec_d, single_action_buf, 0)
                wp.synchronize()
                print(f"[settle] {settle} MPC settle steps at phi=0 "
                      f"({settle * self.ctrl_dt:.2f}s); plan Y pre-warmed")
            else:
                a0 = ref_actions_all[0].astype(np.float32).reshape(1, 1, -1)
                pin = bool(self.args.settle_pin_root)
                preload = pin and bool(self.args.settle_preload)
                q_init = exec_d.qpos.numpy()[0].copy()
                root0 = q_init[:7].copy()
                hold = a0.copy()
                phases = 2 if preload else 1
                steps_per = max(1, settle // phases)
                for ph in range(phases):
                    wp.copy(single_action_buf, wp.array(hold, dtype=float))
                    for _ in range(steps_per):
                        env.launch_ctrl(exec_d, single_action_buf, 0)
                        env.step_physics(exec_d, single_action_buf, 0)
                        if pin:
                            q = exec_d.qpos.numpy()[0].copy()
                            v = exec_d.qvel.numpy()[0].copy()
                            q[:7] = root0
                            v[:6] = 0.0
                            wp.copy(exec_d.qpos, wp.array(q.reshape(1, -1), dtype=float))
                            wp.copy(exec_d.qvel, wp.array(v.reshape(1, -1), dtype=float))
                    wp.synchronize()
                    if preload and ph == 0:
                        jr = np.asarray(env.joint_range, dtype=np.float32)
                        nj = jr.shape[0]
                        half = np.maximum((jr[:, 1] - jr[:, 0]) / 2.0, 1e-6)
                        q_settled = exec_d.qpos.numpy()[0].copy()
                        sag = q_init[7:7 + nj] - q_settled[7:7 + nj]
                        delta_a = (sag / (half * float(env.action_scale))).astype(np.float32)
                        hold[0, 0, :nj] += delta_a
                        Y[:, :nj] += delta_a[None, :]
                        print(f"[settle-preload] gravity sag max {np.degrees(np.abs(sag).max()):.2f} deg, "
                              f"mean {np.degrees(np.abs(sag).mean()):.2f} deg -> plan pre-loaded")
                wp.synchronize()
                print(f"[settle] {phases * steps_per} PD-hold pre-roll steps "
                      f"(pin_root={pin}, preload={preload}) before tracking")

        return Y

    def run(self, print_every: int = 10) -> dict:
        env = self.env
        rng = np.random.default_rng(self.args.seed)

        # Single-world execution env (applies the chosen action per tick).
        exec_d = env.make_data(nworld=1)
        env.reset_data(exec_d)
        env.broadcast_reset_state(exec_d)

        ref_actions_all = self._build_ref_actions()
        Y = self._init_Y_from_ref(ref_actions_all)
        if ref_actions_all is not None:
            print(
                f"Warm-started Y from {min(self.Hsample + 1, ref_actions_all.shape[0])} "
                f"reference frames (total ref frames: {ref_actions_all.shape[0]})"
            )
        else:
            print("No reference qpos found — Y initialized to zeros.")
        phi = 0.0

        log = {"rewards": [], "phases": [], "qpos": [], "qvel": [],
               "diffuse_iters": [], "improvement_hist": [],
               "joint_targets": [], "steps": 0}
        # Minimum meaningful phase step (dphi_star). Used to early-out
        # when the next tick would be implicitly clipped by the phi <= 1 rule
        # in _phase_update_kernel, so the run ends once the motion is done.
        dphi_star = float(env.dt) / float(env.motion_dur)
        phi_wp_single = wp.zeros(1, dtype=float)
        dphi_wp_single = wp.zeros(1, dtype=float)
        rdphi_wp_single = wp.zeros(1, dtype=float)
        frame_idx_wp_single = wp.zeros(1, dtype=wp.int32)
        frame_alpha_wp_single = wp.zeros(1, dtype=float)
        single_action_buf = wp.zeros((1, 1, self.n_action_dim), dtype=float)
        single_rew = wp.zeros((1, 1), dtype=float)
        single_self_coll = wp.zeros(1, dtype=float)
        single_undesired = wp.zeros(1, dtype=float)
        single_foot_slip = wp.zeros(1, dtype=float)
        single_foot_impact = wp.zeros(1, dtype=float)
        single_foot_yaw_rate = wp.zeros(1, dtype=float)
        single_prev_action = wp.zeros((1, env.nu), dtype=float)
        single_prev_dphi = wp.zeros(1, dtype=float)
        single_prev_qvel = wp.zeros((1, env.nv), dtype=float)
        single_prev_qpos = wp.zeros((1, env.nu), dtype=float)

        self._build_reference_dynamism()
        self._build_sigma_frame_gain()
        self._init_certainty_anneal()
        self._announce_search_schedule()
        Y = self._settle(exec_d, Y, rng, ref_actions_all, single_action_buf)

        with tqdm(range(self.args.n_steps), desc="Rollout") as pbar:
            for t in pbar:
                qpos0 = exec_d.qpos.numpy()[0].copy()
                qvel0 = exec_d.qvel.numpy()[0].copy()

                n_diffuse, start_exp = self._diffusion_budget(t, phi)
                self._select_sigma_base(t, phi)
                t0 = time.time()
                _dec = max(1, int(self.args.mpc_decimation))
                if t % _dec == 0:
                    self._last_reverse_iters = n_diffuse
                    self._last_imp_hist = []
                    Y = self.reverse(qpos0, qvel0, phi, Y, rng, n_diffuse, start_exp)
                    log["diffuse_iters"].append(int(self._last_reverse_iters))
                    log["improvement_hist"].append(
                        np.asarray(self._last_imp_hist, dtype=np.float32))
                else:
                    # decimated step: execute the shifted plan open-loop
                    log["diffuse_iters"].append(0)
                    log["improvement_hist"].append(np.empty(0, dtype=np.float32))

                us = self._node2u_np(Y)
                action = us[0]
                # Log the commanded PD joint target (radians) every step, decoded from the
                # applied action exactly as _compute_ctrl_kernel does:
                # target = joint_lo + ((a*action_scale+1)/2)*(joint_hi-joint_lo), clamped.
                _jr = np.asarray(self.env.joint_range, dtype=np.float32)        # (nu, 2)
                _nu = _jr.shape[0]
                _anorm = (action[:_nu] * float(self.env.action_scale) + 1.0) * 0.5
                _qt = _jr[:, 0] + _anorm * (_jr[:, 1] - _jr[:, 0])
                log["joint_targets"].append(np.clip(_qt, _jr[:, 0], _jr[:, 1]).astype(np.float32))

                wp.copy(rdphi_wp_single,
                        wp.array(np.array([action[-1]], dtype=np.float32), dtype=float))
                wp.copy(phi_wp_single,
                        wp.array(np.array([phi], dtype=np.float32), dtype=float))
                env.launch_phase_update(rdphi_wp_single, phi_wp_single,
                                        dphi_wp_single, frame_idx_wp_single,
                                        frame_alpha_wp_single)

                wp.copy(single_action_buf,
                        wp.array(action.reshape(1, 1, -1).astype(np.float32), dtype=float))
                env.launch_ctrl(exec_d, single_action_buf, 0)
                env.step_physics(exec_d, single_action_buf, 0)
                env.launch_cost(exec_d, dphi_wp_single, frame_idx_wp_single,
                                frame_alpha_wp_single,
                                single_rew, step_t=0,
                                self_coll_buf_wp=single_self_coll,
                                undesired_buf_wp=single_undesired,
                                foot_slip_buf_wp=single_foot_slip,
                                foot_impact_buf_wp=single_foot_impact,
                                foot_yaw_rate_buf_wp=single_foot_yaw_rate,
                                actions_buf_wp=single_action_buf,
                                prev_action_wp=single_prev_action,
                                prev_dphi_wp=single_prev_dphi,
                                prev_qvel_wp=single_prev_qvel,
                                prev_qpos_wp=single_prev_qpos)

                wp.synchronize()
                reward = float(single_rew.numpy()[0, 0])
                self._update_certainty_entry(reward)
                phi = float(phi_wp_single.numpy()[0])
                freq = 1.0 / max(time.time() - t0, 1e-6)
                log["rewards"].append(reward)
                log["phases"].append(phi)
                log["qpos"].append(exec_d.qpos.numpy()[0].copy())
                log["qvel"].append(exec_d.qvel.numpy()[0].copy())

                pbar.set_postfix({
                    "rew": f"{reward:+.2e}",
                    "phi": f"{phi:.3f}",
                    "rdphi": f"{float(action[-1]):+.3f}",
                    "freq": f"{freq:.2f}",
                })

                Y = self._shift_Y_with_ref_np(Y, t, ref_actions_all)

                if phi >= 1.0 or (1.0 - phi) < dphi_star:
                    pbar.write(
                        f"episode terminated at step {t} "
                        f"(phi={phi:.4f}, dphi*={dphi_star:.4e})"
                    )
                    break

        self._summarise(log)
        return log


