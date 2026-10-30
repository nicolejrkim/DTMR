"""Phi-targeted tracking score for the Warp-backend tracking rollouts.

Companion to ``dtw.py``. Instead of finding an optimal alignment
(DTW), this evaluates how close the rollout was to the reference frame
phi *actually selected* at each step. Mirrors ``_phase_update_kernel`` in
``dtmr.env`` for the phase→frame_idx mapping.

The mapping is ``frame_idx = clip(floor(phi * T), 0, T-1)`` (no fractional
lerp — keeps the metric simple; truncation error is at most one ref-frame
duration).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import mujoco
import yaml


# --- phi-tracking metric (pure numpy) ---
def _quat_geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic angle (radians) between unit quaternions. Broadcasts over leading dims.

    Uses theta = 2 * arccos(|<q1, q2>|) with clipping. Sign of either quat is
    irrelevant since |.| handles the q ~ -q ambiguity.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    n1 = np.linalg.norm(q1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(q2, axis=-1, keepdims=True)
    q1 = q1 / np.clip(n1, 1e-12, None)
    q2 = q2 / np.clip(n2, 1e-12, None)
    dot = np.sum(q1 * q2, axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), -1.0, 1.0))


def phi_tracking_metrics_from_arrays(
    cur_link_pos: np.ndarray,
    ref_link_pos: np.ndarray,
    cur_link_quat: np.ndarray | None = None,
    ref_link_quat: np.ndarray | None = None,
    cur_qpos: np.ndarray | None = None,
    ref_qpos_aligned: np.ndarray | None = None,
    base_dims: int = 7,
) -> dict:
    """Pure-numpy MPJPE-style metrics for phi-targeted (unwarped) tracking.

    Arrays must already be 1:1 aligned along axis 0: row t of every cur_* and
    ref_* input corresponds to simulation step t, with `ref_*` reflecting the
    reference frame phi targeted at that step. No DTW; no other alignment.

    Args:
        cur_link_pos, ref_link_pos: (T, n_links, 3) link world positions.
        cur_link_quat, ref_link_quat: optional (T, n_links, 4), wxyz convention.
        cur_qpos, ref_qpos_aligned: optional (T, nq). ref_qpos_aligned[t] must
            already be the reference qpos at the phi-targeted frame of step t.
        base_dims: number of leading qpos dims occupied by the floating base.
            Default 7 = 3 translation + 4 quaternion (MuJoCo free joint).

    Returns:
        dict keyed by `phi/*` scalars. Empty dict if inputs are unusable.
    """
    out: dict = {}

    cur_link_pos = np.asarray(cur_link_pos, dtype=np.float32)
    ref_link_pos = np.asarray(ref_link_pos, dtype=np.float32)
    if cur_link_pos.ndim != 3 or ref_link_pos.ndim != 3:
        print(f"phi_tracking: bad link_pos shapes cur={cur_link_pos.shape} ref={ref_link_pos.shape}; skipping")
        return out
    if cur_link_pos.shape[1:] != ref_link_pos.shape[1:]:
        print(f"phi_tracking: link dims mismatch cur={cur_link_pos.shape[1:]} ref={ref_link_pos.shape[1:]}; skipping")
        return out

    T = int(min(cur_link_pos.shape[0], ref_link_pos.shape[0]))
    if T == 0:
        return out
    if cur_link_pos.shape[0] != ref_link_pos.shape[0]:
        print(f"phi_tracking: length mismatch cur={cur_link_pos.shape[0]} ref={ref_link_pos.shape[0]}; truncating to {T}")
    cur_p = cur_link_pos[:T]
    ref_p = ref_link_pos[:T]

    per_link_err = np.linalg.norm(cur_p - ref_p, axis=-1)
    out["phi/mpjpe_world_m"] = float(per_link_err.mean())

    cur_centroid = cur_p.mean(axis=1, keepdims=True)
    ref_centroid = ref_p.mean(axis=1, keepdims=True)
    relcom_err = np.linalg.norm((cur_p - cur_centroid) - (ref_p - ref_centroid), axis=-1)
    out["phi/mpjpe_relcom_m"] = float(relcom_err.mean())

    if cur_link_quat is not None and ref_link_quat is not None:
        cur_q = np.asarray(cur_link_quat, dtype=np.float32)
        ref_q = np.asarray(ref_link_quat, dtype=np.float32)
        if cur_q.ndim == 3 and ref_q.ndim == 3 and cur_q.shape[1:] == ref_q.shape[1:]:
            Tq = min(cur_q.shape[0], ref_q.shape[0], T)
            if Tq > 0:
                angles = _quat_geodesic_angle(cur_q[:Tq], ref_q[:Tq])
                out["phi/mean_orient_err_rad"] = float(angles.mean())
                out["phi/mean_orient_err_deg"] = float(np.degrees(angles.mean()))

    if cur_qpos is not None and ref_qpos_aligned is not None:
        cq = np.asarray(cur_qpos, dtype=np.float64)
        rq = np.asarray(ref_qpos_aligned, dtype=np.float64)
        if cq.ndim == 2 and rq.ndim == 2 and cq.shape[1] == rq.shape[1]:
            Tq = min(cq.shape[0], rq.shape[0])
            if Tq > 0 and cq.shape[1] >= base_dims:
                cq = cq[:Tq]
                rq = rq[:Tq]
                if base_dims >= 3:
                    base_pos_err = np.linalg.norm(cq[:, :3] - rq[:, :3], axis=-1)
                    out["phi/qpos_base_pos_err_m"] = float(base_pos_err.mean())
                if base_dims >= 7:
                    base_ang = _quat_geodesic_angle(cq[:, 3:7], rq[:, 3:7])
                    out["phi/qpos_base_orient_err_rad"] = float(base_ang.mean())
                    out["phi/qpos_base_orient_err_deg"] = float(np.degrees(base_ang.mean()))
                if cq.shape[1] > base_dims:
                    joint_err = np.abs(cq[:, base_dims:] - rq[:, base_dims:])
                    out["phi/qpos_joint_mae_rad"] = float(joint_err.mean())

    out["phi/n_steps"] = int(T)
    return out


def _replay_link_poses(
    mj_model: mujoco.MjModel,
    qpos_log: np.ndarray,
    body_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay qpos through mj_kinematics; return (xpos, xquat) at body_indices.

    Shapes: xpos (T, n_links, 3), xquat (T, n_links, 4) — both float32.
    """
    data = mujoco.MjData(mj_model)
    T = int(qpos_log.shape[0])
    n_links = int(body_indices.shape[0])
    xpos = np.empty((T, n_links, 3), dtype=np.float32)
    xquat = np.empty((T, n_links, 4), dtype=np.float32)
    for t in range(T):
        data.qpos[:] = qpos_log[t]
        mujoco.mj_kinematics(mj_model, data)
        xpos[t] = data.xpos[body_indices].astype(np.float32)
        xquat[t] = data.xquat[body_indices].astype(np.float32)
    return xpos, xquat


def _phase_to_frame_idx(phases: np.ndarray, T_ref: int) -> np.ndarray:
    """Mirror env._phase_update_kernel: floor(phi * T_ref), clipped to [0, T_ref-1]."""
    phases = np.asarray(phases, dtype=np.float64)
    f_cont = phases * float(T_ref)
    fidx = np.floor(f_cont).astype(np.int64)
    return np.clip(fidx, 0, max(T_ref - 1, 0))


def compute_run_phi_tracking(
    env,
    qpos_log: np.ndarray,
    phases: np.ndarray,
    output_dir: str,
    timestamp: str,
) -> Optional[dict]:
    """Compute phi-targeted MPJPE metrics for a Warp-backend run.

    Writes ``{output_dir}/{timestamp}_phi_tracking_metrics.yaml`` and returns
    the metrics dict. Returns None if mandatory env data is missing.
    """
    qpos_log = np.asarray(qpos_log, dtype=np.float32)
    phases = np.asarray(phases, dtype=np.float32).reshape(-1)
    if qpos_log.ndim != 2 or qpos_log.shape[0] == 0:
        print(f"[phi_tracking] qpos_log must be (T, nq); got {qpos_log.shape}; skipping")
        return None

    ref_pos = getattr(env, "_ref_pos_np", None)
    ref_quat = getattr(env, "_ref_quat_np", None)
    body_indices = getattr(env, "body_indices_np", None)
    mj_model = getattr(env, "mj_model", None)
    if ref_pos is None or body_indices is None or mj_model is None:
        print("[phi_tracking] env missing _ref_pos_np / body_indices_np / mj_model; skipping")
        return None

    ref_pos = np.asarray(ref_pos, dtype=np.float32)
    T_ref = int(ref_pos.shape[0])
    T_sim = int(min(qpos_log.shape[0], phases.shape[0]))
    if T_sim == 0 or T_ref == 0:
        return None
    if qpos_log.shape[0] != phases.shape[0]:
        print(f"[phi_tracking] qpos len ({qpos_log.shape[0]}) != phases len ({phases.shape[0]}); truncating to {T_sim}")

    qpos_log = qpos_log[:T_sim]
    phases = phases[:T_sim]
    frame_idx = _phase_to_frame_idx(phases, T_ref)

    rollout_xpos, rollout_xquat = _replay_link_poses(mj_model, qpos_log, body_indices)
    ref_xpos_aligned = ref_pos[frame_idx]
    ref_xquat_aligned = None
    if ref_quat is not None:
        rq = np.asarray(ref_quat, dtype=np.float32)
        if rq.shape[0] == T_ref:
            ref_xquat_aligned = rq[frame_idx]

    ref_qpos_aligned = None
    state_ref = getattr(env, "_state_ref", None)
    if isinstance(state_ref, dict) and "qpos" in state_ref:
        try:
            ref_qpos_full = np.asarray(state_ref["qpos"], dtype=np.float32)
            if ref_qpos_full.ndim == 2 and ref_qpos_full.shape[1] == qpos_log.shape[1]:
                # _state_ref may be shorter/longer than _ref_pos_np; reclip
                fidx_q = np.clip(frame_idx, 0, ref_qpos_full.shape[0] - 1)
                ref_qpos_aligned = ref_qpos_full[fidx_q]
        except Exception as e:
            print(f"[phi_tracking] qpos alignment failed: {e}; skipping qpos terms")
            ref_qpos_aligned = None

    metrics = phi_tracking_metrics_from_arrays(
        cur_link_pos=rollout_xpos,
        ref_link_pos=ref_xpos_aligned,
        cur_link_quat=rollout_xquat,
        ref_link_quat=ref_xquat_aligned,
        cur_qpos=qpos_log,
        ref_qpos_aligned=ref_qpos_aligned,
    )
    if not metrics:
        return None

    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, f"{timestamp}_phi_tracking_metrics.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    parts = [f"mpjpe_world={metrics.get('phi/mpjpe_world_m', float('nan')):.4f}m",
             f"mpjpe_relcom={metrics.get('phi/mpjpe_relcom_m', float('nan')):.4f}m"]
    if "phi/mean_orient_err_deg" in metrics:
        parts.append(f"orient={metrics['phi/mean_orient_err_deg']:.2f}deg")
    if "phi/qpos_joint_mae_rad" in metrics:
        parts.append(f"qpos_joint={metrics['phi/qpos_joint_mae_rad']:.4f}rad")
    print(f"[phi_tracking] wrote {yaml_path} ({' '.join(parts)}, n={metrics.get('phi/n_steps', 0)})")
    return metrics
