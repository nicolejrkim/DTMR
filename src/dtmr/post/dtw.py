"""Post-run DTW scoring for the Warp-backend tracking rollouts.

Replays a rollout's `qpos` trajectory through MuJoCo forward-kinematics to
recover link world positions, aligns against the reference link trajectory
(`env._ref_pos_np`) with fastdtw (`dtw_distance`), and writes a small YAML summary.

The Warp entry point calls it after saving `*_log.npz`; a cached run can be
re-scored by passing the qpos loaded from that npz.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import mujoco
import yaml
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


@dataclass(frozen=True)
class DTWResult:
    total_cost: float
    path_length: int
    mean_cost: float
    n_a: int
    n_b: int


def _as_2d_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"DTW expects a 2D array (T, D), got shape {x.shape}")
    if x.size == 0:
        raise ValueError("DTW expects non-empty sequences")
    return x.astype(np.float32, copy=False)


def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    band: Optional[int] = None,
    return_path: bool = False,
) -> Tuple[DTWResult, Optional[np.ndarray]]:
    """Computes DTW distance between sequences a and b using fastdtw.

    Args:
        a: Array of shape (N, D)
        b: Array of shape (M, D)
        band: Optional radius parameter for fastdtw. Controls the size of the
            neighborhood when expanding the path. Default is 1 if not specified.
        return_path: If True, returns the optimal warping path as int32 array
            of shape (K, 2) with (i, j) pairs.

    Returns:
        (result, path)
    """
    a2 = _as_2d_float32(a)
    b2 = _as_2d_float32(b)

    n, d = a2.shape
    m, d2 = b2.shape
    if d != d2:
        raise ValueError(f"Feature dims mismatch: a has {d}, b has {d2}")

    # fastdtw radius parameter (similar to Sakoe-Chiba band)
    radius = band if band is not None else 1

    # Compute DTW using fastdtw
    total_cost, path = fastdtw(a2, b2, radius=radius, dist=euclidean)

    path_length = len(path)
    mean_cost = float(total_cost / max(path_length, 1))

    result = DTWResult(
        total_cost=float(total_cost),
        path_length=path_length,
        mean_cost=mean_cost,
        n_a=int(n),
        n_b=int(m),
    )

    if not return_path:
        return result, None

    return result, np.asarray(path, dtype=np.int32)


def _rollout_link_positions(
    mj_model: mujoco.MjModel,
    qpos_log: np.ndarray,
    body_indices: np.ndarray,
) -> np.ndarray:
    """Replay qpos through mj_kinematics and gather xpos at body_indices.

    Returns (T_sim, n_links, 3) float32.
    """
    data = mujoco.MjData(mj_model)
    n_frames = int(qpos_log.shape[0])
    n_links = int(body_indices.shape[0])
    out = np.empty((n_frames, n_links, 3), dtype=np.float32)
    for t in range(n_frames):
        data.qpos[:] = qpos_log[t]
        mujoco.mj_kinematics(mj_model, data)
        out[t] = data.xpos[body_indices].astype(np.float32)
    return out


def compute_run_dtw(
    env,
    qpos_log: np.ndarray,
    rewards: Optional[np.ndarray],
    output_dir: str,
    timestamp: str,
    band: int = 4,
    save_path: bool = False,
) -> dict:
    """Compute DTW position error between rollout and reference.

    Writes `{output_dir}/{timestamp}_dtw_metrics.yaml` (scalars only) and,
    when `save_path=True`, also `{output_dir}/{timestamp}_dtw_path.npy` with
    the warping path as (rollout_idx, ref_idx) int32 pairs. The returned
    dict also carries the path under key `"dtw_path"` so callers can reuse
    it without reloading from disk.
    """
    qpos_log = np.asarray(qpos_log, dtype=np.float32)
    if qpos_log.ndim != 2 or qpos_log.shape[0] == 0:
        raise ValueError(
            f"qpos_log must be (T, nq); got shape {qpos_log.shape}"
        )

    rollout_xyz = _rollout_link_positions(
        env.mj_model, qpos_log, env.body_indices_np,
    )
    ref_xyz = np.asarray(env._ref_pos_np, dtype=np.float32)
    if ref_xyz.shape[1:] != rollout_xyz.shape[1:]:
        raise ValueError(
            f"reference link shape {ref_xyz.shape[1:]} != rollout link shape "
            f"{rollout_xyz.shape[1:]}"
        )

    t_sim, n_links, _ = rollout_xyz.shape
    rollout_flat = rollout_xyz.reshape(t_sim, n_links * 3)
    ref_flat = ref_xyz.reshape(ref_xyz.shape[0], n_links * 3)

    result, path = dtw_distance(rollout_flat, ref_flat, band=band, return_path=True)

    mean_reward = None
    if rewards is not None and len(rewards):
        mean_reward = float(np.mean(rewards))

    metrics_yaml = {
        "total_cost": float(result.total_cost),
        "mean_cost": float(result.mean_cost),
        "path_length": int(result.path_length),
        "n_rollout": int(result.n_a),
        "n_ref": int(result.n_b),
        "mean_reward": mean_reward,
    }

    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, f"{timestamp}_dtw_metrics.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(metrics_yaml, f, sort_keys=False)
    print(f"[dtw] wrote {yaml_path} "
          f"(mean_cost={metrics_yaml['mean_cost']:.4f}, "
          f"path_len={metrics_yaml['path_length']})")

    if save_path and path is not None:
        npy_path = os.path.join(output_dir, f"{timestamp}_dtw_path.npy")
        np.save(npy_path, np.asarray(path, dtype=np.int32))
        print(f"[dtw] wrote {npy_path}")

    metrics = dict(metrics_yaml)
    metrics["dtw_path"] = np.asarray(path, dtype=np.int32) if path is not None else None
    return metrics
