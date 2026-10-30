"""Host-side numpy helpers (finite-diff velocities, quaternion rotate) for the
Warp tracking environment. No GPU / Warp — used during env construction."""
from __future__ import annotations

import numpy as np
from typing import Optional


def _quat_rotate_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v (3,) by quaternions q (T, 4) wxyz. Numpy port of the
    Warp `_rotate_vec_wxyz` func (assumes unit quaternions)."""
    q0, q1, q2, q3 = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    v0, v1, v2 = float(v[0]), float(v[1]), float(v[2])
    t0 = 2.0 * (q2 * v2 - q3 * v1)
    t1 = 2.0 * (q3 * v0 - q1 * v2)
    t2 = 2.0 * (q1 * v1 - q2 * v0)
    r0 = v0 + q0 * t0 + (q2 * t2 - q3 * t1)
    r1 = v1 + q0 * t1 + (q3 * t0 - q1 * t2)
    r2 = v2 + q0 * t2 + (q1 * t1 - q2 * t0)
    return np.stack([r0, r1, r2], axis=-1).astype(np.float32)


def _safe_round_range(lo: np.ndarray, hi: np.ndarray, round_to: float) -> tuple[np.ndarray, np.ndarray]:
    if round_to is None or round_to <= 0:
        return lo, hi
    lo_r = np.floor(lo / round_to) * round_to
    hi_r = np.ceil(hi / round_to) * round_to
    return lo_r, hi_r


def _compute_joint_range_from_qpos(
    joint_q: np.ndarray,
    *,
    margin_abs: float,
    margin_ratio: float,
    round_to: float,
    min_width: float,
    physical_range: Optional[np.ndarray] = None,
    physical_outset_ratio: float = 0.0,
    physical_outset_abs: float = 0.0,
) -> np.ndarray:
    """Compute per-joint [min,max] limits from a reference trajectory.

    Adds a safety margin and rounding, then clamps to physical_range if provided.

    physical_outset_{ratio,abs} extend the physical clamp OUTWARD before clamping —
    by `ratio * physical_span + abs` per joint — so the commanded PD target may sit that far
    past the mechanical stop (the sim still clamps the actual joint). This is what lets a
    position-PD controller generate static holding torque at a joint resting near its limit.
    ratio scales with each joint's travel; abs is a flat rad offset on every joint. 0/0 =
    clamp to the hard physical limits (default).
    """
    if joint_q.ndim != 2:
        raise ValueError(f"Expected joint_q shape (T, n_joints), got {joint_q.shape}")
    if joint_q.shape[0] < 1:
        raise ValueError("Need at least one frame to compute joint_range")

    if physical_range is not None and (physical_outset_ratio or physical_outset_abs):
        physical_range = np.asarray(physical_range, dtype=np.float32).copy()
        pspan = np.maximum(physical_range[:, 1] - physical_range[:, 0], 0.0)
        _outset = float(physical_outset_ratio) * pspan + float(physical_outset_abs)
        physical_range[:, 0] -= _outset
        physical_range[:, 1] += _outset

    min_ref = np.min(joint_q, axis=0)
    max_ref = np.max(joint_q, axis=0)
    span = np.maximum(max_ref - min_ref, 0.0)

    margin = np.maximum(float(margin_abs), span * float(margin_ratio))
    lo = min_ref - margin
    hi = max_ref + margin

    lo, hi = _safe_round_range(lo, hi, float(round_to))

    # Ensure a minimum width after rounding.
    width = hi - lo
    if min_width is not None and min_width > 0:
        mw = float(min_width)
        center = 0.5 * (lo + hi)
        lo = np.where(width < mw, center - 0.5 * mw, lo)
        hi = np.where(width < mw, center + 0.5 * mw, hi)

    # Clamp to physical joint limits where available.
    if physical_range is not None:
        phys = np.asarray(physical_range, dtype=np.float32)
        if phys.shape != (joint_q.shape[1], 2):
            raise ValueError(
                f"physical_range shape mismatch: expected {(joint_q.shape[1], 2)}, got {phys.shape}"
            )
        phys_lo = phys[:, 0]
        phys_hi = phys[:, 1]
        phys_valid = np.isfinite(phys_lo) & np.isfinite(phys_hi) & (phys_hi > phys_lo)
        lo = np.where(phys_valid, np.maximum(lo, phys_lo), lo)
        hi = np.where(phys_valid, np.minimum(hi, phys_hi), hi)

    # Final guard: if any joint ends up invalid, fall back to physical range if valid,
    # otherwise expand around the reference min/max.
    invalid = ~(hi > lo)
    if np.any(invalid):
        if physical_range is not None:
            phys = np.asarray(physical_range, dtype=np.float32)
            phys_valid = np.isfinite(phys[:, 0]) & np.isfinite(phys[:, 1]) & (phys[:, 1] > phys[:, 0])
            use_phys = invalid & phys_valid
            lo = np.where(use_phys, phys[:, 0], lo)
            hi = np.where(use_phys, phys[:, 1], hi)
        # still invalid => make a tiny window
        invalid2 = ~(hi > lo)
        if np.any(invalid2):
            eps = max(float(round_to) if round_to else 1e-3, 1e-3)
            center = 0.5 * (min_ref + max_ref)
            lo = np.where(invalid2, center - eps, lo)
            hi = np.where(invalid2, center + eps, hi)

    return np.stack([lo, hi], axis=1).astype(np.float32)
