"""resample_reference.py

Resample a robot state CSV from its recorded rate to the tracker's reference
rate (50 Hz by default) and write the reference file the tracker reads: one npz
holding the state rows and the world pose of every body (MuJoCo forward
kinematics on the robot's tracking scene, quaternions wxyz).

CSV layout, one row per frame, no header:
    * root position (3)
    * root quaternion (4), wxyz (or xyzw with --quat-format xyzw)
    * joint positions (N), in the tracking model's joint order
    * root linear velocity (3)
    * root angular velocity (3)
    * joint velocities (N)

Reference npz keys: qpos (T, nq), qvel (T, nv), fps, pos (T, B, 3), rot (T, B, 4)
wxyz, link_names (B,), link_idx, timestamps, include_base.

Usage:
python -m dtmr.data.resample_reference data/my_clip/my_clip.csv --robot unitree_g1 --input-fps 30
"""

from __future__ import annotations

import argparse
import csv
import math
import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple

import mujoco

DEFAULT_TARGET_FPS = 50.0   # the tracker's reference rate

ROOT_POS_WIDTH = 3
ROOT_QUAT_WIDTH = 4
ROOT_QPOS_WIDTH = ROOT_POS_WIDTH + ROOT_QUAT_WIDTH




def _read_csv(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        for line_idx, row in enumerate(reader, start=1):
            if not row:
                continue
            try:
                rows.append([float(val) for val in row])
            except ValueError as exc:  # pragma: no cover - defensive parsing
                raise ValueError(f"Failed to parse numeric values on line {line_idx}") from exc
    if len(rows) < 2:
        raise ValueError("Need at least two frames to resample.")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("CSV rows must all share the same width.")
    return rows


def _lerp_array(a: Sequence[float], b: Sequence[float], u: float) -> List[float]:
    return [x + (y - x) * u for x, y in zip(a, b)]


def _normalize_quat(q: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-9:
        raise ValueError("Encountered near-zero quaternion norm.")
    inv = 1.0 / norm
    return [v * inv for v in q]


def _slerp(q0: Sequence[float], q1: Sequence[float], u: float) -> List[float]:
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = [-v for v in q1]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if 1.0 - dot < 1e-6:
        return _normalize_quat(_lerp_array(q0, q1, u))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-8:
        return _normalize_quat(_lerp_array(q0, q1, u))
    w0 = math.sin((1.0 - u) * theta) / sin_theta
    w1 = math.sin(u * theta) / sin_theta
    return _normalize_quat([w0 * a + w1 * b for a, b in zip(q0, q1)])


def _interpolate_rows(row0: Sequence[float], row1: Sequence[float], u: float) -> List[float]:
    new_row: List[float] = []
    new_row.extend(_lerp_array(row0[:ROOT_POS_WIDTH], row1[:ROOT_POS_WIDTH], u))
    quat0 = row0[ROOT_POS_WIDTH:ROOT_QPOS_WIDTH]
    quat1 = row1[ROOT_POS_WIDTH:ROOT_QPOS_WIDTH]
    new_row.extend(_slerp(quat0, quat1, u))
    new_row.extend(_lerp_array(row0[ROOT_QPOS_WIDTH:], row1[ROOT_QPOS_WIDTH:], u))
    return new_row


def _sample_at_time(rows: Sequence[Sequence[float]], frame_dt: float, t: float) -> List[float]:
    max_index = len(rows) - 1
    duration = max_index * frame_dt
    if t <= 0.0:
        return list(rows[0])
    if t >= duration:
        return list(rows[-1])
    exact_idx = t / frame_dt
    i0 = min(int(math.floor(exact_idx)), max_index - 1)
    i1 = i0 + 1
    t0 = i0 * frame_dt
    t1 = t0 + frame_dt
    if t1 <= t0 or abs(t - t0) < 1e-12:
        return list(rows[i0])
    u = (t - t0) / (t1 - t0)
    return _interpolate_rows(rows[i0], rows[i1], u)


def _resample(rows: Sequence[Sequence[float]], frame_dt: float, target_dt: float) -> Tuple[List[float], List[List[float]]]:
    if target_dt <= 0.0:
        raise ValueError("target_dt must be positive.")
    duration = (len(rows) - 1) * frame_dt
    steps = int(math.floor(duration / target_dt))
    times = [i * target_dt for i in range(steps + 1)]
    if not math.isclose(times[-1], duration, rel_tol=0.0, abs_tol=1e-9):
        times.append(duration)
    resampled = [_sample_at_time(rows, frame_dt, t) for t in times]
    return times, resampled


def derive_output_path(input_path: Path, target_hz: float) -> Path:
    return input_path.with_name(f"{input_path.stem}_{int(target_hz)}Hz.npz")


def _infer_joint_dofs(width: int) -> int:
    remainder = width - (ROOT_QPOS_WIDTH + 6)
    if remainder <= 0 or remainder % 2 != 0:
        raise ValueError(
            "CSV width does not match expected layout "
            f"(got {width} columns, cannot infer joint count)."
        )
    return remainder // 2


def _link_world_poses(
    rows: Sequence[Sequence[float]],
    robot: str,
    model_file: str,
    quat_format: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Forward kinematics with MuJoCo: world position + orientation of every body
    (except the world body) for each state row.

    Returns (pos (T, B, 3), rot (T, B, 4) in wxyz, link_names (B,)) with B =
    nbody - 1, in MuJoCo body order (root body first).
    """
    from dtmr.utils import get_model_path

    model = mujoco.MjModel.from_xml_path(str(get_model_path(robot, model_file)))
    data = mujoco.MjData(model)

    arr = np.asarray(rows, dtype=np.float64)
    joint_dofs = _infer_joint_dofs(arr.shape[1])
    q = arr[:, : ROOT_QPOS_WIDTH + joint_dofs].copy()
    if quat_format == "xyzw":
        q[:, 3:7] = q[:, [6, 3, 4, 5]]  # -> wxyz for MuJoCo
    if q.shape[1] != model.nq:
        raise ValueError(
            f"State width {q.shape[1]} (7 + {joint_dofs} joints) does not match "
            f"{robot}/{model_file} nq={model.nq}"
        )

    T = q.shape[0]
    B = model.nbody - 1
    pos = np.zeros((T, B, 3), dtype=np.float32)
    rot = np.zeros((T, B, 4), dtype=np.float32)
    for t in range(T):
        data.qpos[:] = q[t]
        mujoco.mj_kinematics(model, data)
        pos[t] = data.xpos[1:]
        rot[t] = data.xquat[1:]
    link_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
                  for b in range(1, model.nbody)]
    return pos, rot, link_names


def write_reference_npz(
    npz_path: Path,
    rows: Sequence[Sequence[float]],
    timestamps: Sequence[float],
    pos: np.ndarray,
    rot: np.ndarray,
    link_names: Sequence[str],
    fps: float,
    include_base: bool = True,
) -> None:
    """Write the reference file: state rows (qpos | qvel) plus the link world poses."""
    offset = 0
    if not include_base:
        pos, rot, link_names, offset = pos[:, 1:], rot[:, 1:], list(link_names)[1:], 1
    T, B, _ = pos.shape
    timestamps = list(timestamps) if timestamps is not None else list(range(T))
    if len(timestamps) != T:
        raise ValueError("Timestamp count must match pose sequence length.")
    state = np.asarray(rows, dtype=np.float64)
    if state.shape[0] != T:
        raise ValueError("State row count must match pose sequence length.")
    nq = ROOT_QPOS_WIDTH + _infer_joint_dofs(state.shape[1])
    np.savez_compressed(
        npz_path,
        qpos=state[:, :nq].astype(np.float32),
        qvel=state[:, nq:].astype(np.float32),
        fps=np.array(float(fps)),
        pos=pos,
        rot=rot,
        link_names=np.array(list(link_names)),
        link_idx=np.array(range(offset, offset + B)),
        timestamps=np.array(timestamps),
        include_base=np.array(include_base),
    )
    print(f"Saved reference to {npz_path} (frames={T}, links={B})")


def export_reference(
    rows: Sequence[Sequence[float]],
    timestamps: Sequence[float],
    npz_path: Path,
    robot: str = "unitree_g1",
    model_file: str = "scene_mjx_torque_tracking.xml",
    fps: float = DEFAULT_TARGET_FPS,
    quat_format: str = "wxyz",
    include_base: bool = True,
) -> Path:
    """Run FK on the state rows and write the reference npz; returns its path."""
    pos, rot, link_names = _link_world_poses(rows, robot, model_file, quat_format)
    write_reference_npz(Path(npz_path), rows, timestamps, pos, rot, link_names, fps, include_base=include_base)
    return Path(npz_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample a robot state CSV reference and export link world poses (MuJoCo FK).")
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV exported at the original rate.")
    parser.add_argument(
        "--robot", type=str, default="unitree_g1",
        help="Model directory under src/dtmr/models used for the link-pose forward "
             "kinematics (unitree_g1, unitree_h1_2, unitree_r1, booster_t1, booster_k1, ...).")
    parser.add_argument(
        "--model-file", type=str, default="scene_mjx_torque_tracking.xml",
        help="MJCF scene inside --robot's directory (default: the tracking scene).")
    parser.add_argument("--input-fps", type=float, required=True, help="Recording frequency of the CSV (Hz).")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS,
                        help=f"Output frequency (Hz; default {DEFAULT_TARGET_FPS:g}, the tracker reference rate).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <input stem>_<target Hz>Hz.npz next to the input file.",
    )
    parser.add_argument(
        "--quat-format",
        type=str,
        default="wxyz",
        choices=("xyzw", "wxyz"),
        help=(
            "Quaternion ordering used by the input CSV root quaternion. "
            "Use 'wxyz' if your CSV stores (w,x,y,z); use 'xyzw' if it stores (x,y,z,w)."
        ),
    )
    parser.add_argument(
        "--no-base-link",
        action="store_true",
        help="Exclude the world/base link from the exported link poses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input_csv
    frame_dt = 1.0 / args.input_fps
    target_dt = 1.0 / args.target_fps

    rows = _read_csv(input_path)
    timestamps, resampled = _resample(rows, frame_dt, target_dt)
    output_path = args.output or derive_output_path(input_path, args.target_fps)
    export_reference(
        resampled, timestamps, output_path,
        robot=args.robot, model_file=args.model_file, fps=args.target_fps,
        quat_format=args.quat_format, include_base=not args.no_base_link,
    )
    print(f"{len(resampled)} frames @ {args.target_fps:g} Hz from {len(rows)} @ {args.input_fps:g} Hz; "
          f"set motion_path: {output_path}")


if __name__ == "__main__":
    main()
