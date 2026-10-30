"""The contract between any motion retargeter and the tracker.

A retargeter hands the tracker a *robot* motion: the floating-base pose plus one
angle per actuated joint, frame by frame.  `RobotMotion` is that hand-off, and
`write_reference` turns it into the one file a tracking config points at:

    <out-dir>/<name>_<hz>Hz.npz    state rows (qpos | qvel) and the world pose of every body
                                   per frame; what `motion_path` names

One importer per source lives next to this module (`soma_csv_to_reference.py`,
`lafan1_csv_to_reference.py`, `gmr_pkl_to_reference.py`,
`omniretarget_npz_to_reference.py`, `phuma_npy_to_reference.py`): each is a
reader that builds a `RobotMotion` plus a `main()` with the shared flags, so a
new source is a copy of one of them with the reader replaced. The handcrafted
generators in `handcrafted/` build a `RobotMotion` directly.  To connect your
own retargeter, construct a `RobotMotion` and call `write_reference`::

    from dtmr.data.robot_motion import RobotMotion, write_reference

    motion = RobotMotion(root_pos, root_quat_wxyz, joint_pos, joint_names, fps=30.0)
    write_reference(motion, robot="unitree_g1", out_dir=Path("data/my_clip"), name="my_clip")

Joint columns are matched to the robot model BY NAME, so the joint order of the
source does not matter.  Joints the source does not drive are held at zero
(at most `MAX_UNMATCHED_JOINTS` of them); joints the model does not have are
ignored.  Units: metres, radians, wxyz quaternions, frames at `fps`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from dtmr.data.resample_reference import (
    DEFAULT_TARGET_FPS,
    export_reference,
    _resample,
)
from dtmr.utils import get_model_path

DEFAULT_MODEL_FILE = "scene_mjx_torque_tracking.xml"
MAX_UNMATCHED_JOINTS = 2   # e.g. a head the source rig does not drive; held at zero


class MotionFormatError(ValueError):
    """The source file is not what the importer expected."""


@dataclass
class RobotMotion:
    root_pos: np.ndarray        # (T, 3) metres, world frame
    root_quat_wxyz: np.ndarray  # (T, 4) unit quaternions, scalar first
    joint_pos: np.ndarray       # (T, n) radians
    joint_names: Sequence[str]  # n names, matching the robot model's joint names
    fps: float

    def __post_init__(self) -> None:
        self.root_pos = np.asarray(self.root_pos, dtype=np.float64)
        self.root_quat_wxyz = np.asarray(self.root_quat_wxyz, dtype=np.float64)
        self.joint_pos = np.asarray(self.joint_pos, dtype=np.float64)
        self.joint_names = list(self.joint_names)
        T = len(self.root_pos)
        if self.root_pos.shape != (T, 3) or self.root_quat_wxyz.shape != (T, 4):
            raise MotionFormatError(f"root arrays must be (T,3) and (T,4), got "
                                    f"{self.root_pos.shape} and {self.root_quat_wxyz.shape}")
        if self.joint_pos.shape != (T, len(self.joint_names)):
            raise MotionFormatError(f"joint_pos is {self.joint_pos.shape}, expected "
                                    f"({T}, {len(self.joint_names)}) for the given joint_names")
        if T < 2:
            raise MotionFormatError("a motion needs at least two frames")
        if self.fps <= 0:
            raise MotionFormatError("fps must be positive")
        norms = np.linalg.norm(self.root_quat_wxyz, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-3):
            raise MotionFormatError("root quaternions are not unit length (wrong column order?)")

    @property
    def duration(self) -> float:
        return (len(self.root_pos) - 1) / self.fps

    def trim(self, start: float = 0.0, end: float | None = None) -> "RobotMotion":
        """The window [start, end] in seconds (end=None: to the last frame)."""
        i0 = max(0, int(round(start * self.fps)))
        i1 = len(self.root_pos) if end is None else min(len(self.root_pos), int(round(end * self.fps)) + 1)
        if i1 - i0 < 2:
            raise MotionFormatError(f"trim [{start}, {end}] s leaves fewer than two frames "
                                    f"(clip is {self.duration:.2f} s)")
        return RobotMotion(self.root_pos[i0:i1], self.root_quat_wxyz[i0:i1], self.joint_pos[i0:i1],
                           self.joint_names, self.fps)


def model_joint_names(robot: str, model_file: str = DEFAULT_MODEL_FILE) -> list[str]:
    """Hinge/slide joint names of the robot in qpos order."""
    model = mujoco.MjModel.from_xml_path(str(get_model_path(robot, model_file)))
    joints = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    joints.sort(key=lambda j: model.jnt_qposadr[j])
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Reorder (T, 4) quaternions from xyzw (GMR, PHUMA, Unitree's LAFAN1 CSVs) to scalar first."""
    return np.asarray(quat, dtype=np.float64)[:, [3, 0, 1, 2]]


def angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from consecutive wxyz quaternions."""
    out = np.zeros((len(quat_wxyz), 3))
    for i in range(1, len(quat_wxyz)):
        w0, x0, y0, z0 = quat_wxyz[i - 1] * np.array([1.0, -1.0, -1.0, -1.0])   # conjugate
        w1, x1, y1, z1 = quat_wxyz[i]
        dq = np.array([
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
        ])
        if dq[0] < 0:
            dq = -dq
        norm = np.linalg.norm(dq[1:])
        # atan2 stays well-conditioned for the tiny per-frame rotations of 120 fps data,
        # where 2*arccos(w) would amplify rounding in w.
        angle = 2.0 * np.arctan2(norm, dq[0])
        out[i] = (dq[1:] / norm if norm > 1e-12 else np.zeros(3)) * angle / dt
    if len(out) > 1:
        out[0] = out[1]
    return out


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    v = np.zeros_like(x)
    v[1:] = np.diff(x, axis=0) / dt
    if len(v) > 1:
        v[0] = v[1]
    return v


def match_joints(motion: RobotMotion, model_names: Sequence[str], *, tag: str = "motion",
                 max_unmatched: int | None = None) -> np.ndarray:
    """(T, len(model_names)) joint angles in the model's order, matched by name.

    `max_unmatched` (default MAX_UNMATCHED_JOINTS) is how many model joints the source may
    leave undriven; an importer for a rig that lacks, say, the wrists raises it.
    """
    if max_unmatched is None:
        max_unmatched = MAX_UNMATCHED_JOINTS
    source = {name: i for i, name in enumerate(motion.joint_names)}
    joints = np.zeros((len(motion.root_pos), len(model_names)))
    unmatched = []
    for j, name in enumerate(model_names):
        if name in source:
            joints[:, j] = motion.joint_pos[:, source[name]]
        else:
            unmatched.append(name)
    unused = [n for n in motion.joint_names if n not in model_names]
    if len(unmatched) > max_unmatched:
        raise MotionFormatError(
            f"{len(unmatched)} of {len(model_names)} model joints are missing from the {tag} "
            f"(first: {unmatched[:3]}; the {tag} has e.g. {unused[:3]}). Was the clip retargeted "
            f"to this robot?")
    if unmatched:
        print(f"[{tag}] joints held at zero (not in the source): {unmatched}")
    if unused:
        print(f"[{tag}] source joints not on this robot, ignored: {unused}")
    return joints


def motion_to_state(motion: RobotMotion, model_names: Sequence[str], *, tag: str = "motion",
                    max_unmatched: int | None = None) -> np.ndarray:
    """(T, 13 + 2n) state rows: root pos | root quat wxyz | joints | velocities."""
    joints = match_joints(motion, model_names, tag=tag, max_unmatched=max_unmatched)
    dt = 1.0 / motion.fps
    return np.hstack([motion.root_pos, motion.root_quat_wxyz, joints,
                      finite_difference(motion.root_pos, dt),
                      angular_velocity(motion.root_quat_wxyz, dt),
                      finite_difference(joints, dt)])


def write_reference(motion: RobotMotion, *, robot: str, out_dir: Path, name: str,
                    model_file: str = DEFAULT_MODEL_FILE, target_fps: float = DEFAULT_TARGET_FPS,
                    start: float = 0.0, end: float | None = None,
                    tag: str = "motion", max_unmatched: int | None = None) -> Path:
    """Trim, resample, run FK, write the reference npz.

    Returns the path of the `<name>_<hz>Hz.npz` to use as `motion_path`.
    """
    if start > 0.0 or end is not None:
        motion = motion.trim(start, end)
    names = model_joint_names(robot, model_file)
    state = motion_to_state(motion, names, tag=tag, max_unmatched=max_unmatched)
    timestamps, resampled = _resample(state.tolist(), 1.0 / motion.fps, 1.0 / target_fps)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{name}_{int(round(target_fps))}Hz.npz"
    export_reference(resampled, timestamps, npz, robot=robot, model_file=model_file, fps=target_fps)
    print(f"[{tag}] {len(state)} frames @ {motion.fps:g} Hz -> {len(resampled)} frames @ "
          f"{target_fps:g} Hz; set motion_path: {npz}")
    return npz


def add_reference_args(parser: argparse.ArgumentParser, *, robot_required: bool = True,
                       default_robot: str | None = None) -> None:
    """The output-side CLI flags every importer shares."""
    parser.add_argument("--robot", required=robot_required, default=default_robot,
                        help="robot descriptor / model directory: unitree_g1, unitree_h1_2, "
                             "unitree_r1, booster_t1, booster_k1, ...")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE,
                        help="MJCF scene inside the robot's model directory used for FK")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None, help="output basename (default: the input stem)")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS, help="tracker reference rate")
    parser.add_argument("--start", type=float, default=0.0, help="keep from this time (s) of the source")
    parser.add_argument("--end", type=float, default=None, help="keep up to this time (s) of the source")
