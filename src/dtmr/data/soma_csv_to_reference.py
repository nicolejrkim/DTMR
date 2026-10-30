"""Convert a soma-retargeter robot CSV into the tracker's reference files.

soma-retargeter (https://github.com/NVIDIA/soma-retargeter) writes one CSV per
clip: `Frame`, `root_translate{X,Y,Z}` in centimetres, `root_rotate{X,Y,Z}` as
XYZ Euler angles in degrees, and one column per joint in degrees, at the source
frame rate (120 fps for SEED). soma-retargeter v0.2.0 and later name a joint column
by the joint; v0.1.0 appended `_dof`. Both are read.

This tool reads that CSV into a `RobotMotion` (joints matched to the robot
model BY NAME, units converted) and writes the reference through
`robot_motion.write_reference`:

    <out-dir>/<clip>_<hz>Hz.npz    state rows (qpos | qvel) and link world poses; what `motion_path` points at

  python -m dtmr.data.soma_csv_to_reference clip.csv \\
      --robot unitree_g1 --out-dir data/clip
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dtmr.data.robot_motion import (   # re-exported, see __all__
    MAX_UNMATCHED_JOINTS,
    MotionFormatError,
    RobotMotion,
    add_reference_args,
    angular_velocity,
    finite_difference,
    model_joint_names,
    motion_to_state,
    write_reference,
)

__all__ = ["read_soma_csv", "soma_csv_to_state", "main", "MAX_UNMATCHED_JOINTS", "MotionFormatError",
           "RobotMotion", "angular_velocity", "finite_difference", "model_joint_names", "write_reference"]

CM_TO_M = 0.01
SOMA_FPS = 120.0
JOINT_SUFFIX = "_dof"           # v0.1.0 joint columns; v0.2.0 and later use bare joint names
ROOT_COLS = ("Frame", *[f"root_translate{a}" for a in "XYZ"], *[f"root_rotate{a}" for a in "XYZ"])


def read_soma_csv(path: Path, fps: float = SOMA_FPS) -> RobotMotion:
    """Parse a soma-retargeter CSV into metres / radians / wxyz."""
    with open(path) as f:
        rows = list(csv.reader(f))
    header, values = rows[0], np.asarray(rows[1:], dtype=np.float64)
    col = {name: i for i, name in enumerate(header)}
    need = [f"root_translate{a}" for a in "XYZ"] + [f"root_rotate{a}" for a in "XYZ"]
    if any(c not in col for c in need):
        raise MotionFormatError(f"{path}: not a soma-retargeter CSV (missing root columns)")

    pos = values[:, [col[f"root_translate{a}"] for a in "XYZ"]] * CM_TO_M
    euler = values[:, [col[f"root_rotate{a}"] for a in "XYZ"]]
    quat_wxyz = Rotation.from_euler("xyz", euler, degrees=True).as_quat()[:, [3, 0, 1, 2]]
    joint_cols = [h for h in header if h not in ROOT_COLS]
    if not joint_cols:
        raise MotionFormatError(f"{path}: no joint columns after the root columns")
    joints = np.deg2rad(values[:, [col[h] for h in joint_cols]])
    names = [h[: -len(JOINT_SUFFIX)] if h.endswith(JOINT_SUFFIX) else h for h in joint_cols]
    return RobotMotion(pos, quat_wxyz, joints, names, fps=fps)


def soma_csv_to_state(path: Path, joint_names: list[str], fps: float) -> np.ndarray:
    """(T, 13 + 2n) rows: root pos (m) | root quat wxyz | joints (rad) | velocities."""
    try:
        return motion_to_state(read_soma_csv(path, fps), joint_names, tag="soma")
    except MotionFormatError as e:
        sys.exit(str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path)
    add_reference_args(parser)
    parser.add_argument("--source-fps", type=float, default=SOMA_FPS, help="frame rate of the CSV (SEED: 120)")
    args = parser.parse_args()

    try:
        motion = read_soma_csv(args.input_csv, args.source_fps)
        write_reference(motion, robot=args.robot, out_dir=args.out_dir,
                        name=args.name or args.input_csv.stem, model_file=args.model_file,
                        target_fps=args.target_fps,
                        start=args.start, end=args.end, tag="soma")
    except MotionFormatError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
