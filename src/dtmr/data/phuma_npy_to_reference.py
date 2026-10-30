"""Convert a PHUMA clip (.npy) into the tracker's reference files.

PHUMA (https://github.com/DAVIAN-Robotics/PHUMA, dataset at
https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA) stores one dictionary per clip
as a `.npy` under `data/humanoid_pose/g1/`:

    fps          frame rate (30)
    root_trans   (T, 3) metres
    root_ori     (T, 4) quaternion, xyzw
    dof_pos      (T, 23) radians: the 29-DoF Unitree G1 without its six wrist joints

The wrists are held at zero; the importer raises the matcher's missing-joint allowance
for that and says so in its output.

  python -m dtmr.data.phuma_npy_to_reference data/humanoid_pose/g1/kick_chunk_0000.npy \\
      --out-dir data/phuma_kick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dtmr.data.robot_motion import (
    MotionFormatError,
    RobotMotion,
    add_reference_args,
    write_reference,
    xyzw_to_wxyz,
)

ROBOT = "unitree_g1"
KEYS = ("fps", "root_trans", "root_ori", "dof_pos")

# PHUMA's G1 model (asset/humanoid_model/g1): the 29-DoF model without the six wrist joints.
G1_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
)
MAX_UNMATCHED = 6   # the wrists


def read_phuma_npy(path: Path, robot: str = ROBOT) -> RobotMotion:
    if robot != ROBOT:
        raise MotionFormatError(f"this importer reads PHUMA's {ROBOT} clips; got --robot {robot}")
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    if not isinstance(data, dict) or any(k not in data for k in KEYS):
        raise MotionFormatError(f"{path}: not a PHUMA motion dict (needs {list(KEYS)})")
    joints = np.asarray(data["dof_pos"], dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != len(G1_JOINTS):
        raise MotionFormatError(f"{path}: dof_pos is {joints.shape}, expected (T, {len(G1_JOINTS)})")
    return RobotMotion(data["root_trans"], xyzw_to_wxyz(data["root_ori"]), joints, G1_JOINTS,
                       fps=float(data["fps"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_npy", type=Path)
    add_reference_args(parser, robot_required=False, default_robot=ROBOT)
    args = parser.parse_args()
    try:
        motion = read_phuma_npy(args.input_npy, args.robot)
        write_reference(motion, robot=args.robot, out_dir=args.out_dir,
                        name=args.name or args.input_npy.stem, model_file=args.model_file,
                        target_fps=args.target_fps, start=args.start, end=args.end,
                        tag="phuma", max_unmatched=MAX_UNMATCHED)
    except MotionFormatError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
