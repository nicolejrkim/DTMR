"""Convert a GMR robot-motion pickle into the tracker's reference files.

GMR (General Motion Retargeting, https://github.com/YanjieZe/GMR) saves one pickle per
clip from `scripts/bvh_to_robot.py`, `scripts/smplx_to_robot.py` and their `_dataset`
batch variants:

    fps          frame rate
    root_pos     (T, 3) metres
    root_rot     (T, 4) quaternion, xyzw
    dof_pos      (T, n) radians, in the joint order of GMR's robot model
    local_body_pos, link_body_list   unused here

GMR's models of the Unitree G1 (29 joints), Unitree H1-2 (27) and Booster T1 (23) list
the joints in the same order as the models in `src/dtmr/models/`. Its R1 model has 29
joints in the order written out below; the five wrist joints the model here lacks are
dropped. Its K1 uses other joint names and is not wired.

  python -m dtmr.data.gmr_pkl_to_reference data/gmr/walk_unitree_g1.pkl \\
      --robot unitree_g1 --out-dir data/gmr_walk
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from dtmr.data.robot_motion import (
    MotionFormatError,
    RobotMotion,
    add_reference_args,
    model_joint_names,
    write_reference,
    xyzw_to_wxyz,
)

KEYS = ("fps", "root_pos", "root_rot", "dof_pos")
ROBOTS = ("unitree_g1", "unitree_h1_2", "unitree_r1", "booster_t1")

# GMR's unitree_r1 model (assets/unitree_r1/r1_mocap_29dof.xml), joint order as saved.
R1_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def gmr_joint_names(robot: str) -> list[str]:
    """The joint order of GMR's model of `robot`."""
    if robot == "unitree_r1":
        return list(R1_JOINTS)
    if robot in ROBOTS:
        return model_joint_names(robot)      # verified identical to GMR's model order
    raise MotionFormatError(f"no GMR joint order is known for {robot!r}; this importer covers "
                            f"{list(ROBOTS)}. For another robot, add its order to gmr_joint_names.")


def read_gmr_pkl(path: Path, robot: str) -> RobotMotion:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or any(k not in data for k in KEYS):
        raise MotionFormatError(f"{path}: not a GMR motion pickle (needs {list(KEYS)})")
    names = gmr_joint_names(robot)
    joints = np.asarray(data["dof_pos"], dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != len(names):
        raise MotionFormatError(f"{path}: dof_pos is {joints.shape}, expected (T, {len(names)}) "
                                f"for GMR's {robot} model")
    return RobotMotion(data["root_pos"], xyzw_to_wxyz(data["root_rot"]), joints, names,
                       fps=float(data["fps"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_pkl", type=Path)
    add_reference_args(parser, robot_required=False, default_robot="unitree_g1")
    args = parser.parse_args()
    try:
        motion = read_gmr_pkl(args.input_pkl, args.robot)
        write_reference(motion, robot=args.robot, out_dir=args.out_dir,
                        name=args.name or args.input_pkl.stem, model_file=args.model_file,
                        target_fps=args.target_fps, start=args.start, end=args.end, tag="gmr")
    except MotionFormatError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
