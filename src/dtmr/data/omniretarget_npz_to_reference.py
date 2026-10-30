"""Convert an OmniRetarget trajectory (.npz) into the tracker's reference files.

OmniRetarget (dataset: https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset;
code: `holosoma_retargeting` in https://github.com/amazon-far/holosoma) saves one npz
per clip:

    fps     frame rate
    qpos    (T, 36) or (T, 43): base quaternion qw qx qy qz | base position x y z |
            29 joints in the Unitree g1_29dof URDF order (the order of the model here)
            [| object pose, 7, when the clip has an object]

The retargeter's own output adds `human_joints` and `cost`, which are ignored. Only
the Unitree G1 is released.

  python -m dtmr.data.omniretarget_npz_to_reference robot-terrain/scene_04.npz \\
      --out-dir data/omni_scene04
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
    model_joint_names,
    write_reference,
)

ROBOT = "unitree_g1"
BASE, JOINTS = 7, 29


def read_omniretarget_npz(path: Path, robot: str = ROBOT) -> RobotMotion:
    if robot != ROBOT:
        raise MotionFormatError(f"OmniRetarget releases {ROBOT} motions only; got --robot {robot}")
    with np.load(path) as data:
        if "qpos" not in data or "fps" not in data:
            raise MotionFormatError(f"{path}: not an OmniRetarget npz (needs qpos and fps)")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        fps = float(data["fps"])
    if qpos.ndim != 2 or qpos.shape[1] not in (BASE + JOINTS, BASE + JOINTS + 7):
        raise MotionFormatError(f"{path}: qpos is {qpos.shape}, expected (T, {BASE + JOINTS}) "
                                f"or (T, {BASE + JOINTS + 7})")
    names = model_joint_names(ROBOT)
    return RobotMotion(qpos[:, 4:7], qpos[:, 0:4], qpos[:, BASE:BASE + JOINTS], names, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_npz", type=Path)
    add_reference_args(parser, robot_required=False, default_robot=ROBOT)
    args = parser.parse_args()
    try:
        motion = read_omniretarget_npz(args.input_npz, args.robot)
        write_reference(motion, robot=args.robot, out_dir=args.out_dir,
                        name=args.name or args.input_npz.stem, model_file=args.model_file,
                        target_fps=args.target_fps, start=args.start, end=args.end, tag="omniretarget")
    except MotionFormatError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
