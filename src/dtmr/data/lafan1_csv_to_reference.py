"""Convert a LAFAN1 clip already retargeted to a Unitree robot into the tracker's reference files.

Source: the LAFAN1 Retargeting Dataset
(https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), Unitree's
IK retargeting of Ubisoft's LAFAN1 mocap to the G1, H1 and H1_2.  Each
`<robot>/<clip>.csv` is header-less, one row per frame at 30 fps:

    root x y z (m) | root quaternion qx qy qz qw | joint angles (rad) in the order below

The dataset is a derivative of LAFAN1 and inherits its CC BY-NC-ND 4.0 licence;
download it yourself and keep the attribution.  The G1 column order is written
out here so the joints can be matched to the model by name, which also makes a
wrong `--robot` fail loudly instead of silently permuting limbs.

  python -m dtmr.data.lafan1_csv_to_reference g1/walk1_subject1.csv \\
      --robot unitree_g1 --out-dir data/lafan1_walk1
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

LAFAN1_FPS = 30.0
ROOT_COLS = 7   # xyz + xyzw

# The dataset's G1 CSVs (README) list the 29 joints in the G1 tracking model's qpos order.
G1_JOINT_ORDER = tuple(model_joint_names("unitree_g1"))

JOINT_ORDERS = {"unitree_g1": G1_JOINT_ORDER}


def read_lafan1_csv(path: Path, joint_order: tuple[str, ...] = G1_JOINT_ORDER,
                    fps: float = LAFAN1_FPS) -> RobotMotion:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    width = ROOT_COLS + len(joint_order)
    if values.shape[1] != width:
        raise MotionFormatError(f"{path}: {values.shape[1]} columns, expected {width} "
                                f"(root pose 7 + {len(joint_order)} joints)")
    pos = values[:, 0:3]
    quat_wxyz = values[:, [6, 3, 4, 5]]          # dataset stores xyzw
    return RobotMotion(pos, quat_wxyz, values[:, ROOT_COLS:], joint_order, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path)
    add_reference_args(parser, robot_required=False, default_robot="unitree_g1")
    parser.add_argument("--source-fps", type=float, default=LAFAN1_FPS)
    args = parser.parse_args()

    if args.robot not in JOINT_ORDERS:
        sys.exit(f"no LAFAN1 column order is defined for {args.robot!r}; this importer covers "
                 f"{sorted(JOINT_ORDERS)}. For another robot, add its order to JOINT_ORDERS.")
    try:
        motion = read_lafan1_csv(args.input_csv, JOINT_ORDERS[args.robot], args.source_fps)
        write_reference(motion, robot=args.robot, out_dir=args.out_dir,
                        name=args.name or args.input_csv.stem, model_file=args.model_file,
                        target_fps=args.target_fps,
                        start=args.start, end=args.end, tag="lafan1")
    except MotionFormatError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
