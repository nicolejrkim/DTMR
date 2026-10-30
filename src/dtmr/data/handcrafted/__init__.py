"""Handcrafted reference motions: no human mocap, no retargeter.

Each generator plans a motion in task space (world-pinned footholds, ballistic
flights), solves the legs with damped least-squares IK against the robot model
and writes the tracker's reference files.  They exist so the tracker can be
tried without downloading any dataset, and as stress clips whose dynamics are
CLI dials (speed, flight time, jump distance) rather than whatever the mocap
happened to contain.

    sprint_g1       stand -> accelerate -> steady sprint -> decelerate -> stand
    jump_g1         N consecutive standing broad jumps
    triplejump_g1   run-up -> hop -> step -> jump (World Athletics pattern)

All three encode Unitree G1 geometry (foot offsets, ankle height, joint names).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dtmr.data.robot_motion import RobotMotion, model_joint_names, write_reference
from dtmr.utils import get_model_path

ROBOT = "unitree_g1"
G1_XML = get_model_path(ROBOT, "g1_mjx_torque.xml")   # the tracker's own G1 model (no scene)
ROOT_WIDTH = 7                                         # free joint: xyz + wxyz


def add_output_args(parser: argparse.ArgumentParser, *, default_name: str) -> None:
    parser.add_argument("--out-dir", type=Path, default=Path("data/handcrafted"))
    parser.add_argument("--name", default=default_name, help="output basename")


def save_reference(qpos: np.ndarray, fps: float, args: argparse.Namespace, *, tag: str) -> Path:
    """Write a (T, 7 + n) qpos trajectory as the tracker's reference files."""
    qpos = np.asarray(qpos, dtype=np.float64)
    names = model_joint_names(ROBOT, G1_XML.name)
    if qpos.shape[1] != ROOT_WIDTH + len(names):
        raise ValueError(f"qpos has {qpos.shape[1]} columns, expected {ROOT_WIDTH + len(names)}")
    motion = RobotMotion(qpos[:, 0:3], qpos[:, 3:7], qpos[:, ROOT_WIDTH:], names, fps=fps)
    # the generators run at the tracker rate, so target_fps=fps is a no-op resample
    return write_reference(motion, robot=ROBOT, out_dir=args.out_dir, name=args.name,
                           target_fps=fps, tag=tag)
