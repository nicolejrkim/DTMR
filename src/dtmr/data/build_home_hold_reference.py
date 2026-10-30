"""build_home_hold_reference.py

Build a "maintain a static pose" tracking reference for any robot.

FK a model keyframe (default ``home``) once, then hold every tracked-link world
pose constant for ``--frames`` frames. Writes the reference npz
(link poses plus the held qpos / zero qvel) that the tracking environment reads.

Tracking a constant home reference with the standard tracking cost (COM +
COM-relative link positions + link orientations) is equivalent to a stay-upright
/ hold-pose task, with no custom cost kernel — this is the baseline used by
``src/dtmr/examples/unitree_g1_home_hold_warp.yaml``.

Usage:
    python -m dtmr.data.build_home_hold_reference \
        --robot unitree_g1 --keyframe home --fps 50 --frames 1000 \
        --out-dir data/home_hold --stem home_hold_50Hz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

def build_home_hold_reference(
    model_path: str,
    keyframe: str = "home",
    fps: int = 50,
    frames: int = 1000,
    out_dir: str = "data/home_hold",
    stem: str = "home_hold_50Hz",
    link_names: list[str] | None = None,
) -> str:
    """FK the keyframe and write the held-constant reference npz (link poses + state).

    Returns the npz path.
    """
    m = mujoco.MjModel.from_xml_path(model_path)
    if link_names is None:      # every body of the robot, root first (the world body is skipped)
        link_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)]
    link_names = list(link_names)
    nq, nv = m.nq, m.nv
    home_qpos = np.array(m.keyframe(keyframe).qpos, dtype=np.float64)

    # FK the pose once.
    d = mujoco.MjData(m)
    d.qpos[:] = home_qpos
    mujoco.mj_forward(m, d)

    link_idx = np.array(
        [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in link_names],
        dtype=np.int64,
    )
    if not (link_idx > 0).all():
        missing = [n for n, i in zip(link_names, link_idx) if i <= 0]
        raise ValueError(f"unknown link name(s) for model {model_path}: {missing}")

    # World pos (xpos) + quat (xquat, wxyz) per tracked link, held constant.
    pos0 = np.array([d.xpos[b] for b in link_idx], dtype=np.float32)   # (L, 3)
    rot0 = np.array([d.xquat[b] for b in link_idx], dtype=np.float32)  # (L, 4) wxyz
    pos = np.broadcast_to(pos0[None], (frames, len(link_names), 3)).astype(np.float32).copy()
    rot = np.broadcast_to(rot0[None], (frames, len(link_names), 4)).astype(np.float32).copy()
    timestamps = np.arange(frames, dtype=np.float64) / float(fps)

    # State: keyframe qpos held, qvel = 0.
    qpos = np.broadcast_to(home_qpos.astype(np.float32)[None], (frames, nq)).copy()
    qvel = np.zeros((frames, nv), dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez_compressed(
        npz_path,
        qpos=qpos, qvel=qvel, fps=np.array(float(fps)),
        pos=pos, rot=rot,
        link_names=np.array(link_names, dtype="<U25"),
        link_idx=link_idx,
        timestamps=timestamps,
        include_base=np.array(True),
    )
    print(f"model {model_path}  nq={nq} nv={nv}  keyframe={keyframe!r} base z={home_qpos[2]:.4f}")
    print(f"tracked links: {len(link_names)}  frames: {frames} @ {fps} Hz  pelvis xpos={pos0[0]}")
    print(f"[write] {npz_path}  qpos{qpos.shape} pos{pos.shape} rot{rot.shape}; set motion_path: {npz_path}")
    return npz_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="unitree_g1",
                   help="Robot descriptor name (src/dtmr/robots/); its tracking scene is used.")
    p.add_argument("--model", default=None,
                   help="MJCF path to use instead of the robot's tracking scene (must contain the keyframe).")
    p.add_argument("--keyframe", default="home", help="Keyframe name to FK and hold.")
    p.add_argument("--fps", type=int, default=50, help="Reference frame rate.")
    p.add_argument("--frames", type=int, default=1000,
                   help="Number of (identical) frames; keep long so phi never reaches 1.0.")
    p.add_argument("--out-dir", default="data/home_hold", help="Output directory.")
    p.add_argument("--stem", default="home_hold_50Hz", help="Output basename (writes <stem>.npz).")
    args = p.parse_args()
    if args.model is None:
        from dtmr.robots import load_robot
        from dtmr.utils import get_model_path

        desc = load_robot(args.robot)
        if desc is None:
            p.error(f"unknown robot {args.robot!r}")
        args.model = str(get_model_path(desc["model_dir"], desc["model_file"]))
    build_home_hold_reference(
        model_path=args.model, keyframe=args.keyframe, fps=args.fps,
        frames=args.frames, out_dir=args.out_dir, stem=args.stem,
    )


if __name__ == "__main__":
    main()
