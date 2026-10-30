"""Render a tracking run as an mp4 (headless MuJoCo, EGL by default).

    python -m dtmr.tools.render_rollout --run output/demo_g1_jump --robot unitree_g1 \\
        --out output/demo_g1_jump/rollout.mp4 [--ref data/handcrafted/hc_jump_50Hz.npz]
    python -m dtmr.tools.render_rollout --robot unitree_g1 \\
        --ref data/handcrafted/hc_jump_50Hz.npz --out reference.mp4     # the reference alone

Reads the newest `<stamp>_log.npz` in `--run` (one qpos row per MPC step, 50 Hz
by default) and renders the simulated robot into a clean scene: gradient sky,
checker floor, a sun with shadows, and a camera that follows the root when the
clip travels.  With `--ref`, the reference npz an importer wrote is drawn as a
translucent twin at the *phase-aligned*
frame (the run log stores the tracker's phase per step), so the video shows
what the time warping did.  Without `--run`, the `--ref` clip itself is rendered:
a quick look at a converted reference before spending GPU time on it.  Any robot with a descriptor works; the floor and
lights of the robot's own scene are dropped in favour of this one's.

Needs ffmpeg on PATH.  Set MUJOCO_GL=osmesa (or glfw with a display) if EGL is
unavailable.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from dtmr.robots import load_robot
from dtmr.utils import get_model_path
from dtmr.utils import load_reference_state

ROOT_QPOS = 7
MIN_TRAVEL_FOR_FOLLOW_M = 1.5
TAIL_HOLD_S = 0.5
GHOST_RGBA = (0.25, 0.55, 0.95, 0.35)

WORLD_XML = """
<mujoco model="render_rollout">
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="38"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <map shadowclip="3" shadowscale="0.8"/>
    <rgba haze="0.9 0.92 0.95 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.86 0.89 0.94" rgb2="0.98 0.98 0.99" width="64" height="64"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.76 0.77 0.80" rgb2="0.86 0.87 0.89"
             width="512" height="512" mark="edge" markrgb="0.66 0.66 0.68"/>
    <material name="grid" texture="grid" texrepeat="4 4" texuniform="true" reflectance="0" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="1 -5 8" dir="-0.15 0.55 -0.82" diffuse="0.5 0.5 0.5"
           specular="0.1 0.1 0.1" castshadow="true"/>
    <light name="fill" directional="true" pos="-6 -4 5" dir="0.55 0.4 -0.6" diffuse="0.25 0.25 0.27" castshadow="false"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="grid"/>
  </worldbody>
</mujoco>"""


def _robot_spec(scene_path: Path) -> mujoco.MjSpec:
    """The robot's scene with its own floor and lights removed."""
    spec = mujoco.MjSpec.from_file(str(scene_path))
    for geom in list(spec.worldbody.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(geom)
    for light in list(spec.lights):
        spec.delete(light)
    return spec


def build_model(scene_path: Path, width: int, height: int, with_ghost: bool) -> mujoco.MjModel:
    world = mujoco.MjSpec.from_string(WORLD_XML.format(W=width, H=height))
    world.attach(_robot_spec(scene_path), prefix="robot/", frame=world.worldbody.add_frame())
    if with_ghost:
        ghost = _robot_spec(scene_path)
        for geom in ghost.geoms:
            geom.rgba = GHOST_RGBA
            geom.contype = geom.conaffinity = 0
        for mat in ghost.materials:          # mesh colours come from materials, override those too
            mat.rgba = GHOST_RGBA
        world.attach(ghost, prefix="ref/", frame=world.worldbody.add_frame())
    return world.compile()


def qpos_index(model: mujoco.MjModel, prefix: str) -> np.ndarray:
    idx: list[int] = []
    # Match on the joint's body: attach() prefixes every body, but an unnamed
    # free joint (T1, K1, AgiBot) stays unnamed.
    for j in range(model.njnt):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])) or ""
        if body.startswith(prefix):
            n = ROOT_QPOS if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
            idx.extend(range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j]) + n))
    return np.array(idx)


def smooth(xy: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return xy
    pad = window // 2
    padded = np.concatenate([np.repeat(xy[:1], pad, 0), xy, np.repeat(xy[-1:], pad, 0)])
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid")[: len(xy)] for i in range(xy.shape[1])], 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="tracker output directory (newest *_log.npz is used); omit to render --ref alone")
    parser.add_argument("--robot", required=True, help="robot descriptor name, e.g. unitree_g1, booster_t1")
    parser.add_argument("--out", required=True, help="mp4 path")
    parser.add_argument("--ref", default=None, help="the reference npz (its qpos; an older state CSV also works), drawn as a ghost")
    parser.add_argument("--fps", type=int, default=50, help="MPC step rate of the log")
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--azimuth", type=float, default=140.0)
    parser.add_argument("--elevation", type=float, default=-14.0)
    parser.add_argument("--distance", type=float, default=0.0, help="0 = from the robot's height and travel")
    parser.add_argument("--smooth", type=float, default=0.6, help="follow-camera smoothing window (s)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed; 0.5 = half speed")
    parser.add_argument("--stride", type=int, default=1, help="render every n-th step (faster previews)")
    args = parser.parse_args()

    if args.run is None and args.ref is None:
        sys.exit("give --run (a tracker output directory), --ref (a reference npz), or both")
    log = None
    if args.run is not None:
        logs = sorted(glob.glob(os.path.join(args.run, "*_log.npz")))
        if not logs:
            sys.exit(f"no *_log.npz in {args.run}")
        log = np.load(logs[-1], allow_pickle=True)
    robot = load_robot(args.robot)
    if robot is None:
        sys.exit(f"unknown robot {args.robot!r} (no src/dtmr/robots/{args.robot}.yaml)")
    scene = get_model_path(robot["model_dir"], robot["model_file"])
    width, height = (int(v) for v in args.size.split("x"))

    model = build_model(scene, width, height, with_ghost=log is not None and args.ref is not None)
    data = mujoco.MjData(model)
    robot_idx = qpos_index(model, "robot/")
    if log is None:                      # the reference alone, drawn solid
        qpos = np.asarray(load_reference_state(args.ref)["qpos"], dtype=np.float64)[:, : len(robot_idx)]
    else:
        qpos = np.asarray(log["qpos"], dtype=np.float64)
    if qpos.shape[1] != len(robot_idx):
        sys.exit(f"{'log' if log is not None else 'reference'} qpos has {qpos.shape[1]} columns "
                 f"but the {args.robot} model has {len(robot_idx)}")

    ghost_idx = ref_qpos = None
    if log is not None and args.ref is not None:
        ghost_idx = qpos_index(model, "ref/")
        ref_qpos = np.asarray(load_reference_state(args.ref)["qpos"], dtype=np.float64)[:, : len(ghost_idx)]
        phases = np.clip(np.asarray(log["phases"], dtype=np.float64), 0.0, 1.0)
        ref_frames = np.minimum((phases * (len(ref_qpos) - 1)).round().astype(int), len(ref_qpos) - 1)

    steps = np.arange(0, len(qpos), args.stride)
    root_xy = qpos[:, :2]
    travel = float(np.ptp(root_xy, axis=0).max())
    follow = travel > MIN_TRAVEL_FOR_FOLLOW_M
    root_z = float(np.median(qpos[:, 2]))
    lookat_xy = smooth(root_xy, int(round(args.smooth * args.fps)) | 1) if follow else np.repeat(root_xy.mean(0, keepdims=True), len(qpos), 0)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = args.azimuth, args.elevation
    camera.distance = args.distance or np.clip(4.0 * root_z, 1.6, 3.4) + (0.0 if follow else 0.4 * travel)
    renderer = mujoco.Renderer(model, height=height, width=width)
    out_fps = max(1, int(round(args.fps * args.speed / args.stride)))
    print(f"[render] {len(qpos)} steps ({len(qpos) / args.fps:.2f} s), travel {travel:.2f} m, "
          f"{'follow' if follow else 'fixed'} camera at {camera.distance:.2f} m -> {args.out} @ {out_fps} fps")

    def frame(t: int) -> np.ndarray:
        data.qpos[robot_idx] = qpos[t]
        if ghost_idx is not None:
            data.qpos[ghost_idx] = ref_qpos[ref_frames[t]]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = [lookat_xy[t, 0], lookat_xy[t, 1], root_z * 1.1]
        renderer.update_scene(data, camera=camera)
        return renderer.render()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(out_fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    assert ffmpeg.stdin is not None
    for t in steps:
        ffmpeg.stdin.write(frame(int(t)).tobytes())
    last = frame(int(steps[-1])).tobytes()
    for _ in range(int(TAIL_HOLD_S * out_fps)):
        ffmpeg.stdin.write(last)
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"[render] wrote {args.out}")


if __name__ == "__main__":
    main()
