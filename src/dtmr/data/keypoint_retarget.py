"""Kinematic retargeting onto any robot in `src/dtmr/robots/` through body keypoints.

A 100STYLE take (`bvh_100style.py`) is reduced to keypoints: pelvis, torso, and per
side hip, knee, ankle, shoulder, elbow and hand positions, plus pelvis, torso and
foot orientations and which feet are on the floor.  Then:

  * scale: positions by the leg-length ratio s (hip to ankle), each foot moved
    sideways with its hip joint by the difference in hip width, plus a constant height
    shift so a flat foot sits on the floor at the target's own ankle height
  * pose: per frame, damped least-squares IK on all dofs: foot pose and
    hip-centre position from the scaled source, pelvis/torso/foot orientation,
    and knee/elbow/hand points rebuilt from the source's bone directions with
    the target's own bone lengths; a weak pull toward the home keyframe fixes
    whatever the keypoints leave free (head, wrists)
  * contact: wherever a source foot is on the floor, the target foot's floor
    collision geometry is put back on z = 0 and the frame re-solved

    python -m dtmr.data.keypoint_retarget data/100style/100STYLE/Zombie/Zombie_FW.bvh \
        --robot unitree_g1 --out-dir data/100style_ref/unitree_g1 --start 0 --end 10
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from dtmr.robots import load_robot
from dtmr.data.robot_motion import RobotMotion, model_joint_names, write_reference
from dtmr.utils import get_model_path

SIDES = ("left", "right")

# Keypoint bodies per robot. "{s}"/"{S}"/"{X}" expand to left/Left/L. A tuple
# adds an offset (zero-pose world frame, y mirrored for the right side) for
# rigs whose body origin is not at the joint the keypoint stands for.
KEYPOINTS = {
    "unitree_g1": dict(pelvis="pelvis", torso="torso_link", hip="{s}_hip_roll_link",
                       knee="{s}_knee_link", ankle="{s}_ankle_roll_link",
                       shoulder="{s}_shoulder_roll_link", elbow="{s}_elbow_link",
                       hand="{s}_wrist_yaw_link"),
    "unitree_h1_2": dict(pelvis="pelvis", torso="torso_link", hip="{s}_hip_roll_link",
                         knee="{s}_knee_link", ankle="{s}_ankle_roll_link",
                         shoulder="{s}_shoulder_roll_link", elbow="{s}_elbow_link",
                         hand="{s}_wrist_yaw_link"),
    "unitree_r1": dict(pelvis="pelvis", torso="torso_link", hip="{s}_hip_roll_link",
                       knee="{s}_knee_link", ankle="{s}_ankle_roll_link",
                       shoulder="{s}_shoulder_roll_link", elbow="{s}_elbow_link",
                       hand="{s}_wrist_roll_link"),
    # T1/K1 arms are shoulder pitch/roll, upper-arm twist ("elbow pitch") and the
    # elbow bend ("elbow yaw"): the elbow is the last arm body, the hand its far end
    "booster_t1": dict(pelvis="Waist", torso="Trunk", hip="Hip_Roll_{S}", knee="Shank_{S}",
                       ankle="{s}_foot_link", shoulder="A{X}2", elbow="{s}_hand_link",
                       hand=("{s}_hand_link", (0.0, 0.20, 0.0))),
    "booster_k1": dict(pelvis="trunk", torso="trunk", hip="{s}_hip_roll_link",
                       knee="{s}_knee_pitch_link", ankle="{s}_ankle_roll_link",
                       shoulder="{s}_shoulder_roll_link", elbow="{s}_elbow_yaw_link",
                       hand=("{s}_elbow_yaw_link", (0.0, 0.19, 0.0))),
    "agibot_a3": dict(pelvis="pelvis_link", torso="torso_Link", hip="{s}_hip_roll_Link",
                      knee="{s}_knee_Link", ankle="{s}_ankle_roll_Link",
                      shoulder="{s}_shoulder_roll_Link", elbow="{s}_elbow_Link",
                      hand="{s}_wrist_yaw_Link"),
    "agibot_x2_ultra": dict(pelvis="pelvis", torso="torso_link", hip="{s}_hip_roll_link",
                            knee="{s}_knee_link", ankle="{s}_ankle_roll_link",
                            shoulder="{s}_shoulder_roll_link", elbow="{s}_elbow_link",
                            hand="{s}_wrist_roll_link"),
}
LIMB = ("hip", "knee", "ankle", "shoulder", "elbow", "hand")

# task weights (residuals in m and rad)
W_FOOT_POS, W_FOOT_ROT = 1.0, 0.5
W_FOOT_YAW = 0.1           # foot yaw gives way first (H1-2 hip yaw turns only +-25 deg)
W_STANCE_POS = 3.0         # a planted foot outweighs the hip-centre target
W_HIP, W_PELVIS_ROT, W_TORSO_ROT = 1.0, 0.5, 0.3
W_KNEE, W_ARM = 0.3, 0.3
POSTURE, DAMPING = 1e-5, 1e-4
RETRY_M = 0.05             # foot error that triggers a cold re-solve of the frame


def _expand(spec, side):
    body, off = (spec, (0.0, 0.0, 0.0)) if isinstance(spec, str) else spec
    sign = 1.0 if side == "left" else -1.0
    name = body.format(s=side, S=side.capitalize(), X=side[0].upper())
    return name, np.array([off[0], sign * off[1], off[2]])


def _rotvec(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, R.ravel())
    v = np.empty(3); mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class Rig:
    """A robot model with its keypoints resolved to (body id, body-frame offset)."""

    def __init__(self, robot: str):
        desc = load_robot(robot)
        if desc is None or robot not in KEYPOINTS:
            raise SystemExit(f"[retarget] no keypoint table for robot {robot!r}; "
                             f"known: {', '.join(KEYPOINTS)}")
        self.robot = robot
        self.m = mujoco.MjModel.from_xml_path(str(get_model_path(desc["model_dir"], desc["model_file"])))
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.q_home = m.key("home").qpos.copy()
        self.zero_pose()
        kp = KEYPOINTS[robot]
        self.body, self.off = {}, {}
        for key in ("pelvis", "torso"):
            self.body[key] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, kp[key])
            self.off[key] = np.zeros(3)
        for key in LIMB:
            for side in SIDES:
                name, off_world = _expand(kp[key], side)
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                if b < 0:
                    raise ValueError(f"{robot}: no body {name!r}")
                self.body[key, side] = b
                self.off[key, side] = d.xmat[b].reshape(3, 3).T @ off_world
        # zero-pose geometry: rest orientations, bone lengths, leg length, sole height
        self.R0 = {k: d.xmat[self.body[k]].reshape(3, 3).copy()
                   for k in ("pelvis", "torso", ("ankle", "left"), ("ankle", "right"))}
        p = {k: self.point(k) for k in self.body}
        L = lambda a, b: float(np.linalg.norm(p[a, "left"] - p[b, "left"]))
        self.bone = dict(thigh=L("hip", "knee"), upper=L("shoulder", "elbow"), fore=L("elbow", "hand"))
        self.leg = float(p["hip", "left"][2] - p["ankle", "left"][2])
        self.hip_width = float(p["hip", "left"][1] - p["hip", "right"][1])
        self.sole_geoms = {side: self._floor_geoms(self.body["ankle", side]) for side in SIDES}
        self.ankle_flat = float(p["ankle", "left"][2] - self.sole_z("left"))
        jnt = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.hinge_qadr = np.array([m.jnt_qposadr[j] for j in jnt])
        self.hinge_vadr = np.array([m.jnt_dofadr[j] for j in jnt])
        lim = np.array([m.jnt_limited[j] for j in jnt], dtype=bool)
        rng = np.array([m.jnt_range[j] for j in jnt])
        self.lo = np.where(lim, rng[:, 0], -np.inf)
        self.hi = np.where(lim, rng[:, 1], np.inf)

    def zero_pose(self, height: float = 1.0) -> None:
        self.d.qpos[:] = 0.0
        self.d.qpos[2] = height
        self.d.qpos[3] = 1.0
        mujoco.mj_kinematics(self.m, self.d)

    def point(self, key) -> np.ndarray:
        b = self.body[key]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.off[key]

    def _floor_geoms(self, foot: int) -> list[int]:
        """Geoms of the foot (and bodies rigidly below it) that collide with the floor."""
        m = self.m
        bodies = {foot} | {b for b in range(m.nbody) if m.body_parentid[b] == foot}
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]
        paired = {int(g) for p in range(m.npair) for g in (m.pair_geom1[p], m.pair_geom2[p])}
        collide = [g for g in geoms if g in paired or (m.geom_contype[g] | m.geom_conaffinity[g])]
        return collide or geoms

    def sole_z(self, side: str) -> float:
        """Lowest point of the foot's floor geometry at the current kinematics."""
        m, d = self.m, self.d
        low = np.inf
        for g in self.sole_geoms[side]:
            R, c, size = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                z = c[2] - size[0]
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                z = c[2] - abs(R[2, 2]) * size[1] - size[0]
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                z = c[2] - np.abs(R[2]) @ size
            elif t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                z = (c + v @ R.T)[:, 2].min()
            else:
                continue
            low = min(low, float(z))
        return low


@dataclass
class SourceMotion:
    """Keypoints of a source motion, everything in metres, z up, at `fps`.

    kp[key] / kp[key, side]: (T, 3) positions; kp["R", key]: (T, 3, 3) rotation
    of the pelvis, torso or ("ankle", side) frame relative to its upright rest
    pose; kp["contact"]: (T, 2) feet on the floor.
    """
    kp: dict
    leg: float          # hip to ankle, leg straight
    hip_width: float
    ankle_flat: float   # ankle height when the foot is flat on the floor
    fps: float


def _frame(x, y_hint):
    """Rotation with first axis x and second axis as close to y_hint as possible."""
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, y_hint)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], axis=-1)


def human_source(h) -> SourceMotion:
    """A `HumanMotion` as keypoints, heading turned to +x.

    Pelvis and torso frames come from the hip and shoulder joints and the spine,
    the foot frame from the foot bone, calibrated to be flat on a flat stance frame.
    """
    pos = lambda b, which="start": getattr(h, which)[b]
    side_bone = {"hip": "femur", "knee": "tibia", "ankle": "foot", "ball": "toes",
                 "shoulder": "humerus", "elbow": "radius", "hand": "wrist"}
    kp = {(k, side): pos(side[0] + b) for k, b in side_bone.items() for side in SIDES}
    kp["pelvis"] = pos("root")
    kp["torso"] = pos("lowerneck")
    up = np.array([0.0, 0.0, 1.0])
    # body frames as (x forward, y left, z up): build from lateral + up axes
    def body_frame(lat, upv):
        y = lat / np.linalg.norm(lat, axis=-1, keepdims=True)
        x = np.cross(y, upv)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return np.stack([x, y, np.cross(x, y)], axis=-1)
    R_pelvis = body_frame(kp["hip", "left"] - kp["hip", "right"], pos("lowerback", "end") - kp["pelvis"])
    R_torso = body_frame(kp["shoulder", "left"] - kp["shoulder", "right"], kp["torso"] - pos("upperback", "end"))
    # heading: the first frame's pelvis faces +x, root over the origin
    yaw0 = np.arctan2(R_pelvis[0, 1, 0], R_pelvis[0, 0, 0])
    c, s_ = np.cos(-yaw0), np.sin(-yaw0)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    shift = np.array([kp["pelvis"][0, 0], kp["pelvis"][0, 1], 0.0])
    for k in list(kp):
        kp[k] = (kp[k] - shift) @ Rz.T
    kp["R", "pelvis"] = Rz @ R_pelvis
    kp["R", "torso"] = Rz @ R_torso
    # floor and contact from the lower of ankle and ball
    low = np.minimum(np.stack([kp["ankle", s][:, 2] for s in SIDES], 1),
                     np.stack([kp["ball", s][:, 2] for s in SIDES], 1))
    floor = np.percentile(low, 2)
    for k in list(kp):
        if not (isinstance(k, tuple) and k[0] == "R"):
            kp[k] = kp[k] - [0.0, 0.0, floor]
    low -= floor
    fps = h.fps
    speed = np.stack([np.linalg.norm(np.gradient(kp["ankle", s][:, :2], axis=0), axis=1) * fps
                      for s in SIDES], 1)
    kp["contact"] = (low < 0.04) & (speed < 0.6)
    ankle_flat = float(np.median(np.concatenate([kp["ankle", s][kp["contact"][:, f], 2]
                                                 for f, s in enumerate(SIDES)])))
    for f, side in enumerate(SIDES):
        # The ankle-to-ball line gives a flat-footed frame but no yaw when the foot points
        # down (on the toes); the foot bone's rotation is always defined. Calibrate the bone
        # frame once, on the flattest stance frame, against the line-based frame.
        fwd = kp["ball", side] - kp["ankle", side]
        fwd_flat = fwd.copy(); fwd_flat[:, 2] = 0.0
        pitch = np.arctan2(-fwd[:, 2], np.linalg.norm(fwd[:, :2], axis=1))
        stance = np.where(kp["contact"][:, f])[0]
        p0 = float(np.median(pitch[stance])) if len(stance) else 0.0
        t0 = stance[np.argmin(np.abs(pitch[stance] - p0))] if len(stance) else 0
        c0, s0 = np.cos(-p0), np.sin(-p0)
        flat = _frame(fwd[t0], np.cross(up, fwd_flat[t0])) @ np.array([[c0, 0, s0], [0, 1, 0], [-s0, 0, c0]])
        R_bone = Rz @ h.rot[side[0] + "foot"]
        kp["R", ("ankle", side)] = R_bone @ (R_bone[t0].T @ flat)
        del kp["ball", side]
    leg = float(np.linalg.norm(h.end["lfemur"][0] - h.start["lfemur"][0])
                + np.linalg.norm(h.end["ltibia"][0] - h.start["ltibia"][0]))
    hip_width = float(np.linalg.norm(kp["hip", "left"][0] - kp["hip", "right"][0]))
    return SourceMotion(kp, leg, hip_width, ankle_flat, fps)


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def _limited_step(H, g, q_hinge, rig, passes: int = 4) -> np.ndarray:
    """Newton step H dq = g with hinge joints held at the limits they would cross.

    Clipping after an unconstrained step stalls the solve when a joint is pinned
    (H1-2's hip yaw turns only +-25 deg); freezing it and re-solving lets the other
    joints take up the motion."""
    nv = len(g)
    free = np.ones(nv, dtype=bool)
    dq = np.zeros(nv)
    for _ in range(passes):
        rhs = g[free] - H[np.ix_(free, ~free)] @ dq[~free]
        dq[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
        q_new = q_hinge + dq[rig.hinge_vadr]
        over = (q_new < rig.lo) | (q_new > rig.hi)
        over &= free[rig.hinge_vadr]
        if not over.any():
            break
        v = rig.hinge_vadr[over]
        dq[v] = np.clip(q_new[over], rig.lo[over], rig.hi[over]) - q_hinge[over]
        free[v] = False
    return dq


def solve_frame(rig: Rig, kp: dict, i: int, q: np.ndarray, scale, z_shift,
                foot_lift=np.zeros(2), iters: int = 20) -> np.ndarray:
    """IK for frame i, warm-started from q (edited in place). Returns the two foot errors (m)."""
    m, d = rig.m, rig.d
    s_len, half_width = scale
    S = lambda p: s_len * p + [0.0, 0.0, z_shift]
    rot = lambda k: kp["R", k][i] @ rig.R0[k]   # the source's change from rest, applied to ours
    hip_mid = 0.5 * (kp["hip", "left"][i] + kp["hip", "right"][i])
    # the target's hip joints sit further out (or in) than the scaled source's: move
    # each foot with its hip, along the pelvis lateral axis
    lat = kp["R", "pelvis"][i][:, 1].copy()
    lat[2] = 0.0
    lat /= max(np.linalg.norm(lat), 1e-9)
    tgt_foot = [S(kp["ankle", s][i]) + sign * half_width * lat + [0, 0, foot_lift[f]]
                for f, (s, sign) in enumerate(zip(SIDES, (1.0, -1.0)))]
    w_foot = np.where(kp["contact"][i], W_STANCE_POS, W_FOOT_POS)
    tgt_hip = S(hip_mid)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    d.qpos[:] = q
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rows, errs = [], []

        def pos_task(key, target, w):
            p = rig.point(key)
            mujoco.mj_jac(m, d, jacp, jacr, p, rig.body[key])
            rows.append(w * jacp.copy()); errs.append(w * (target - p))

        def rot_task(key, R_tgt, w, w_yaw=None):
            b = rig.body[key]
            mujoco.mj_jacBody(m, d, jacp, jacr, b)
            W = np.diag([w, w, w if w_yaw is None else w_yaw])   # world x/y tilt, z yaw
            rows.append(W @ jacr)
            errs.append(W @ _rotvec(R_tgt @ d.xmat[b].reshape(3, 3).T))

        for f, side in enumerate(SIDES):
            pos_task(("ankle", side), tgt_foot[f], w_foot[f])
            rot_task(("ankle", side), rot(("ankle", side)), W_FOOT_ROT, W_FOOT_YAW)
        # hip centre: average of the two hip-point tasks
        hl, hr = rig.point(("hip", "left")), rig.point(("hip", "right"))
        mujoco.mj_jac(m, d, jacp, jacr, hl, rig.body["hip", "left"]); J = jacp.copy()
        mujoco.mj_jac(m, d, jacp, jacr, hr, rig.body["hip", "right"]); J = 0.5 * (J + jacp)
        rows.append(W_HIP * J); errs.append(W_HIP * (tgt_hip - 0.5 * (hl + hr)))
        rot_task("pelvis", rot("pelvis"), W_PELVIS_ROT)
        rot_task("torso", rot("torso"), W_TORSO_ROT)
        for side in SIDES:
            g = lambda k: kp[k, side][i]
            pos_task(("knee", side), rig.point(("hip", side))
                     + rig.bone["thigh"] * _unit(g("knee") - g("hip")), W_KNEE)
            elbow = rig.point(("shoulder", side)) + rig.bone["upper"] * _unit(g("elbow") - g("shoulder"))
            pos_task(("elbow", side), elbow, W_ARM)
            pos_task(("hand", side), elbow + rig.bone["fore"] * _unit(g("hand") - g("elbow")), W_ARM)
        J = np.vstack(rows); e = np.concatenate(errs)
        H = J.T @ J + DAMPING * np.eye(m.nv)
        g_vec = J.T @ e
        H[rig.hinge_vadr, rig.hinge_vadr] += POSTURE
        g_vec[rig.hinge_vadr] += POSTURE * (rig.q_home[rig.hinge_qadr] - d.qpos[rig.hinge_qadr])
        dq = _limited_step(H, g_vec, d.qpos[rig.hinge_qadr], rig)
        dq *= min(1.0, 0.5 / max(np.abs(dq).max(), 1e-12))
        mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
        d.qpos[rig.hinge_qadr] = np.clip(d.qpos[rig.hinge_qadr], rig.lo, rig.hi)
        if np.linalg.norm(dq) < 1e-6:
            break
    mujoco.mj_kinematics(m, d)
    q[:] = d.qpos
    return np.array([np.linalg.norm(tgt_foot[f] - rig.point(("ankle", s))) for f, s in enumerate(SIDES)])


def retarget(src: SourceMotion, robot: str, *, contact_rounds: int = 3) -> tuple[np.ndarray, float, dict]:
    """Source keypoints -> target qpos (T, nq) and its frame rate."""
    rig = Rig(robot)
    kp = src.kp
    s = rig.leg / src.leg
    half_width = 0.5 * (rig.hip_width - s * src.hip_width)
    z_shift = rig.ankle_flat - s * src.ankle_flat
    T = len(kp["pelvis"])
    q = rig.q_home.copy()
    q[0:2] = s * kp["pelvis"][0, :2]
    qpos = np.zeros((T, rig.m.nq))
    err = np.zeros((T, 2))
    root_quat = np.zeros(4)
    for i in range(T):
        # seed the floating base with the source pelvis orientation so a fast spin
        # (a flip, a cartwheel) cannot leave the solve on a twisted branch
        mujoco.mju_mat2Quat(root_quat, kp["R", "pelvis"][i].ravel())
        q[3:7] = root_quat
        err[i] = solve_frame(rig, kp, i, q, (s, half_width), z_shift, iters=60 if i == 0 else 20)
        if err[i].max() > RETRY_M:            # warm start lost: solve again from the home pose
            q2 = rig.q_home.copy()
            q2[0:3], q2[3:7] = q[0:3], root_quat
            e2 = solve_frame(rig, kp, i, q2, (s, half_width), z_shift, iters=60)
            if e2.max() < err[i].max():
                q[:], err[i] = q2, e2
        qpos[i] = q
    # contact: a foot the source has on the floor goes back onto the floor here
    lift = np.zeros((T, 2))
    for _ in range(contact_rounds):
        fixed = 0
        for i in range(T):
            rig.d.qpos[:] = qpos[i]
            mujoco.mj_kinematics(rig.m, rig.d)
            z = np.array([rig.sole_z(side) for side in SIDES])
            dz = np.where(kp["contact"][i], -z, np.maximum(-z, 0.0))
            if np.all(np.abs(dz) < 5e-4):
                continue
            lift[i] += dz
            q[:] = qpos[i]
            solve_frame(rig, kp, i, q, (s, half_width), z_shift, foot_lift=lift[i])
            qpos[i] = q
            fixed += 1
        if not fixed:
            break
    # a swing foot misses its target where the source tucks deeper than this robot's knee allows
    stance = kp["contact"]
    info = dict(scale=s, hip_shift_mm=1000 * half_width,
                stance_foot_mm=1000 * float(err[stance].max(initial=0.0)),
                swing_foot_mm=1000 * float(err[~stance].max(initial=0.0)))
    return qpos, src.fps, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="a 100STYLE .bvh")
    ap.add_argument("--robot", required=True, choices=sorted(KEYPOINTS))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", default=None, help="output basename (default: the source's stem)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the source")
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    if args.source.suffix != ".bvh":
        raise SystemExit(f"[retarget] {args.source}: expected a 100STYLE .bvh")
    from dtmr.data.bvh_100style import load_style
    human = load_style(args.source)
    cut = slice(int(args.start * human.fps), None if args.end is None else int(args.end * human.fps))
    for part in (human.start, human.end, human.rot):
        for b in part:
            part[b] = part[b][cut]
    src = human_source(human)
    qpos, fps, info = retarget(src, args.robot)
    print(f"[retarget] 100style -> {args.robot}: length x{info['scale']:.3f}, feet "
          f"{info['hip_shift_mm']:+.0f} mm outward; worst foot error "
          f"{info['stance_foot_mm']:.1f} mm in stance, {info['swing_foot_mm']:.1f} mm in swing")
    name = args.name or args.source.stem.removesuffix("_50Hz")
    motion = RobotMotion(qpos[:, :3], qpos[:, 3:7], qpos[:, 7:], model_joint_names(args.robot), fps=fps)
    npz = write_reference(motion, robot=args.robot, out_dir=args.out_dir, name=name, tag="retarget")
    print(f"[retarget] wrote {npz}")


if __name__ == "__main__":
    main()
