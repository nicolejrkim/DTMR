"""Handcraft a G1 triple forward jump (three consecutive standing broad jumps).

Same construction as sprint_g1: task-space plan (world-pinned feet on
ground phases, exact ballistic root during flight) + damped-LS leg IK +
joint-space arm countermovement swing.  Per jump:

  crouch (arms back, butt-back squat) -> push (toe-pivot heel lift, root
  launches with matched takeoff velocity) -> flight (root exactly ballistic,
  feet swing to the next foothold) -> land (forefoot, absorb) -> recover

Output: the tracker's reference files (state CSV + link_world_poses npz).

  python -m dtmr.data.handcrafted.jump_g1 \
      --out-dir data/handcrafted [--jump-dist 0.8] [--n-jumps 3]
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from dtmr.data.handcrafted import G1_XML, add_output_args, save_reference
from dtmr.data.handcrafted.sprint_g1 import (LEG_JOINTS, ANKLE_Z_FLAT, TOE, smoothstep,
                            toe_pivot, yaw_pitch_quat, solve_legs)

G = 9.81


def hermite(s, v0, v1, m0, m1, T):
    """Cubic Hermite on s in [0,1] with endpoint velocities m (units/sec)."""
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * v0 + (s3 - 2 * s2 + s) * m0 * T
            + (-2 * s3 + 3 * s2) * v1 + (s3 - s2) * m1 * T)


def main():
    ap = argparse.ArgumentParser()
    add_output_args(ap, default_name="hc_jump")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--n-jumps", type=int, default=3)
    ap.add_argument("--jump-dist", type=float, default=0.8, help="foothold advance per jump m")
    ap.add_argument("--flight-time", type=float, default=0.40)
    ap.add_argument("--crouch", type=float, default=0.45, help="crouch duration s")
    ap.add_argument("--push", type=float, default=0.16)
    ap.add_argument("--land", type=float, default=0.28)
    ap.add_argument("--recover", type=float, default=0.36)
    ap.add_argument("--stand-in", type=float, default=0.8)
    ap.add_argument("--stand-out", type=float, default=1.0)
    ap.add_argument("--z-stand", type=float, default=0.72)
    ap.add_argument("--z-crouch", type=float, default=0.58)
    ap.add_argument("--z-takeoff", type=float, default=0.70)
    ap.add_argument("--z-absorb", type=float, default=0.60)
    ap.add_argument("--crouch-back", type=float, default=0.05, help="butt-back root shift m")
    ap.add_argument("--push-dx", type=float, default=0.12)
    ap.add_argument("--land-dx", type=float, default=0.15)
    ap.add_argument("--foot-y", type=float, default=0.115)
    ap.add_argument("--swing-height", type=float, default=0.20)
    ap.add_argument("--takeoff-pitch", type=float, default=0.50, help="heel lift at toe-off rad")
    ap.add_argument("--land-pitch", type=float, default=0.12)
    par = ap.parse_args()

    fps = par.fps
    Tf = par.flight_time
    vz = G * Tf / 2.0
    flight_dx = par.jump_dist - par.push_dx - par.land_dx
    vx = flight_dx / Tf

    # ---- phase timeline: (kind, duration, payload) -------------------------
    phases = [("stand", par.stand_in)]
    for _ in range(par.n_jumps):
        phases += [("crouch", par.crouch), ("push", par.push),
                   ("flight", Tf), ("land", par.land), ("recover", par.recover)]
    phases += [("stand", par.stand_out)]

    # per-phase arm/posture keyframes: values at END of each phase
    # (shoulder_pitch +=back; elbow qpos 0 = 90deg bent, + straightens)
    KEY = {   # kind: (sh_pitch, elbow, root_pitch, waist_pitch)
        "stand":   (0.10,  0.00, 0.02, 0.02),
        "crouch":  (0.90,  0.25, 0.28, 0.12),
        "push":    (-0.85,  0.35, 0.10, 0.00),
        "flight":  (-0.80,  0.35, 0.05, 0.02),
        "land":    (-0.20,  0.10, 0.18, 0.10),
        "recover": (0.10,  0.00, 0.02, 0.02),
    }

    frames = []          # list of per-frame dicts
    x_c = 0.0            # current foothold center x
    z_cur = par.z_stand  # root z at current phase start
    arm_cur = KEY["stand"]
    for kind, dur in phases:
        n = max(1, int(round(dur * fps)))
        arm_tgt = KEY[kind]
        for k in range(n):
            s = (k + 0.5) / n
            fr = dict(kind=kind, foot_pitch=0.0, pinned=True)
            e = smoothstep(s)
            fr["arm"] = tuple(a + (b - a) * e for a, b in zip(arm_cur, arm_tgt))
            if kind == "stand":
                fr["x"] = x_c; fr["z"] = z_cur
                fr["foot_x"] = x_c
            elif kind == "crouch":
                fr["x"] = x_c - par.crouch_back * e
                fr["z"] = hermite(s, z_cur, par.z_crouch, 0, 0, dur)
                fr["foot_x"] = x_c
            elif kind == "push":
                fr["x"] = hermite(s, x_c - par.crouch_back, x_c + par.push_dx,
                                  0, vx, dur)
                fr["z"] = hermite(s, par.z_crouch, par.z_takeoff, 0, vz, dur)
                fr["foot_x"] = x_c
                fr["foot_pitch"] = par.takeoff_pitch * smoothstep(s) ** 1.3
            elif kind == "flight":
                t = s * Tf
                fr["x"] = x_c + par.push_dx + vx * t
                fr["z"] = par.z_takeoff + vz * t - 0.5 * G * t * t
                fr["pinned"] = False
                fr["swing_s"] = s
                fr["foot_from"] = x_c
                fr["foot_to"] = x_c + par.jump_dist
            elif kind == "land":
                x0 = x_c + par.push_dx + flight_dx
                fr["x"] = hermite(s, x0, x0 + par.land_dx, vx, 0, dur)
                fr["z"] = hermite(s, par.z_takeoff, par.z_absorb, -vz, 0, dur)
                fr["foot_x"] = x_c + par.jump_dist
                fr["foot_pitch"] = par.land_pitch * (1 - smoothstep(s / 0.6))
            elif kind == "recover":
                fr["x"] = x_c + par.jump_dist
                fr["z"] = hermite(s, par.z_absorb, par.z_takeoff, 0, 0, dur)
                fr["foot_x"] = x_c + par.jump_dist
            frames.append(fr)
        arm_cur = arm_tgt
        if kind == "recover":
            x_c += par.jump_dist
        z_cur = dict(stand=z_cur, crouch=par.z_crouch, push=par.z_takeoff,
                     flight=par.z_takeoff, land=par.z_absorb,
                     recover=par.z_takeoff)[kind]

    N = len(frames)
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    d = mujoco.MjData(m)

    # ---- targets -----------------------------------------------------------
    tgt_pos = np.zeros((2, N, 3)); tgt_pitch = np.zeros((2, N))
    stance = np.zeros((2, N), dtype=bool)
    root_pos = np.zeros((N, 3)); root_quat = np.zeros((N, 4))
    for i, fr in enumerate(frames):
        root_pos[i] = [fr["x"], 0.0, fr["z"]]
        root_quat[i] = yaw_pitch_quat(0.0, fr["arm"][2], 0.0)
        for f, sign in enumerate((+1, -1)):
            y = sign * par.foot_y
            if fr["pinned"]:
                flat = np.array([fr["foot_x"], y, ANKLE_Z_FLAT])
                tgt_pos[f, i] = toe_pivot(flat, fr["foot_pitch"])
                tgt_pitch[f, i] = fr["foot_pitch"]
                stance[f, i] = True
            else:
                s = fr["swing_s"]
                lo = toe_pivot(np.array([fr["foot_from"], y, ANKLE_Z_FLAT]),
                               par.takeoff_pitch)
                td = toe_pivot(np.array([fr["foot_to"], y, ANKLE_Z_FLAT]),
                               par.land_pitch)
                p = lo + (td - lo) * smoothstep(s)
                p[2] += par.swing_height * np.sin(np.pi * s) ** 1.1
                tgt_pos[f, i] = p
                tgt_pitch[f, i] = (par.takeoff_pitch * np.exp(-(s / 0.25) ** 2)
                                   - 0.05 * np.exp(-((s - 0.5) / 0.2) ** 2)
                                   + par.land_pitch * np.exp(-((s - 1) / 0.2) ** 2))

    # ---- qpos assembly + IK ------------------------------------------------
    qpos = np.zeros((N, m.nq))
    qpos[:, 0:3] = root_pos; qpos[:, 3:7] = root_quat
    def setj(name, vals):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[:, m.jnt_qposadr[jid]] = vals
    sh = np.array([fr["arm"][0] for fr in frames])
    el = np.array([fr["arm"][1] for fr in frames])
    wp = np.array([fr["arm"][3] for fr in frames])
    for side, roll in (("left", 0.12), ("right", -0.12)):
        setj(f"{side}_shoulder_pitch_joint", sh)
        setj(f"{side}_shoulder_roll_joint", np.full(N, roll))
        setj(f"{side}_elbow_joint", el)
    setj("waist_pitch_joint", wp)

    jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in LEG_JOINTS]
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{sd}_ankle_roll_link")
            for sd in ("left", "right")]
    sole_pts = np.array(
        [[0.04 + sx * 0.09, sy * 0.03, -0.037] for sx in (-1, 1) for sy in (-1, 1)]
        + [[0.075, -0.026, -0.035], [0.075, 0.026, -0.035]])

    worst = 0.0
    q_frame = qpos[0].copy()
    for i in range(N):
        q_frame[:] = qpos[i]
        if i > 0:
            q_frame[jadr] = qpos[i - 1, jadr]
        r = solve_legs(m, d, q_frame, tgt_pos[:, i], tgt_pitch[:, i])
        qpos[i] = q_frame
        worst = max(worst, r)

    # contact-fix rounds (same as sprint: penetration lift + stance toe-pin)
    toe_anchor = np.full((2, N, 2), np.nan)
    for f in range(2):
        for i in range(N):
            if stance[f, i]:
                c, si = np.cos(tgt_pitch[f, i]), np.sin(tgt_pitch[f, i])
                rt = np.array([TOE[0] * c + TOE[2] * si, 0.0,
                               -TOE[0] * si + TOE[2] * c])
                toe_anchor[f, i] = (tgt_pos[f, i] + rt)[:2]
    n_fix = 0
    tgt_xyz = tgt_pos.copy()
    for rnd in range(4):
        dirty = 0
        for i in range(N):
            d.qpos[:] = qpos[i]; mujoco.mj_kinematics(m, d)
            need = False
            tp = tgt_xyz[:, i]
            for f in range(2):
                R = d.xmat[bids[f]].reshape(3, 3)
                lowz = float((d.xpos[bids[f]] + sole_pts @ R.T)[:, 2].min())
                if 0.0005 - lowz > 1e-4:
                    tp[f, 2] += min(0.0005 - lowz, 0.04); need = True
                if stance[f, i]:
                    toe_w = (d.xpos[bids[f]] + R @ TOE)[:2]
                    drift = toe_w - toe_anchor[f, i]
                    if np.linalg.norm(drift) > 0.002:
                        tp[f, :2] -= np.clip(drift, -0.03, 0.03); need = True
            if not need:
                continue
            dirty += 1
            q_frame[:] = qpos[i]
            solve_legs(m, d, q_frame, tp, tgt_pitch[:, i])
            qpos[i] = q_frame
        n_fix = max(n_fix, dirty)
        if dirty == 0:
            break
    if n_fix:
        print(f"[jump_g1] contact-fix pass adjusted {n_fix} frames "
              f"({rnd + 1} rounds)")

    npz = save_reference(qpos, fps, par, tag="jump_g1")
    apex = par.z_takeoff + vz * vz / (2 * G)
    print(f"[jump_g1] {N} frames @ {fps} fps ({N/fps:.1f} s), "
          f"{par.n_jumps} jumps x {par.jump_dist} m, apex z {apex:.2f} m, "
          f"worst IK residual {worst*1000:.2f} mm")
    print(f"[jump_g1] wrote {npz}")


if __name__ == "__main__":
    main()
