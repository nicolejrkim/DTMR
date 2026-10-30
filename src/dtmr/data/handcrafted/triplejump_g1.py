"""Handcraft a G1 competition triple jump (World Athletics foot pattern).

Run-up sprint -> takeoff from the board -> HOP (take off and land on the SAME
foot, right) -> STEP (land on the OPPOSITE foot, left) -> JUMP (land on both
feet, deep absorb).  Same construction as the other handcrafted generators:
task-space plan (world-pinned footholds, exact ballistic flights, Hermite
velocity-matched stances) + damped-LS leg IK + keyframed double-arm technique.

Output: the tracker's reference files (state CSV + link_world_poses npz).

  python -m dtmr.data.handcrafted.triplejump_g1 --out-dir data/handcrafted
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from dtmr.data.handcrafted import G1_XML, add_output_args, save_reference
from dtmr.data.handcrafted.sprint_g1 import (LEG_JOINTS, ANKLE_Z_FLAT, TOE, smoothstep,
                            toe_pivot, yaw_pitch_quat, solve_legs)
from dtmr.data.handcrafted.jump_g1 import hermite

G = 9.81
FPS = 50
FOOT_Y = 0.115
L, R = 0, 1


class Timeline:
    """Sequential builder: root trajectory + per-foot pin/waypoint records."""

    def __init__(self):
        self.X, self.Z, self.RP = [], [], []      # root x, z, pitch
        self.ARM = []                             # (shoulder, elbow, waist)/frame
        self.pin = {L: [], R: []}                 # foothold x or None per frame
        self.fpitch = {L: [], R: []}              # pinned foot pitch per frame
        self.wp = {L: [], R: []}                  # (frame, dx_root, z, pitch, arc)
        self.x, self.vx, self.z = 0.0, 0.0, 0.72
        self.arm = (0.10, 0.0, 0.02)
        self.rp = 0.02
        self.hold = {L: 0.0, R: 0.0}              # current foothold if pinned
        self.down = {L: True, R: True}

    def seg(self, dur, vx1=None, z1=None, vz0=0.0, vz1=0.0, ballistic=False,
            arm=None, rpitch=None, pin_on=None, pin_off=None, lead=0.2,
            fp=None):
        """Append one segment. pin_on: (foot, lead) pins foot at seg start.
        pin_off: foot lifts at seg start. fp: {foot: (p0,p1)} pinned pitch ramp."""
        n = max(1, int(round(dur * FPS)))
        vx1 = self.vx if vx1 is None else vx1
        z1 = self.z if z1 is None else z1
        arm1 = self.arm if arm is None else arm
        rp1 = self.rp if rpitch is None else rpitch
        if pin_off is not None:
            for f in (pin_off if isinstance(pin_off, tuple) else (pin_off,)):
                self.down[f] = False
        if pin_on is not None:
            for f in (pin_on if isinstance(pin_on, tuple) else (pin_on,)):
                self.down[f] = True
                self.hold[f] = self.x + lead
        x0, vx0, z0, arm0, rp0 = self.x, self.vx, self.z, self.arm, self.rp
        x1 = x0 + dur * (vx0 + vx1) / 2.0
        for k in range(n):
            s = (k + 1.0) / n
            t = s * dur
            self.X.append(hermite(s, x0, x1, vx0, vx1, dur))
            if ballistic:
                self.Z.append(z0 + vz0 * t - 0.5 * G * t * t)
            else:
                self.Z.append(hermite(s, z0, z1, vz0, vz1, dur))
            e = smoothstep(s)
            self.RP.append(rp0 + (rp1 - rp0) * e)
            self.ARM.append(tuple(a + (b - a) * e for a, b in zip(arm0, arm1)))
            for f in (L, R):
                if self.down[f]:
                    self.pin[f].append(self.hold[f])
                    p0, p1 = fp.get(f, (0.0, 0.0)) if fp else (0.0, 0.0)
                    self.fpitch[f].append(p0 + (p1 - p0) * e)
                else:
                    self.pin[f].append(None)
                    self.fpitch[f].append(0.0)
        self.x, self.vx = x1, vx1
        self.z = self.Z[-1]
        self.arm, self.rp = arm1, rp1

    def waypoint(self, foot, dx, z, pitch=0.0, arc=0.10, back=0):
        """Author a swing waypoint at (current end frame - back), positioned
        dx ahead of the root at that frame."""
        i = len(self.X) - 1 - back
        self.wp[foot].append((i, dx, z, pitch, arc))


def main():
    ap = argparse.ArgumentParser()
    add_output_args(ap, default_name="hc_triplejump")
    ap.add_argument("--v-board", type=float, default=5.0, help="approach speed m/s")
    ap.add_argument("--t-hop", type=float, default=0.37)
    ap.add_argument("--t-step", type=float, default=0.34)
    ap.add_argument("--t-jump", type=float, default=0.43)
    par = ap.parse_args()

    tl = Timeline()
    A = lambda sh, el, w=0.02: (sh, el, w)

    # ---------------- approach: stand -> 6 accelerating steps --------------
    tl.seg(0.6)                                                   # stand
    # first push: L lifts, root starts moving on R support
    tl.seg(0.17, vx1=1.0, z1=0.69, arm=A(0.35, 0.10), rpitch=0.08, pin_off=L)
    tl.waypoint(L, 0.05, 0.12, arc=0.06)
    tl.seg(0.15, vx1=1.6, z1=0.655, arm=A(-0.15, 0.05), pin_on=L, lead=0.26)
    # L1 stance: R lifts and swings
    tl.seg(0.10, vx1=2.0, z1=0.645, pin_off=R, arm=A(0.35, 0.10))
    tl.waypoint(R, 0.0, 0.14, arc=0.06)
    tl.seg(0.08, vx1=2.2, vz0=0.0, vz1=0.34, arm=A(0.4, 0.10))
    # accelerating run: alternating stances up to board speed
    vs = np.linspace(2.9, par.v_board, 4)
    t_sts = np.linspace(0.15, 0.13, 4)
    t_fls = np.linspace(0.07, 0.09, 4)
    arms = [A(-0.3, 0.05), A(0.4, 0.10), A(-0.35, 0.05), A(0.45, 0.12)]
    foot_cycle = [R, L, R, L]
    prev = L
    for foot, v_out, t_st, t_fl, armk in zip(foot_cycle, vs, t_sts, t_fls, arms):
        vz_f = G * t_fl / 2
        tl.seg(t_fl, ballistic=True, vz0=vz_f, pin_off=prev)      # flight
        tl.seg(t_st, vx1=v_out, z1=0.635, vz0=-vz_f, vz1=0.0,
               pin_on=foot, lead=0.24, arm=armk,
               fp={foot: (0.08, 0.32)})
        tl.waypoint(R if foot == L else L, -0.02, 0.15, arc=0.07)
        prev = foot
    # onto the board
    tl.seg(0.10, ballistic=True, vz0=0.49, arm=A(0.55, 0.15),
           pin_off=L)

    # ---------------- BOARD takeoff (right foot) ---------------------------
    vz_hop = 1.72
    v_bd = par.v_board - 0.3
    tl.seg(0.14, vx1=v_bd, z1=0.665, vz0=-0.49, vz1=vz_hop,
           pin_on=R, lead=0.20,
           arm=A(-0.85, 0.30, 0.0), rpitch=0.04, fp={R: (0.10, 0.55)})
    tl.waypoint(L, -0.25, 0.15, arc=0.05)

    # ---------------- HOP: fly and land on the SAME (right) foot -----------
    x_board = tl.hold[R]
    tl.seg(par.t_hop, ballistic=True, vz0=vz_hop, pin_off=R,
           arm=A(-0.45, 0.15), rpitch=0.03)
    tl.waypoint(R, -0.10, 0.30, pitch=0.1, arc=0.16, back=int(par.t_hop*FPS*0.5))
    tl.waypoint(L, -0.15, 0.20, arc=0.06, back=int(par.t_hop*FPS*0.4))
    vz_td = vz_hop - G * par.t_hop
    vz_step = 1.57
    v_hop = par.v_board * 0.82
    tl.seg(0.15, vx1=v_hop, z1=0.66, vz0=vz_td, vz1=vz_step,      # hop stance (R)
           pin_on=R, lead=0.20, arm=A(0.5, 0.10), rpitch=0.10,
           fp={R: (0.10, 0.55)})
    x_hop = tl.hold[R]
    tl.waypoint(L, 0.12, 0.24, arc=0.08)                          # knee drive

    # ---------------- STEP: land on the OPPOSITE (left) foot ---------------
    tl.seg(par.t_step, ballistic=True, vz0=vz_step, pin_off=R,
           arm=A(-0.75, 0.25), rpitch=0.04)
    tl.waypoint(R, -0.28, 0.18, arc=0.06, back=int(par.t_step*FPS*0.5))
    vz_td = vz_step - G * par.t_step
    vz_jump = 1.94
    v_step = par.v_board * 0.70
    tl.seg(0.15, vx1=v_step, z1=0.67, vz0=vz_td, vz1=vz_jump,     # step stance (L)
           pin_on=L, lead=0.20, arm=A(0.5, 0.10), rpitch=0.10,
           fp={L: (0.10, 0.55)})
    x_step = tl.hold[L]
    tl.waypoint(R, 0.05, 0.22, arc=0.08)

    # ---------------- JUMP: fly and land on BOTH feet ----------------------
    tl.seg(par.t_jump, ballistic=True, vz0=vz_jump, pin_off=L,
           arm=A(-1.0, 0.35, 0.05), rpitch=0.06)
    mid = int(par.t_jump * FPS * 0.45)
    tl.waypoint(L, 0.02, 0.28, pitch=-0.15, arc=0.10, back=mid)
    tl.waypoint(R, 0.02, 0.28, pitch=-0.15, arc=0.10, back=mid)
    vz_td = vz_jump - G * par.t_jump
    tl.seg(0.10, vx1=1.2, z1=0.585, vz0=vz_td, vz1=-0.3,          # absorb 1
           pin_on=(L, R), lead=0.32, arm=A(-0.4, 0.20), rpitch=0.16)
    x_land = tl.hold[L]
    tl.seg(0.22, vx1=0.0, z1=0.55, vz0=-0.3, vz1=0.0,             # absorb 2
           arm=A(0.2, 0.05), rpitch=0.10)
    tl.seg(0.5, z1=0.70, arm=A(0.10, 0.0), rpitch=0.02)           # recover
    tl.seg(0.8)                                                   # stand

    # ---------------- swing fill -------------------------------------------
    N = len(tl.X)
    X = np.array(tl.X); Z = np.array(tl.Z)
    tgt_pos = np.zeros((2, N, 3)); tgt_pitch = np.zeros((2, N))
    stance = np.zeros((2, N), dtype=bool)
    for f, sign in ((L, +1), (R, -1)):
        y = sign * FOOT_Y
        pins = tl.pin[f]
        for i in range(N):
            if pins[i] is not None:
                flat = np.array([pins[i], y, ANKLE_Z_FLAT])
                tgt_pos[f, i] = toe_pivot(flat, tl.fpitch[f][i])
                tgt_pitch[f, i] = tl.fpitch[f][i]
                stance[f, i] = True
        # gaps between pinned intervals -> waypoint-guided swings
        i = 0
        while i < N:
            if pins[i] is not None:
                i += 1; continue
            j = i
            while j < N and pins[j] is None:
                j += 1
            lo_i, td_i = i - 1, j
            nodes = [(lo_i, tgt_pos[f, lo_i].copy(), tgt_pitch[f, lo_i], 0.0)]
            for (wf, dx, wz, wpit, arc) in sorted(tl.wp[f]):
                if i <= wf < j:
                    nodes.append((wf, np.array([X[wf] + dx, y, wz]), wpit, arc))
            if td_i < N:
                nodes.append((td_i, tgt_pos[f, td_i].copy(),
                              tgt_pitch[f, td_i], 0.10))
            else:
                nodes.append((N - 1, tgt_pos[f, lo_i].copy(),
                              tgt_pitch[f, lo_i], 0.0))
            for idx, ((a, pa, pia, _), (b, pb, pib, arc)) in enumerate(
                    zip(nodes[:-1], nodes[1:])):
                # carry spans ride with the root (relative x); the final span
                # into touchdown interpolates absolutely so the plant is
                # world-fixed with zero foot velocity at contact
                absolute = (idx == len(nodes) - 2)
                dxa, dxb = pa[0] - X[a], pb[0] - X[b]
                for k in range(max(a + 1, i), min(b + 1, j)):
                    s = (k - a) / max(b - a, 1)
                    e = smoothstep(s)
                    p = pa + (pb - pa) * e
                    if not absolute:
                        p[0] = X[k] + dxa + (dxb - dxa) * e
                    p[2] += arc * np.sin(np.pi * s) ** 1.1
                    tgt_pos[f, k] = p
                    tgt_pitch[f, k] = pia + (pib - pia) * e
            i = j

    # ---------------- qpos assembly + IK -----------------------------------
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    d = mujoco.MjData(m)
    qpos = np.zeros((N, m.nq))
    qpos[:, 0] = X; qpos[:, 2] = Z
    for i in range(N):
        qpos[i, 3:7] = yaw_pitch_quat(0.0, tl.RP[i], 0.0)
    def setj(name, vals):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[:, m.jnt_qposadr[jid]] = vals
    sh = np.array([a[0] for a in tl.ARM]); el = np.array([a[1] for a in tl.ARM])
    wp_ = np.array([a[2] for a in tl.ARM])
    for side, roll in (("left", 0.12), ("right", -0.12)):
        setj(f"{side}_shoulder_pitch_joint", sh)
        setj(f"{side}_shoulder_roll_joint", np.full(N, roll))
        setj(f"{side}_elbow_joint", el)
    setj("waist_pitch_joint", wp_)

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
        worst = max(worst, solve_legs(m, d, q_frame, tgt_pos[:, i], tgt_pitch[:, i]))
        qpos[i] = q_frame

    # contact-fix: penetration lift + pinned toe-pin
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
                Rm = d.xmat[bids[f]].reshape(3, 3)
                lowz = float((d.xpos[bids[f]] + sole_pts @ Rm.T)[:, 2].min())
                if 0.0005 - lowz > 1e-4:
                    tp[f, 2] += min(0.0005 - lowz, 0.04); need = True
                if stance[f, i]:
                    drift = (d.xpos[bids[f]] + Rm @ TOE)[:2] - toe_anchor[f, i]
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
        print(f"[triplejump_g1] contact-fix adjusted {n_fix} frames "
              f"({rnd + 1} rounds)")

    npz = save_reference(qpos, FPS, par, tag="triplejump_g1")
    print(f"[triplejump_g1] {N} frames ({N/FPS:.1f} s), board x={x_board:.2f}: "
          f"hop {x_hop-x_board:.2f} m + step {x_step-x_hop:.2f} m + "
          f"jump {x_land-x_step:.2f} m = {x_land-x_board:.2f} m total, "
          f"worst IK residual {worst*1000:.1f} mm")
    print(f"[triplejump_g1] wrote {npz}")


if __name__ == "__main__":
    main()
