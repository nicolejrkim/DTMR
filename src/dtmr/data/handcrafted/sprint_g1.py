"""Handcraft a G1 sprint reference motion — no human mocap, no retargeter.

Task-space gait synthesis + MuJoCo leg IK against the tracker's G1 model:
  * footholds are world-fixed during stance (zero foot slip by construction)
  * duty factor < 0.5 gives true flight phases; root bounce is phased so the
    apex sits mid-flight (ballistic-consistent to first order)
  * swing path is a handcrafted heel-recovery arc (sprint form, high tuck)
  * arms pump counter-phase in joint space; waist counter-rotates
  * stand -> accelerate -> steady sprint -> decelerate -> stand envelope

Output: the tracker's reference files (state CSV + link_world_poses npz) via
`robot_motion.write_reference`, so the clip drops straight into a config's
`motion_path`.  Only G1 geometry is encoded (foot offsets, ankle height, joint
names); other robots need their own constants.

  python -m dtmr.data.handcrafted.sprint_g1 \
      --out-dir data/handcrafted [--speed 3.0]
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from dtmr.data.handcrafted import G1_XML, add_output_args, save_reference

# qpos layout (36): root pos(3) + quat wxyz(4) + 29 joints in model order.
LEG_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right")
              for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                        "ankle_pitch", "ankle_roll")]
ANKLE_Z_FLAT = 0.0361          # ankle_roll link origin height when sole is flat
FOOT_Y = 0.1185                # lateral foot offset at zero pose


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def rotvec_of(R):
    """Rotation vector of a 3x3 matrix."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    v = np.empty(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def yaw_pitch_quat(yaw, pitch, roll):
    """wxyz quaternion from yaw*pitch*roll (world z, then y, then x)."""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    # q = qz(yaw) * qy(pitch) * qx(roll)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


# ---------------------------------------------------------------- gait plan
def plan(par):
    fps = par.fps
    t_stand0, t_acc, t_run, t_dec, t_stand1 = (
        par.stand_in, par.accel, par.steady, par.decel, par.stand_out)
    T_total = t_stand0 + t_acc + t_run + t_dec + t_stand1
    N = int(round(T_total * fps))
    t = np.arange(N) / fps

    # speed envelope
    v = np.zeros(N)
    a0, a1 = t_stand0, t_stand0 + t_acc
    d0, d1 = a1 + t_run, a1 + t_run + t_dec
    v = np.where((t >= a0) & (t < a1), par.speed * smoothstep((t - a0) / t_acc), v)
    v = np.where((t >= a1) & (t < d0), par.speed, v)
    v = np.where((t >= d0) & (t < d1), par.speed * smoothstep(1 - (t - d0) / t_dec), v)
    x_root = np.cumsum(v) / fps

    # cadence scales with speed^0.6 (shorter, quicker steps when slow)
    f_cyc = np.where(v > 0.05, par.f_min + (par.f_cycle - par.f_min)
                     * (v / par.speed) ** 0.6, 0.0)
    phi = np.cumsum(f_cyc) / fps           # global cycle phase

    blend = np.clip(v / par.speed, 0, 1)   # posture blend stand->sprint
    return dict(N=N, t=t, v=v, x=x_root, phi=phi, blend=blend, fps=fps)


TOE = np.array([0.13, 0.0, -ANKLE_Z_FLAT])   # ankle origin -> toe contact point


def toe_pivot(flat_foot, pitch):
    """Ankle position when the flat foothold pivots on its toe by `pitch`."""
    c, si = np.cos(pitch), np.sin(pitch)
    rt = np.array([TOE[0] * c + TOE[2] * si, 0.0, -TOE[0] * si + TOE[2] * c])
    toe_w = flat_foot + TOE
    return toe_w - rt


def stance_pitch(s_st, par):
    """Foot pitch over stance phase s_st in [0,1]: forefoot landing settles
    flat, then heel-off ramps up for toe-off."""
    land = par.land_pitch * (1 - smoothstep(s_st / 0.18))
    heel = par.heeloff_pitch * smoothstep((s_st - 0.55) / 0.45) ** 1.5
    return land + heel


def swing_shape(s, lo, lo_pitch, td, td_pitch, par, v_td):
    """Ankle path for one swing, s in [0,1]. lo/td = 3D ankle liftoff/landing.
    Handcrafted sprint arc: toe-off -> high heel tuck -> reach -> land."""
    ease = smoothstep(s)
    base = lo + (td - lo) * ease
    # vertical arc on top: high tuck peaking ~40% through swing
    h = par.step_height * (0.35 + 0.65 * min(v_td / par.speed, 1.0))
    z_arc = h * np.sin(np.pi * np.clip(s, 0, 1) ** 0.85) ** 1.2
    pitch = (lo_pitch * np.exp(-(s / 0.20) ** 2)              # carry toe-off pitch
             - 0.05 * np.exp(-((s - 0.55) / 0.22) ** 2)       # clearance dorsiflex
             + td_pitch * np.exp(-((s - 1.0) / 0.18) ** 2))   # forefoot landing
    return base + np.array([0, 0, z_arc]), pitch


def foot_targets(g, par):
    """Per-frame 6D ankle targets for both feet from the phase plan."""
    N = g["N"]
    duty = par.duty
    tgt_pos = np.zeros((2, N, 3))
    tgt_pitch = np.zeros((2, N))
    stance = np.zeros((2, N), dtype=bool)

    for leg, (off, sign) in enumerate([(0.0, +1), (0.5, -1)]):  # left, right
        y_nom = sign * par.foot_y
        pl = g["phi"] + off
        moving = g["v"] > 0.05
        foot = np.array([0.0, y_nom, ANKLE_Z_FLAT])   # current flat foothold
        pos = np.zeros((N, 3)); pitch = np.zeros(N); st = np.zeros(N, bool)
        i = 0
        while i < N:
            if not moving[i]:
                pos[i] = foot; st[i] = True; i += 1; continue
            frac = pl[i] % 1.0
            if frac < duty:                      # stance: toe-pivot on foothold
                th = stance_pitch(frac / duty, par)
                pos[i] = toe_pivot(foot, th); pitch[i] = th
                st[i] = True; i += 1; continue
            # swing: find the end of this swing segment
            j = i
            while j < N and moving[j] and (pl[j] % 1.0) >= duty:
                j += 1
            lo = toe_pivot(foot, par.heeloff_pitch)
            v_td = g["v"][min(j, N - 1)]
            f_td = par.f_min + (par.f_cycle - par.f_min) * (v_td / par.speed) ** 0.6
            lead = par.lead_frac * v_td * duty / max(f_td, 1e-6)
            td_flat = np.array([g["x"][min(j, N - 1)] + lead, y_nom, ANKLE_Z_FLAT])
            td = toe_pivot(td_flat, par.land_pitch)
            for k in range(i, j):
                s = (pl[k] % 1.0 - duty) / (1 - duty)
                pos[k], pitch[k] = swing_shape(
                    s, lo, par.heeloff_pitch, td, par.land_pitch, par, v_td)
            foot = td_flat
            i = j
        tgt_pos[leg], tgt_pitch[leg], stance[leg] = pos, pitch, st
    return tgt_pos, tgt_pitch, stance


def clamp_reach(m, d, root_pos, root_quat, tgt_pos, max_reach=0.648):
    """Soft-clamp ankle targets into the reachable sphere around each hip."""
    n_clamped, worst = 0, 0.0
    for i in range(len(root_pos)):
        d.qpos[:] = 0
        d.qpos[0:3] = root_pos[i]; d.qpos[3:7] = root_quat[i]
        mujoco.mj_kinematics(m, d)
        for f, side in enumerate(("left", "right")):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{side}_hip_pitch_link")
            hip = d.xpos[bid].copy()
            v = tgt_pos[f, i] - hip
            dist = np.linalg.norm(v)
            if dist > max_reach:
                tgt_pos[f, i] = hip + v * (max_reach / dist)
                n_clamped += 1; worst = max(worst, dist - max_reach)
    return n_clamped, worst


# ---------------------------------------------------------------- root plan
def root_plan(g, par):
    N = g["N"]
    duty = par.duty
    b = g["blend"]
    z = par.stand_height + (par.run_height - par.stand_height) * b
    # double-frequency bounce: min at mid-stance (phi=duty/2), max mid-flight
    bounce = -par.bounce * np.cos(4 * np.pi * (g["phi"] - duty / 2))
    z = z + bounce * b
    pitch = par.lean * b + par.pitch_osc * b * np.cos(4 * np.pi * g["phi"]) 
    yaw = par.yaw_osc * b * np.sin(2 * np.pi * g["phi"])       # +yaw when L leg fwd
    roll = par.roll_osc * b * np.sin(2 * np.pi * g["phi"])
    quat = np.stack([yaw_pitch_quat(yaw[i], pitch[i], roll[i]) for i in range(N)])
    pos = np.stack([g["x"], np.zeros(N), z], axis=1)
    return pos, quat, yaw


# ---------------------------------------------------------------- arms/waist
def upper_body(g, par, root_yaw):
    """Joint-space sprint arm pump + waist counter-rotation.
    Signs (probed): +shoulder_pitch = arm back; elbow qpos 0 = 90deg bent,
    + straightens; +L shoulder_roll = outward."""
    N = g["N"]; b = g["blend"]
    swing = np.sin(2 * np.pi * g["phi"])   # +1 ~ left leg forward
    ub = {}
    # left arm counter-phase to left leg: arm back (+) when left leg fwd (+)
    sp_l = b * (par.arm_mean + par.arm_amp * swing)
    sp_r = b * (par.arm_mean - par.arm_amp * swing)
    ub["left_shoulder_pitch_joint"] = sp_l
    ub["right_shoulder_pitch_joint"] = sp_r
    ub["left_shoulder_roll_joint"] = b * par.arm_roll
    ub["right_shoulder_roll_joint"] = -b * par.arm_roll
    ub["left_shoulder_yaw_joint"] = np.zeros(N)
    ub["right_shoulder_yaw_joint"] = np.zeros(N)
    # elbow: ~90deg bent (qpos ~0), extend a touch on the backswing
    ub["left_elbow_joint"] = b * (par.elbow_mean + par.elbow_amp * swing)
    ub["right_elbow_joint"] = b * (par.elbow_mean - par.elbow_amp * swing)
    for s in ("left", "right"):
        for w in ("wrist_roll", "wrist_pitch", "wrist_yaw"):
            ub[f"{s}_{w}_joint"] = np.zeros(N)
    # waist: shoulders counter-rotate the pelvis yaw; slight forward lean
    ub["waist_yaw_joint"] = -par.waist_counter * root_yaw
    ub["waist_roll_joint"] = np.zeros(N)
    ub["waist_pitch_joint"] = b * par.waist_lean
    return ub


# ---------------------------------------------------------------- leg IK
def solve_legs(m, d, qpos, tgt_pos, tgt_pitch, iters=40):
    """Damped-LS IK for the 12 leg joints; both feet 6D. qpos edited in place."""
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEG_JOINTS]
    qadr = np.array([m.jnt_qposadr[j] for j in jids])
    vadr = np.array([m.jnt_dofadr[j] for j in jids])
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
            for s in ("left", "right")]
    q_nom = np.array([ -0.35, 0, 0, 0.7, -0.35, 0] * 2)   # bent-knee posture pull
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))

    d.qpos[:] = qpos
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)          # jacobians need cdof
        err = np.zeros(12); J = np.zeros((12, 12))
        for f in range(2):
            bid = bids[f]
            e_p = tgt_pos[f] - d.xpos[bid]
            R_tgt = np.array([[np.cos(tgt_pitch[f]), 0, np.sin(tgt_pitch[f])],
                              [0, 1, 0],
                              [-np.sin(tgt_pitch[f]), 0, np.cos(tgt_pitch[f])]])
            R_cur = d.xmat[bid].reshape(3, 3)
            e_r = rotvec_of(R_tgt @ R_cur.T)
            mujoco.mj_jacBody(m, d, jacp, jacr, bid)
            err[f*6:f*6+3] = e_p; err[f*6+3:f*6+6] = 0.5 * e_r
            J[f*6:f*6+3] = jacp[:, vadr]; J[f*6+3:f*6+6] = 0.5 * jacr[:, vadr]
        if np.linalg.norm(err) < 1e-6:
            break
        q = d.qpos[qadr]
        JtJ = J.T @ J + 1e-4 * np.eye(12)
        dq = np.linalg.solve(JtJ, J.T @ err + 1e-3 * (q_nom - q))
        d.qpos[qadr] = np.clip(q + np.clip(dq, -0.4, 0.4), lo, hi)
    qpos[qadr] = d.qpos[qadr]
    mujoco.mj_kinematics(m, d)
    res = [np.linalg.norm(tgt_pos[f] - d.xpos[bids[f]]) for f in range(2)]
    return max(res)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    add_output_args(ap, default_name="hc_sprint")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--speed", type=float, default=3.0, help="steady sprint m/s")
    ap.add_argument("--f-cycle", type=float, default=2.1, help="stride Hz at top speed")
    ap.add_argument("--f-min", type=float, default=1.4)
    ap.add_argument("--lead-frac", type=float, default=0.5,
                    help="touchdown lead as fraction of stance travel")
    ap.add_argument("--land-pitch", type=float, default=0.10, help="forefoot strike rad")
    ap.add_argument("--heeloff-pitch", type=float, default=0.35, help="toe-off heel lift rad")
    ap.add_argument("--duty", type=float, default=0.40, help="stance fraction (<0.5 = flight)")
    ap.add_argument("--stand-in", type=float, default=0.6)
    ap.add_argument("--accel", type=float, default=1.6)
    ap.add_argument("--steady", type=float, default=4.0)
    ap.add_argument("--decel", type=float, default=1.6)
    ap.add_argument("--stand-out", type=float, default=0.8)
    ap.add_argument("--stand-height", type=float, default=0.72)
    ap.add_argument("--run-height", type=float, default=0.66)
    ap.add_argument("--bounce", type=float, default=-1.0,
                    help="root bounce amplitude m; -1 = ballistic-consistent 9.81/(4*pi*f)^2")
    ap.add_argument("--lean", type=float, default=0.10, help="root fwd pitch rad")
    ap.add_argument("--pitch-osc", type=float, default=0.02)
    ap.add_argument("--yaw-osc", type=float, default=0.08)
    ap.add_argument("--roll-osc", type=float, default=0.025)
    ap.add_argument("--waist-lean", type=float, default=0.08)
    ap.add_argument("--waist-counter", type=float, default=1.5)
    ap.add_argument("--foot-y", type=float, default=0.105, help="sprint track half-width")
    ap.add_argument("--step-height", type=float, default=0.20, help="swing apex above sole")
    ap.add_argument("--arm-mean", type=float, default=0.15)
    ap.add_argument("--arm-amp", type=float, default=0.85)
    ap.add_argument("--arm-roll", type=float, default=0.15)
    ap.add_argument("--elbow-mean", type=float, default=-0.10)
    ap.add_argument("--elbow-amp", type=float, default=0.30)
    par = ap.parse_args()

    if par.bounce < 0:
        par.bounce = 9.81 / (4 * np.pi * par.f_cycle) ** 2
        print(f"[sprint_g1] ballistic bounce amplitude {par.bounce*1000:.1f} mm")
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    d = mujoco.MjData(m)

    g = plan(par)
    tgt_pos, tgt_pitch, stance = foot_targets(g, par)
    root_pos, root_quat, root_yaw = root_plan(g, par)
    nc, wc = clamp_reach(m, d, root_pos, root_quat, tgt_pos)
    if nc:
        print(f"[sprint_g1] reach-clamped {nc} foot targets "
              f"(worst overshoot {wc*1000:.1f} mm)")
    ub = upper_body(g, par, root_yaw)

    N = g["N"]
    qpos = np.zeros((N, m.nq))
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_quat
    # upper body joints by name
    for name, traj in ub.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[:, m.jnt_qposadr[jid]] = traj

    jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in LEG_JOINTS]
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{sd}_ankle_roll_link")
            for sd in ("left", "right")]
    # sole sample points: collision box corners + toe/heel capsule bottoms
    sole_pts = np.array(
        [[0.04 + sx * 0.09, sy * 0.03, -0.037] for sx in (-1, 1) for sy in (-1, 1)]
        + [[0.075, -0.026, -0.035], [0.075, 0.026, -0.035]])

    def sole_min(i):
        d.qpos[:] = qpos[i]
        mujoco.mj_kinematics(m, d)
        return [float((d.xpos[b] + sole_pts @ d.xmat[b].reshape(3, 3).T)[:, 2].min())
                for b in bids]

    worst = 0.0
    q_frame = qpos[0].copy()
    for i in range(N):
        q_frame[:] = qpos[i, :]
        if i > 0:
            q_frame[jadr] = qpos[i - 1, jadr]   # warm start legs
        r = solve_legs(m, d, q_frame, tgt_pos[:, i], tgt_pitch[:, i])
        qpos[i] = q_frame
        worst = max(worst, r)

    # pass 2: wherever the achieved sole penetrates, lift that frame's ankle
    # target by the penetration depth and re-solve (ankle-limit clamping makes
    # the foot flatter than commanded, which digs the sole in a few mm)
    # planned toe anchor per stance frame: the flat foothold's toe point.
    # (targets during stance are toe_pivot(foothold, pitch), so the foothold's
    # toe is tgt + R_y(pitch) @ TOE, which is pitch-invariant by construction)
    toe_anchor = np.full((2, N, 2), np.nan)
    for f in range(2):
        for i in range(N):
            if stance[f, i]:
                c, si = np.cos(tgt_pitch[f, i]), np.sin(tgt_pitch[f, i])
                rt = np.array([TOE[0] * c + TOE[2] * si, 0.0,
                               -TOE[0] * si + TOE[2] * c])
                toe_anchor[f, i] = (tgt_pos[f, i] + rt)[:2]

    def toe_xy(i_frame):
        d.qpos[:] = qpos[i_frame]
        mujoco.mj_kinematics(m, d)
        return [(d.xpos[b] + d.xmat[b].reshape(3, 3) @ TOE)[:2].copy()
                for b in bids]

    # correction rounds: lift the sole out of the ground AND pin the stance
    # toe to its planned anchor (ankle-limit clamping breaks both otherwise)
    n_fix = 0
    tgt_xyz = tgt_pos.copy()
    for rnd in range(4):
        dirty = 0
        for i in range(N):
            lows = sole_min(i)
            toes = toe_xy(i)
            need = False
            tp = tgt_xyz[:, i]
            for f in range(2):
                lift = 0.0005 - lows[f]
                if lift > 1e-4:
                    tp[f, 2] += min(lift, 0.04); need = True
                if stance[f, i]:
                    drift = toes[f] - toe_anchor[f, i]
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
        print(f"[sprint_g1] contact-fix pass adjusted {n_fix} frames "
              f"({rnd + 1} rounds)")

    npz = save_reference(qpos, par.fps, par, tag="sprint_g1")
    dist = g["x"][-1]
    print(f"[sprint_g1] {N} frames @ {par.fps} fps  ({N/par.fps:.1f} s), "
          f"distance {dist:.1f} m, worst IK residual {worst*1000:.2f} mm")
    print(f"[sprint_g1] wrote {npz}")


if __name__ == "__main__":
    main()
