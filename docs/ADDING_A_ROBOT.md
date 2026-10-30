# Adding a robot

A robot is an MJCF model directory under `src/dtmr/models/<name>/` and one descriptor file,
`src/dtmr/robots/<name>.yaml`. The descriptor holds the facts about the robot (model file,
gains, which bodies are feet, how joints are grouped); run configs hold only tuning and
start with `robot: <name>`. Nothing in the tracker is specific to a robot. The Unitree
G1 is the example below; the Booster K1 (22 joints, no waist) is the simplest shipped
one.

## 1. The model directory

`src/dtmr/models/<name>/` holds:

| File | Purpose |
|---|---|
| `<robot>.xml` | the robot: bodies, joints, `<motor>` actuators, meshes under `assets/` |
| `scene_mjx_torque_tracking.xml` | the tracking scene: includes the robot, adds a floor and lights, and the `home` / `knees_bent` keyframes |
| `LICENSE`, `README.md` | the vendor's licence and where the files came from |

What the environment reads from the MJCF:

* **One free joint** on the base, first in `qpos`, then one hinge per actuated joint:
  the state layout is `root pos (3) | root quat wxyz (4) | joints (n)` in the model's
  joint order.
* **`<motor>` actuators with real `ctrlrange`** on every driven joint. The environment
  clips the applied torque to it, and the PD law computes the torque from the
  descriptor's `kp`/`kd`.
* **Joint names** the reference files use. Importers match joints by name, so keep the
  vendor's names (`left_hip_pitch_joint`, ...) unless a retargeter you rely on uses
  others.
* **Body names** for the cost terms, listed in the descriptor.
* **Keyframes** `home` (the standing sanity check) and `knees_bent` (the reset pose when
  a reference has no state CSV); both optional.
* **Collision pairs** as in the shipped scenes: the floor collides with the feet, knees
  and hands only. Rollouts have a fixed contact budget (120 contacts per rollout, 60 for
  the executed step) and contacts beyond it are dropped silently, so a model whose every
  link collides with everything tracks badly without an error.

Keep the model MJX-compatible (no `<equality>` weld of the base, no tendon-driven
joints, convex collision meshes); MuJoCo Warp supports the same subset. Check it loads:

```bash
python -c "import mujoco; mujoco.MjModel.from_xml_path('src/dtmr/models/<name>/scene_mjx_torque_tracking.xml')"
```

## 2. The descriptor

`src/dtmr/robots/<name>.yaml`, modelled on `unitree_g1.yaml`:

```yaml
model_dir: <name>                            # directory under src/dtmr/models/
model_file: scene_mjx_torque_tracking.xml    # scene inside it

kp: [...]        # one per actuated joint, model order
kd: [...]

joint_groups:                 # case-insensitive glob patterns on joint names
  legs:  ["*hip*", "*knee*", "*ankle*"]
  waist: ["waist_*"]          # omit if the robot has none
  arms:  ["*shoulder*", "*elbow*", "*wrist*"]

cost_config:
  foot_bodies:              [left_ankle_roll_link, right_ankle_roll_link]
  flight_feet_links:        [left_ankle_roll_link, right_ankle_roll_link]
  knee_ground_bodies:       [left_knee_link, right_knee_link]
  undesired_allowed_bodies: [left_ankle_roll_link, right_ankle_roll_link,
                             left_wrist_yaw_link, right_wrist_yaw_link]
  head_keypoint_parent: torso_link
  head_keypoint_offset: [0.0, 0.0, 0.43]
```

| Field | Used by | Notes |
|---|---|---|
| `kp`, `kd` | `position_pd` control | one per actuator; the loader rejects a mismatch |
| `joint_groups.arms` | the arm-channel noise multiplier (`sigma_arm_mult`) | patterns, not indices, so joint order never matters |
| `joint_groups.legs`, `waist` | the lower-body index set for the velocity-reference and stance costs | the groups must cover every actuated joint exactly once |
| `foot_bodies` | foot slip and ground-clearance costs | the body that carries the sole collision geom |
| `flight_feet_links` | flight-phase detection for the dynamism schedule | usually the same as `foot_bodies` |
| `knee_ground_bodies` | knee-to-floor penalty | the shank link on robots whose knee body has no geometry (Booster T1: `Shank_Left`) |
| `undesired_allowed_bodies` | the contact penalty: every other body touching the floor or another body costs | feet, and hands if the motions use them |
| `head_keypoint_parent`, `head_keypoint_offset` | a synthetic head keypoint so torso tilt shows up as position error | the parent must exist; pick the offset so the point sits at head height |

The loader rejects unknown keys, wrong types and a `kp`/`kd` length that differs from
the model's actuator count. A body name the model lacks is only warned about
(`[warp] ...: unknown body names ignored` in the first lines of output), so read the log
of the first run.

## 3. A run config

Copy the release config of the closest shipped robot, set `robot: <name>` and run it
with `--motion_path`. Tuned values transfer between robots of similar size. Across
sizes, the useful horizon grows with the robot's pendulum time (roughly the square root
of hip height over g) and the tolerable noise scale shrinks with mass times hip height.

## 4. A reference motion

Any route in [Reference motions](HUMAN_MOTION.md) works once the retargeter emits your
joint names. The quickest check of the whole chain is the standing sanity reference,
which needs only the `home` keyframe:

```bash
python -m dtmr.data.build_home_hold_reference --robot <name> --out-dir data/home_hold_<name>
```

Point the config at the resulting `home_hold_50Hz.npz` and run with
`--n_steps 100`; the robot should stand still. Then move on to a retargeted clip, and
add the robot to the table in [Supported robots](robots.md).
