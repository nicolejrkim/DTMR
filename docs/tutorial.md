# Tutorial: track your own clip

One human motion from download to a video of the Unitree G1 tracking it. The clip is a
LAFAN1 walk from Unitree's retargeted release (no account needed, 2.6 MB); from step 2
on, every source in [Reference motions](HUMAN_MOTION.md) works the same way. Run
everything from the repository root in the environment from [Installation](install.md),
or run the whole tutorial as one command with `bash demo_scripts/demo_lafan1.sh`.

## 1. Get a retargeted clip

DTMR tracks a robot motion, so the clip must already be retargeted to the robot.
Unitree's release has the LAFAN1 takes retargeted to the G1:

```bash
mkdir -p input/lafan1
curl -L -o input/lafan1/walk1_subject1.csv \
    https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/walk1_subject1.csv
```

LAFAN1 is CC BY-NC-ND 4.0: fine for trying DTMR, not for redistributing what you make
from it.

## 2. Convert it to a reference

```bash
python -m dtmr.data.lafan1_csv_to_reference input/lafan1/walk1_subject1.csv \
    --robot unitree_g1 --out-dir data/lafan1_walk1 --start 10 --end 20
```

`--start/--end` keep ten seconds of the 261 s take. The importer resamples to 50 Hz,
matches the joints to the G1 model by name and prints the path to track:

```text
[lafan1] 301 frames @ 30 Hz -> 501 frames @ 50 Hz; set motion_path: data/lafan1_walk1/walk1_subject1_50Hz.npz
```

It writes one file, `walk1_subject1_50Hz.npz`: the robot state per frame and every
body's world pose per frame, which is what the tracker reads.

## 3. Look at it before tracking

```bash
python -m dtmr.tools.render_rollout --robot unitree_g1 \
    --ref data/lafan1_walk1/walk1_subject1_50Hz.npz --out output/lafan1_walk1/reference.mp4
```

Without `--run` the renderer plays the reference itself. The robot should stand on the
floor, face its direction of travel and keep every limb the right way round: a problem
in the reference shows up in tracking as a constant error, not as noise. A clip that
starts inside the floor or hovering is left as is; the environment lifts only its reset
pose.

## 4. Track it

```bash
dtmr --config src/dtmr/examples/unitree_g1_release_warp.yaml \
    --motion_path data/lafan1_walk1/walk1_subject1_50Hz.npz \
    --output_dir output/lafan1_walk1 --no_viz --n_steps 30      # a quick check

dtmr --config src/dtmr/examples/unitree_g1_release_warp.yaml \
    --motion_path data/lafan1_walk1/walk1_subject1_50Hz.npz \
    --output_dir output/lafan1_walk1 --no_viz                   # the whole clip
```

`--config` is the robot's release recipe and `--motion_path` the clip. The run stops when
the reference ends; `n_steps` (800 in the G1 config) is only an upper bound at 50 steps
per second of robot time, so raise it with `--n_steps` for clips longer than about ten
seconds. The progress bar shows the reward `rew`, the phase `phi` (0 to 1 through the
reference), the phase rate `rdphi` (above zero: playing faster than real time, below:
slower) and the steps per second `freq`. Expect 1 to 3 minutes of compute per second of
motion on an RTX 4090 ([measured times](quickstart.md#how-long-a-run-takes)).

## 5. Watch the result

```bash
python -m dtmr.tools.render_rollout --run output/lafan1_walk1 --robot unitree_g1 \
    --ref data/lafan1_walk1/walk1_subject1_50Hz.npz --out output/lafan1_walk1/rollout.mp4
```

The solid robot is the simulation, the translucent blue one the reference at the
tracker's phase; where they drift apart in time, DTMR slowed down or sped up the clip.
`--speed 0.5` plays at half speed.

## 6. Read the scores

`<stamp>_phi_tracking_metrics.yaml` compares each step with the reference frame at the
tracker's phase:

| Key | Meaning |
|---|---|
| `phi/mpjpe_world_m` | mean body-position error in the world (m) |
| `phi/mpjpe_relcom_m` | the same, relative to the centre of mass (posture only) |
| `phi/mean_orient_err_deg` | mean body-orientation error |
| `phi/qpos_base_pos_err_m`, `phi/qpos_base_orient_err_deg` | floating-base position and orientation error |
| `phi/qpos_joint_mae_rad` | mean absolute joint-angle error |

`<stamp>_dtw_metrics.yaml` aligns the whole rollout with the whole reference by dynamic
time warping (`mean_cost` per aligned frame, `mean_reward` over the run), scoring the
motion independently of its timing. A good G1 walk is around a centimetre of MPJPE; a
constant offset of many centimetres usually means the reference is mis-scaled, mirrored
or in the floor (step 3).

## Other robots and sources

- **Another robot:** retarget the clip to that robot, pass its name as `--robot` in
  step 2 and use `src/dtmr/examples/<robot>_release_warp.yaml` in step 4.
- **Other sources:** SEED through soma-retargeter, GMR, OmniRetarget, PHUMA, 100STYLE
  or your own retargeter: [Reference motions](HUMAN_MOTION.md). Steps 3 to 6 are
  unchanged.
- **Something went wrong:** [Troubleshooting](troubleshooting.md).
