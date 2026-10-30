# Reference motions

DTMR tracks a robot motion: a clip that is already on the robot's skeleton. Mapping a
human onto the robot (kinematic retargeting) happens before DTMR, with any tool, and the
tracker needs its output in one small format. This page gives that format, an importer
for each supported source, and the hand-off for a retargeter of your own. The examples
use the Unitree G1.

The paper's humanoid results used NVIDIA's
[soma-retargeter](https://github.com/NVIDIA/soma-retargeter) on
[SEED](https://huggingface.co/datasets/bones-studio/seed) mocap and
[GMR](https://github.com/YanjieZe/GMR) on LAFAN1.

Every retargeter below hands DTMR the same thing and has an importer for the file it
writes:

| Retargeter | Human input | Robots | Writes | Importer |
|---|---|---|---|---|
| NVIDIA [soma-retargeter](https://github.com/NVIDIA/soma-retargeter); the [nicolejrkim fork](https://github.com/nicolejrkim/soma-retargeter) adds three robots | SEED BVH on the SOMA skeleton | G1, T1, X2 Ultra, A3; fork: H1-2, R1, K1 | CSV: root in cm, XYZ Euler degrees, one column per joint in degrees, 120 fps | `soma_csv_to_reference` |
| [GMR](https://github.com/YanjieZe/GMR) (MIT) | BVH (LAFAN1, Xsens), SMPL-X | G1, H1-2, R1, T1 | `.pkl`: `fps`, `root_pos`, `root_rot` xyzw, `dof_pos` | `gmr_pkl_to_reference` |
| [OmniRetarget](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset), code in [holosoma](https://github.com/amazon-far/holosoma) (dataset MIT) | SMPL-X and mocap; released as a G1 dataset | G1 | `.npz`: `fps`, `qpos` (wxyz quaternion, position, 29 joints, optional object pose) | `omniretarget_npz_to_reference` |
| [PHUMA](https://github.com/DAVIAN-Robotics/PHUMA) (Apache-2.0) | SMPL-X from video; released as a G1 dataset | G1 (23 DoF, wrists held at zero) | `.npy` dict: `fps`, `root_trans`, `root_ori` xyzw, `dof_pos` | `phuma_npy_to_reference` |
| the bundled keypoint retargeter | 100STYLE BVH | all seven | the reference itself | `keypoint_retarget`, [batches](STYLE100.md) |
| your own | anything | any | a `RobotMotion` | `write_reference` |

Two datasets have a route of their own and are covered separately below: SEED, the
paper's data, goes through soma-retargeter ([walkthrough](SEED_GUIDE.md)); LAFAN1 comes
from Unitree already retargeted, so it needs no retargeter at all ([tutorial](tutorial.md)).

## The hand-off format

A retargeter hands DTMR a `RobotMotion` (`src/dtmr/data/robot_motion.py`):
per frame, the floating-base position (m) and orientation (unit quaternion, wxyz) in
the world frame and one angle (rad) per actuated joint, with the joint names and the
frame rate. `write_reference` turns it into the one file a config's `motion_path`
points at:

```
<out-dir>/<name>_50Hz.npz    qpos, qvel   root pos | root quat wxyz | joints, and their velocities, per frame
                             pos, rot     world pose of every body per frame (plus link_names, timestamps, fps)
```

Joints are matched to the robot model by name, so the retargeter's joint order does not
matter, a joint it does not drive is held at zero, and a clip retargeted to the wrong
robot fails with a message instead of permuting limbs. Velocities are finite
differences, the clip is resampled to 50 Hz, and `--start/--end` cut a window out of a
long take. Every importer shares the flags `--robot --out-dir --name --target-fps
--start --end`.

## SEED through soma-retargeter

The [SEED walkthrough](SEED_GUIDE.md) covers it end to end; the core:

```bash
python -m dtmr.data.seed_fetch fetch --moves egypt_dance_R_003__A275 --out data/seed_bvh
python ./app/bvh_to_csv_converter.py --config seed_to_g1.json --viewer null   # in the soma-retargeter checkout
python -m dtmr.data.soma_csv_to_reference data/soma_g1/egypt_dance_R_003__A275.csv \
    --robot unitree_g1 --out-dir data/egypt_dance
```

soma-retargeter writes centimetres, XYZ Euler degrees and one column per joint in
degrees at 120 fps (named by the joint from v0.2.0 on, `<joint>_dof` in v0.1.0); the
importer undoes all of that. Without a Hugging Face token, the
soma-retargeter checkout ships ten SEED clips already retargeted to the G1 under
`assets/motions/csv/`; they convert directly and carry SEED's licence and credit line.

## LAFAN1 through Unitree's retargeted release

LAFAN1 is CC BY-NC-ND 4.0, so nothing derived from it ships here. Unitree's
[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
(ungated) has the whole set retargeted to the G1, H1 and H1-2 as header-less 30 fps
CSVs; the importer reads the G1 split:

```bash
mkdir -p input/lafan1
curl -L -o input/lafan1/walk1_subject1.csv \
    https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/walk1_subject1.csv
python -m dtmr.data.lafan1_csv_to_reference input/lafan1/walk1_subject1.csv \
    --robot unitree_g1 --out-dir data/lafan1_walk1 --start 10 --end 20
```

The takes are minutes long, hence `--start/--end`. The [tutorial](tutorial.md)
continues from here.

## GMR, OmniRetarget and PHUMA

Three public retargeters for the G1 save the same quantities, each in its own container
(the table above lists the files), and each has an importer:

```bash
python -m dtmr.data.gmr_pkl_to_reference data/gmr/<clip>.pkl --robot unitree_g1 --out-dir data/<clip>
python -m dtmr.data.omniretarget_npz_to_reference robot-terrain/<clip>.npz --out-dir data/<clip>
python -m dtmr.data.phuma_npy_to_reference data/humanoid_pose/g1/<clip>.npy --out-dir data/<clip>
```

**GMR.** Install it in its own environment (Python 3.10, `pip install -e .` in the
checkout) and retarget a BVH or SMPL-X file; the pickle it saves is what the importer
reads:

```bash
# in the GMR checkout
python scripts/bvh_to_robot.py --bvh_file <clip>.bvh --robot unitree_g1 --format lafan1 --save_path data/gmr/<clip>.pkl
python scripts/smplx_to_robot.py --smplx_file <clip>.npz --robot unitree_g1 --save_path data/gmr/<clip>.pkl

# in the DTMR environment
python -m dtmr.data.gmr_pkl_to_reference data/gmr/<clip>.pkl --robot unitree_g1 --out-dir data/gmr_<clip>
```

GMR's models of the G1, H1-2 and T1 list the joints in the same order as the models
here, so `--robot unitree_h1_2` and `--robot booster_t1` work the same way. Its R1 model
has five wrist joints the model here lacks, which are dropped; its K1 uses other joint
names and is not wired.

**OmniRetarget**'s code ships in Amazon's holosoma as `holosoma_retargeting`, and its
released dataset holds G1 trajectories (object carrying, terrain, both) as one `.npz`
per clip. The retargeter saves the same `qpos` and `fps` keys (plus `human_joints` and
`cost`, which the importer ignores); the importer reads the robot part and skips an
appended object pose.

**PHUMA** retargets to the 23-DoF G1, so its clips leave the six wrist joints at zero;
the importer allows that and says so.

## 100STYLE through the keypoint retargeter

One open dataset ships with a reader and a small kinematic retargeter onto any of the
seven robots:

| Dataset | What | Licence |
|---|---|---|
| [100STYLE](https://www.ianxmason.com/100style/) | 100 walking and running styles, each forward, backward, sideways and idle; 810 takes, 60 fps BVH, 1.4 GB | CC BY 4.0: credit "The 100STYLE Dataset - Ian Mason" and cite Mason, Starke and Komura (2022) |

```bash
python -m dtmr.data.bvh_100style fetch --out data/100style      # 1.4 GB, once
python -m dtmr.data.keypoint_retarget data/100style/100STYLE/Zombie/Zombie_FW.bvh \
    --robot unitree_g1 --out-dir data/100style_ref/unitree_g1 --start 0 --end 10
```

The retargeter scales the human to the robot's leg length, puts the robot's feet on the
human's foot paths (planted feet weigh most) and solves every frame by damped least
squares against the joint limits; it prints the worst foot error in stance and in swing
(within 4 mm on 10 s of six styles across all seven robots). It matches keypoints, not
the full body shape: a quick way to many clips, not a replacement for soma-retargeter or
GMR. [Running the 100 styles](STYLE100.md) batches it over the whole dataset.

## Handcrafted G1 motions

`src/dtmr/data/handcrafted/` plans G1 motions in task space and solves the
legs with IK: a sprint with flight phases, standing broad jumps and a triple jump, every
dynamic parameter a flag (`--speed`, `--jump-dist`, ...). `python -m dtmr.demo prepare`
builds all three and `python -m dtmr.demo run g1_jump` tracks one.

## Your own retargeter

Every importer above is one file: a reader that builds a `RobotMotion` and a `main`
with the shared flags. A new source is a copy of one of them with the reader replaced.
Three ways in, in order of effort:

1. **A soma-style CSV.** `root_translate{X,Y,Z}` (cm), `root_rotate{X,Y,Z}` (XYZ Euler,
   degrees) and one `<joint_name>` column per joint (degrees): `soma_csv_to_reference`
   reads it as is (`--source-fps` if it is not 120).
2. **A `RobotMotion` in Python**, with whatever joint order your tool uses:

   ```python
   from pathlib import Path
   from dtmr.data.robot_motion import RobotMotion, write_reference

   motion = RobotMotion(root_pos, root_quat_wxyz, joint_pos, joint_names, fps=30.0)
   write_reference(motion, robot="unitree_g1", out_dir=Path("data/my_clip"), name="my_clip")
   ```

   `lafan1_csv_to_reference.py` and `gmr_pkl_to_reference.py` are short examples of
   exactly this.
3. **A state CSV** (root pos, quat wxyz, joints, then the velocities, no header), which
   `resample_reference.py` resamples and turns into the reference npz. The only route
   that takes your own velocities.

Whatever the route, retarget to the same model the tracker uses (`src/dtmr/models/<robot>/`):
same joint names, zero pose and link lengths. A clip aimed at another description of the
robot (other joint names, link lengths or a different zero pose) may pass the by-name
match and still track poorly. If your retargeter needs a URDF or MJCF of the target,
use the one in the model directory.

Before tracking, render the reference alone (`render_rollout --ref ... --out ...`,
without `--run`): the robot should stand on the floor, face its direction of travel and
keep every limb the right way round. A clip that starts inside the floor or hovering is
left as is; the environment lifts only its reset pose.
