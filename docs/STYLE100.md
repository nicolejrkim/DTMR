# Running the 100 styles

[100STYLE](https://www.ianxmason.com/100style/) (CC BY 4.0) is one actor walking and
running in 100 styles, each in up to ten movement types: 810 takes. This page runs DTMR
over all of them, or any subset, on any of the seven robots, one command per stage:

```text
fetch  ->  prepare (retarget, CPU)  ->  track (DTMR, GPU)  ->  summary  ->  render
```

Every stage skips what is already on disk, so an interrupted batch resumes with the
same command.

<video class="lineup" src="assets/style100_preview.mp4" autoplay loop muted playsinline></video>

<sub>100STYLE <code>TwoFootJump_FW</code> after <code>prepare</code>: the mocap source and, left to right, the seven retargeted references before tracking, each held in its lane.</sub>

## 1. Download the dataset

```bash
python -m dtmr.data.bvh_100style fetch --out data/100style
```

1.4 GB from Zenodo (3.2 GB unpacked), into `data/100style/100STYLE/<Style>/<Style>_<MOVE>.bvh`.
Credit "The 100STYLE Dataset - Ian Mason" and cite Mason, Starke and Komura (2022) for
anything made from it.

## 2. Choose what to run

```bash
python -m dtmr.tools.run_100style list                       # the default: 100 forward walks
python -m dtmr.tools.run_100style list --moves FW BW SW ID   # 400 takes
```

Every stage takes the same selection flags:

| Flag | Default | Meaning |
|---|---|---|
| `--robots` | `unitree_g1` | any of `unitree_g1 booster_k1 unitree_h1_2 unitree_r1 booster_t1 agibot_x2_ultra agibot_a3`, or `all` |
| `--moves` | `FW` | `FW`/`BW` forward/backward walk, `FR`/`BR` runs, `SW`/`SR` sidesteps, `ID` idle, `TR1`-`TR3` transitions (TR2/TR3 only exist for some styles) |
| `--styles` | all 100 | e.g. `--styles Zombie Drunk TwoFootJump` |
| `--limit N` | none | only the first N takes of the selection |
| `--start`, `--seconds` | 0, 10 | the window cut from each take, in seconds after the dataset's own trim of the calibration pose |

## 3. Retarget (CPU)

```bash
python -m dtmr.tools.run_100style prepare --robots unitree_g1
```

Each take goes through the bundled
[keypoint retargeter](HUMAN_MOTION.md#100style-through-the-keypoint-retargeter)
on all CPU cores (`--jobs`), about a second per clip and robot. The references land in
`data/100style_ref/<robot>/<Style>_<MOVE>_50Hz.npz`, and `manifest.json` there records
each clip's worst foot error in stance and in swing; planted feet are typically within a
few millimetres. Render one before spending GPU time
([tutorial, step 3](tutorial.md#3-look-at-it-before-tracking)):

```bash
python -m dtmr.tools.render_rollout --robot unitree_g1 \
    --ref data/100style_ref/unitree_g1/Zombie_FW_50Hz.npz --out zombie_ref.mp4
```

## 4. Track (GPU)

```bash
python -m dtmr.tools.run_100style track --robots unitree_g1
```

Clips run one after another with the robot's release recipe,
`src/dtmr/examples/<robot>_release_warp.yaml`, unchanged, so the scores compare across
clips and robots. Each run writes to `output/100style/<robot>/<Style>_<MOVE>/` with the
tracker output in `run.log`, and one line is printed per clip:

```text
[track] unitree_g1: 0 of 100 runs done, 100 to go (unitree_g1_release_warp.yaml)
  [100/100] Zombie_FW: MPJPE 3.8 cm, phase 0.999, 664 s
```

`phase` is how far through the clip the robot got (1 = the end). On one RTX 4090 a 10 s
clip takes 11 to 28 minutes (a calm walk is quicker than a jumping one), so 100 clips
on one robot take 20 to 45 hours; `--seconds 5` halves that. A batch can run unattended
in `tmux` and be restarted at any time.

## 5. Collect the scores

```bash
python -m dtmr.tools.run_100style summary --robots unitree_g1
```

This writes `output/100style/<robot>/summary.csv`, one row per run, best first, and
prints the median, best and worst MPJPE, how many runs reached the end of their clip,
and the GPU hours spent.

| Column | Meaning |
|---|---|
| `mpjpe_world_cm`, `mpjpe_relcom_cm` | mean body-position error in the world, and relative to the centre of mass |
| `orient_deg`, `joint_mae_rad` | mean body-orientation error, mean joint-angle error |
| `dtw_cost` | error after aligning the whole rollout with the whole reference in time |
| `end_phi` | phase reached; below about 0.99, the run stopped before the end of the clip |
| `steps`, `wall_s` | MPC steps and wall time |
| `stance_foot_mm` | the retargeter's worst planted-foot error for the reference |

A clip with high error and a high `stance_foot_mm` is a retargeting problem, not a
tracking one. [Tutorial, step 6](tutorial.md#6-read-the-scores) explains the metrics.

## 6. Watch the runs

```bash
python -m dtmr.tools.run_100style render --robots unitree_g1
```

This writes `rollout.mp4` into each finished run folder: the simulated robot, with the
reference as a translucent twin at the tracker's phase.

## The whole thing in one block

```bash
python -m dtmr.data.bvh_100style fetch --out data/100style
for stage in prepare track summary render; do
  python -m dtmr.tools.run_100style $stage --robots unitree_g1
done
```

`--robots all` runs every robot (seven times the GPU time); `--moves FW BW SW ID` runs
four takes per style.
