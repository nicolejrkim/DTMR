# Quickstart

## First run: a 100STYLE clip

[100STYLE](https://www.ianxmason.com/100style/) is a freely licensed (CC BY 4.0)
motion-capture dataset of one actor walking and running in 100 styles: zombie,
aeroplane, drunk, high knees, two-foot jumps, robot, tiptoe, and so on. DTMR ships a
reader for it and a keypoint retargeter onto all seven robots, so one download gives
hundreds of human clips. The batch tool takes a take through retargeting, tracking and
rendering, one command per stage:

```bash
python -m dtmr.data.bvh_100style fetch --out data/100style   # 1.4 GB, once
python -m dtmr.tools.run_100style prepare --styles Zombie                    # retarget 10 s onto the G1 (CPU, a second)
python -m dtmr.tools.run_100style track   --styles Zombie                    # DTMR with the G1 release recipe (GPU)
python -m dtmr.tools.run_100style render  --styles Zombie                    # rollout.mp4
```

`bash demo_scripts/demo_100style.sh [Style]` runs the four stages as one command.
Without `--robots` the tool takes the G1, without `--moves` the forward walk, and
`--seconds` defaults to the first 10 s of the take. Every stage skips what is already on
disk, so a stopped command carries on when run again. The run lands in
`output/100style/unitree_g1/Zombie_FW/`: the tracker output in `run.log`, the metrics
files, and after `render` a `rollout.mp4` with the reference drawn as a translucent twin
at the tracker's phase.

Styles to try first (`--styles` takes several):

| Style | Motion |
|---|---|
| `Zombie` | arms held out in front, a shuffling walk |
| `TwoFootJump` | feet together, jumping forward |
| `HighKnees` | knees lifted high at every step |
| `Drunk` | stumbling, with random step sizes |
| `Aeroplane` | arms out as wings, tilting from side to side |
| `Skip` | skipping like a child |

<table class="gallery">
  <tr>
    <td><video src="assets/demo_100style_zombie.mp4" autoplay loop muted playsinline></video><br>
        <sub><code>Zombie_FW</code></sub></td>
    <td><video src="assets/demo_100style_twofootjump.mp4" autoplay loop muted playsinline></video><br>
        <sub><code>TwoFootJump_FW</code></sub></td>
  </tr>
</table>

Two takes tracked on the G1 with the release recipe, rendered by `render`. The solid
robot is the simulation, the translucent blue one the reference at the tracker's phase.
If you use the clips or anything made from them, credit "The 100STYLE Dataset - Ian
Mason" and cite Mason, Starke and Komura (2022). [Running the 100 styles](STYLE100.md)
covers the whole dataset: every movement type, every robot, and the scores table.

## How long a run takes

Measured on one RTX 4090 with the G1 release recipe (4096 samples per iteration), on
the 10 s window of each take:

| Take | Steps run | Wall time |
|---|---|---|
| `Zombie_FW` | 435 | 11 min |
| `TwoFootJump_FW` | 536 | 28 min |

The recipe searches deeper on fast-changing frames, so a calm walk is quicker than a
jumping one; plan on **1 to 3 minutes per second of robot motion**. Both windows have
500 reference frames at 50 Hz, and neither run took 500 steps: DTMR got through the
zombie walk in 435 steps (8.7 s of robot time) and slowed the jumps where the G1 could
not keep up (536 steps, 10.7 s). That is the time warping at work. `--seconds 5`
halves the time.

## Outputs

Every run writes into `--output_dir`:

| File | Contents |
|---|---|
| `<stamp>_log.npz` | state, action, phase and reward per step |
| `<stamp>_dtw_metrics.yaml` | DTW-aligned tracking scores |
| `<stamp>_phi_tracking_metrics.yaml` | phase-aligned joint and link errors |
| `*.html` | Brax replay and ghost overlays (skipped with `--no_viz`; needs the `viz` extra) |

`render_rollout` turns a run into an mp4 (headless MuJoCo, EGL) with the reference drawn
as a translucent twin at the tracker's phase, so the video shows where the time warping
slowed down or sped up; `--speed 0.5` gives slow motion:

```bash
python -m dtmr.tools.render_rollout --run output/100style/unitree_g1/Zombie_FW --robot unitree_g1 \
    --ref data/100style_ref/unitree_g1/Zombie_FW_50Hz.npz --out zombie.mp4
```

## Your own clip

Any retargeted robot motion runs with the robot's release config and `--motion_path`:

```bash
dtmr --config src/dtmr/examples/unitree_g1_release_warp.yaml \
    --motion_path data/<clip>/<clip>_50Hz.npz --output_dir output/<clip> --no_viz
```

The [tutorial](tutorial.md) builds that reference from a download and reads the scores;
[Reference motions](HUMAN_MOTION.md) covers SEED through soma-retargeter, LAFAN1, GMR,
OmniRetarget, PHUMA, 100STYLE and your own retargeter.
