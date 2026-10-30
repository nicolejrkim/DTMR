# Running a SEED motion

One clip from [SEED](https://huggingface.co/datasets/bones-studio/seed) (the Skeletal
Everyday Embodiment Dataset by Bones Studio, the paper's data) through the whole
pipeline: download, retarget with soma-retargeter, track with DTMR. The example is
`egypt_dance_R_003__A275` (5 s of dance) on the Unitree G1.

NVIDIA's [soma-retargeter](https://github.com/NVIDIA/soma-retargeter) covers the G1,
T1, X2 Ultra and A3; the [nicolejrkim/soma-retargeter](https://github.com/nicolejrkim/soma-retargeter)
fork adds the H1-2, R1 and K1, so this guide installs the fork and reaches all seven
robots.

<video class="lineup" src="assets/demo_seed_egypt_dance_lineup.mp4" autoplay loop muted playsinline></video>

<sub>The clip on every robot after the three steps below: the SOMA human and, left to right, G1, K1, H1-2, R1, T1, X2 Ultra and A3, each tracking it with DTMR and its release config.</sub>

## 0. Access

SEED is gated under the Bones Studio Dataset License Agreement and is not redistributed
here. The license requires the credit line **"Motion Data by Bones Studio
(https://bones.studio/)"** on publications, websites and demos that use the data.

1. Accept the license on the [dataset page](https://huggingface.co/datasets/bones-studio/seed)
   while logged in to Hugging Face.
2. Make your token available: `huggingface-cli login`, or `export HF_TOKEN=hf_...`.

## 1. Download a clip

SEED ships its BVH clips in one 43 GB archive; `seed_fetch` streams it and writes only
the clips you name. Find clips with `search` (needs `pip install -e ".[seed]"`) or the
dataset viewer on the Hugging Face page, then fetch them by move name:

```bash
python -m dtmr.data.seed_fetch search --query "egypt" --limit 4
python -m dtmr.data.seed_fetch fetch --moves egypt_dance_R_003__A275 --out data/seed_bvh
```

This writes `data/seed_bvh/<move>.bvh`, 120 fps on the SOMA skeleton soma-retargeter
expects. The archive is unsorted, so a fetch takes from seconds to a full pass over it;
name several clips in one call to share the pass. Clips already on disk are skipped.
Names ending in `_M` are mirrored takes.

## 2. Retarget with soma-retargeter

soma-retargeter needs Python 3.12, so it gets its own environment:

```bash
git clone https://github.com/nicolejrkim/soma-retargeter && cd soma-retargeter
git lfs pull && git submodule update --init --recursive
conda create -n soma-retargeter python=3.12 -y && conda activate soma-retargeter
pip install .
```

Write a config, here `seed_to_g1.json`, with absolute paths and the robot's
`retarget_target` (`unitree_g1`, `unitree_h1_2`, `unitree_r1`, `booster_t1`,
`booster_k1`, `agibot_x2ultra` or `agibot-a3t3`):

```json
{
  "import_folder": "/abs/path/to/DTMR/data/seed_bvh",
  "export_folder": "/abs/path/to/DTMR/data/soma_g1",
  "batch_size": 100,
  "retargeter": "Newton",
  "retarget_source": "soma",
  "retarget_target": "unitree_g1",
  "retarget_source_facing_direction": "Mujoco"
}
```

Run it headless from the soma-retargeter directory:

```bash
python ./app/bvh_to_csv_converter.py --config seed_to_g1.json --viewer null
```

Every BVH in `import_folder` becomes `<export_folder>/<move>.csv`: root position in
centimetres, root rotation as XYZ Euler angles in degrees, one column per joint in
degrees, at 120 fps. The first run compiles kernels and takes about a minute.

## 3. Track it with DTMR

Back in the DTMR environment, convert the CSV into the tracker's reference files and run
the robot's release config on them:

```bash
python -m dtmr.data.soma_csv_to_reference data/soma_g1/egypt_dance_R_003__A275.csv \
    --robot unitree_g1 --out-dir data/egypt_dance
# [soma] 599 frames @ 120 Hz -> 251 frames @ 50 Hz; set motion_path: data/egypt_dance/egypt_dance_R_003__A275_50Hz.npz

dtmr --config src/dtmr/examples/unitree_g1_release_warp.yaml \
    --motion_path data/egypt_dance/egypt_dance_R_003__A275_50Hz.npz \
    --output_dir output/egypt_dance --no_viz
```

The converter maps joints by name onto the robot model, converts units, adds velocities,
resamples to 50 Hz and runs forward kinematics. The run stops when the phase
reaches the end of the clip and writes `<stamp>_log.npz`, `<stamp>_dtw_metrics.yaml` and
`<stamp>_phi_tracking_metrics.yaml`; the [tutorial](tutorial.md#5-watch-the-result)
covers rendering the result and reading the scores.

<video class="lineup" src="assets/demo_seed_egypt_dance.mp4" autoplay loop muted playsinline></video>

<sub>SEED <code>egypt_dance_R_003__A275</code> tracked on the G1 with the release recipe above; the translucent robot is the reference at the tracker's phase.</sub>

## If something goes wrong

| Symptom | Cause |
|---|---|
| `Access denied (401/403)` from `seed_fetch` | the license was not accepted with the account the token belongs to |
| `NOT found: [...]` after a full pass | the move name does not exist; check it with `search` or the dataset viewer |
| converter: "N of M model joints have no CSV column" | `retarget_target` and `--robot` name different robots |
| the run never reaches the end of the clip | `n_steps` is too small for the playback rate the optimizer chose; raise `--n_steps` |
