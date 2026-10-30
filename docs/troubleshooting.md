# Troubleshooting

## `motion_path '...REPLACE_ME...' does not exist`

The shipped configs leave `motion_path` as a placeholder. Build a reference with an
importer (the [tutorial](tutorial.md) walks through one) and pass
`--motion_path <out-dir>/<name>_50Hz.npz`, or set `motion_path` in a
copy of the config.

## `unknown key ...` when loading a config

Config loading is strict: a key nothing reads is an error, with the closest valid name
as a suggestion. Usually a typo, or a key that moved into the robot descriptor
(`src/dtmr/robots/<robot>.yaml`). `--allow_unknown_keys` turns it into a warning.

## The importer says joints do not match

Joints are matched to the robot model by name. A clip retargeted to another robot, or
to another description of the same robot (a 23-DoF G1, different joint names), is
rejected rather than tracked with permuted limbs. Retarget to the model in
`src/dtmr/models/<robot>/`; up to two joints the source does not drive (a head) are held at
zero.

## The robot starts in the floor or falls at the first step

Retargeted clips often start a few centimetres inside the floor or above it. The
environment lifts only its reset pose (`reset_ground: true`), which is what the paper
runs did. Render the reference alone (`render_rollout --ref ...`
without `--run`) to see it.

## Out of GPU memory, or too slow to iterate

`Nsample` (rollouts per iteration) drives both. The paper recipes use 4096 on a 24 GB
RTX 4090. For quick experiments, write a small config that inherits the recipe and
lowers them; `base_config:` is resolved relative to the file that names it:

```yaml
# configs/g1_fast.yaml
base_config: ../src/dtmr/examples/unitree_g1_release_warp.yaml
Nsample: 1024
Ndiffuse: 2
Ndiffuse_init: 2
```

## No HTML replay after a run

The Brax HTML replay needs the optional extra: `pip install -e ".[viz]"`. The demos pass
`--no_viz` and write none; the mp4 renderer works without it.
