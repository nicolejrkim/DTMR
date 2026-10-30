# Configuration

A run is one YAML file plus command-line flags:

```bash
dtmr --config src/dtmr/examples/unitree_g1_release_warp.yaml --output_dir output/my_run
```

## How a config is built

- **`base_config:`** inherits another config and overrides only the keys it sets, so a
  clip-specific config can be three lines.
- **`robot:`** names a descriptor in `src/dtmr/robots/`; the model, gains and body lists
  come from there, not from the run config.
- **`motion_path:`** is the reference npz an importer writes (`<name>_50Hz.npz`); `--motion_path` overrides
  it, so one config serves every clip.
- **`n_steps:`** is the most 20 ms control steps to run; the run stops earlier when the
  reference ends. Leave headroom: with time warping the robot may play the clip slower
  than real time (about 1.5 × clip seconds × 50).

Loading is strict: a misspelt or unread key is an error with a suggestion of what you
probably meant. `--allow_unknown_keys` turns that into a warning.

## The main knobs

| Key | Meaning |
|---|---|
| `Nsample` | rollouts sampled per annealing iteration |
| `Hsample`, `Hnode` | planning horizon in steps, and the number of spline control nodes over it |
| `Ndiffuse`, `Ndiffuse_init` | annealing iterations per MPC step, and on the first step |
| `sigma_scale`, `sigma_scale_phi` | sampling noise for the joint controls and for the phase increment |
| `temp_sample` | softmax temperature that weights the samples |
| `dt`, `timestep` | control step and physics step (s) |
| `cost_config:` | weights of the tracking cost terms (`w_orient_pure`, `w_energy`, `w_dphi`, ...) |

The release configs in `src/dtmr/examples/` are the best-known values per robot; start from
one and change a few keys.

## Command-line flags

| Flag | Effect |
|---|---|
| `--output_dir DIR` | required; all run outputs go here |
| `--motion_path NPZ` | the reference to track (the shipped configs leave `motion_path` as a placeholder) |
| `--n_steps N`, `--seed S` | override the config; `n_steps` is an upper bound, the run stops when the reference ends |
| `--no_viz` | skip the Brax HTML replay |
| `--no_dtw`, `--no_phi_tracking` | skip the post-run scores |
| `--no_ghost` | skip all three ghost overlays |
| `--no_graph` | run eagerly instead of capturing a CUDA graph (for debugging) |

`dtmr --help` lists every flag.
