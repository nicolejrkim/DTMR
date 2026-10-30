# Installation

## Requirements

- Linux with an NVIDIA GPU and a CUDA 12 driver: the rollouts run in
  [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp). Tested on Ubuntu
  22.04 with an RTX 4090.
- Python 3.10.

## Install

```bash
conda create -n dtmr python=3.10 -y
conda activate dtmr
git clone https://github.com/nicolejrkim/DTMR.git
cd DTMR
pip install -e .
```

This installs the `dtmr` command and the `dtmr` Python package.

## Optional extras

| Extra | Adds |
|---|---|
| `pip install -e ".[seed]"` | the SEED catalogue search used by `seed_fetch` |
| `pip install -e ".[viz]"` | the optional Brax HTML replay written after a run |
