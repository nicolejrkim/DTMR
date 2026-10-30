#!/usr/bin/env bash
# One 100STYLE clip on the Unitree G1, end to end: download the dataset once,
# retarget 10 s of a style onto the G1, track it with DTMR, render the rollout.
#
#   bash demo_scripts/demo_100style.sh               # Zombie forward walk
#   bash demo_scripts/demo_100style.sh TwoFootJump   # any of the 100 styles
#
# Every stage skips work that is already on disk, so rerun after an interruption.
# Needs the environment with dtmr installed (docs/install.md) and a GPU for tracking.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STYLE=${1:-Zombie}
python -c "import dtmr" 2>/dev/null || { echo "dtmr is not installed in this environment; see docs/install.md" >&2; exit 1; }

if [ ! -f data/100style/100STYLE/Frame_Cuts.csv ]; then
    python -m dtmr.data.bvh_100style fetch --out data/100style     # 1.4 GB, once
fi
python -m dtmr.tools.run_100style prepare --styles "$STYLE"        # retarget onto the G1 (CPU, a second)
python -m dtmr.tools.run_100style track   --styles "$STYLE"        # DTMR with the G1 release config (GPU, minutes)
python -m dtmr.tools.run_100style render  --styles "$STYLE"        # rollout.mp4 next to the run
echo "done: output/100style/unitree_g1/${STYLE}_FW/rollout.mp4"
