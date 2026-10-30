#!/usr/bin/env bash
# The tutorial (docs/tutorial.md) as one command: a LAFAN1 walk that Unitree retargeted
# to the G1, imported, tracked with DTMR and rendered next to its reference.
#
#   bash demo_scripts/demo_lafan1.sh
#
# Every stage skips work that is already on disk, so rerun after an interruption.
# Needs the environment with dtmr installed (docs/install.md) and a GPU for tracking.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CLIP=walk1_subject1
CSV=input/lafan1/$CLIP.csv
REF=data/lafan1_walk1/${CLIP}_50Hz.npz
OUT=output/lafan1_walk1
python -c "import dtmr" 2>/dev/null || { echo "dtmr is not installed in this environment; see docs/install.md" >&2; exit 1; }
mkdir -p input/lafan1 "$OUT"

if [ ! -f "$CSV" ]; then                                           # Unitree's LAFAN1 release, one clip
    curl -L -o "$CSV" "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/$CLIP.csv"
fi
if [ ! -f "$REF" ]; then                                           # seconds 10 to 20, resampled to 50 Hz
    python -m dtmr.data.lafan1_csv_to_reference "$CSV" --robot unitree_g1 --out-dir data/lafan1_walk1 --start 10 --end 20
fi
if [ ! -f "$OUT/reference.mp4" ]; then                             # the reference on its own
    python -m dtmr.tools.render_rollout --robot unitree_g1 --ref "$REF" --out "$OUT/reference.mp4"
fi
if ! ls "$OUT"/*_log.npz >/dev/null 2>&1; then                     # track the whole clip (GPU, tens of minutes)
    dtmr --config src/dtmr/examples/unitree_g1_release_warp.yaml --motion_path "$REF" --output_dir "$OUT" --no_viz
fi
python -m dtmr.tools.render_rollout --run "$OUT" --robot unitree_g1 --ref "$REF" --out "$OUT/rollout.mp4"
echo "done: $OUT/rollout.mp4, scores in $OUT/*_metrics.yaml"
