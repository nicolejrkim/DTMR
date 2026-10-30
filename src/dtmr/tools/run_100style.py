"""Run DTMR over the 100STYLE dataset: retarget, track, score, render.

100STYLE (https://www.ianxmason.com/100style/, CC BY 4.0) has 100 locomotion
styles, each in up to ten movement types (FW/BW forward/backward walk, FR/BR
runs, SW/SR sidesteps, ID idle, TR1-3 transitions). This tool takes a window of
each chosen take through the bundled keypoint retargeter onto any of the seven
robots, tracks it with the robot's release config, and collects the scores in
one table. Every stage skips work that is already on disk, so an interrupted
batch resumes where it stopped.

    python -m dtmr.data.bvh_100style fetch --out data/100style    # once, 1.4 GB
    python -m dtmr.tools.run_100style prepare --robots unitree_g1                # retarget, CPU
    python -m dtmr.tools.run_100style track   --robots unitree_g1                # DTMR, GPU
    python -m dtmr.tools.run_100style summary --robots unitree_g1                # scores table
    python -m dtmr.tools.run_100style render  --robots unitree_g1                # rollout videos

Default selection: every style, the forward walk (FW), seconds 0-10 of the
trimmed take. Narrow it with --styles / --moves / --seconds, or try a few with
--limit. Layout:

    data/100style_ref/<robot>/<Style>_<MOVE>_50Hz.npz     references (+ manifest.json)
    output/100style/<robot>/<Style>_<MOVE>/                one DTMR run (+ run.log, rollout.mp4)
    output/100style/<robot>/summary.csv                    one row per run
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from dtmr.utils import get_example_path

ROBOTS = ("unitree_g1", "booster_k1", "unitree_h1_2", "unitree_r1", "booster_t1", "agibot_x2_ultra", "agibot_a3")
MOVES = ("FW", "BW", "FR", "BR", "SW", "SR", "ID", "TR1", "TR2", "TR3")


def takes(args) -> list[tuple[str, str, Path]]:
    """(style, move, bvh) for the selection, in alphabetical style order."""
    root = Path(args.data)
    if not (root / "Frame_Cuts.csv").exists():
        sys.exit(f"no 100STYLE data in {root}; download it with\n"
                 f"  python -m dtmr.data.bvh_100style fetch --out {root.parent}")
    styles = sorted(p.name for p in root.iterdir() if p.is_dir())
    if args.styles:
        unknown = sorted(set(args.styles) - set(styles))
        if unknown:
            sys.exit(f"unknown style(s) {unknown}; `list` prints the 100 names")
        styles = [s for s in styles if s in args.styles]
    out = [(s, m, root / s / f"{s}_{m}.bvh") for s in styles for m in args.moves
           if (root / s / f"{s}_{m}.bvh").exists()]
    return out[:args.limit] if args.limit else out


def ref_npz(args, robot, style, move) -> Path:
    return Path(args.refs) / robot / f"{style}_{move}_50Hz.npz"


def run_dir(args, robot, style, move) -> Path:
    return Path(args.out) / robot / f"{style}_{move}"


def is_done(d: Path) -> bool:
    return bool(glob.glob(str(d / "*_phi_tracking_metrics.yaml")))


# --------------------------------------------------------------------------- list
def cmd_list(args) -> None:
    sel = takes(args)
    with open(Path(args.data) / "Dataset_List.csv") as f:
        desc = {r["Style Name"]: r["Description"] for r in csv.DictReader(f)}
    for style, move, _ in sel:
        print(f"{style + '_' + move:28s} {desc.get(style, '')}")
    print(f"{len(sel)} takes")


# --------------------------------------------------------------------------- prepare
FOOT = re.compile(r"worst foot error ([\d.]+) mm in stance, ([\d.]+) mm in swing")


def _retarget(args, robot, style, move, bvh) -> tuple[str, dict | None]:
    out = Path(args.refs) / robot
    cmd = [sys.executable, "-m", "dtmr.data.keypoint_retarget", str(bvh),
           "--robot", robot, "--out-dir", str(out), "--name", f"{style}_{move}",
           "--start", str(args.start), "--end", str(args.start + args.seconds)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    m = FOOT.search(p.stdout)
    if p.returncode or not m:
        return f"{style}_{move}", {"error": (p.stderr or p.stdout).strip().splitlines()[-1:]}
    return f"{style}_{move}", {"stance_foot_mm": float(m.group(1)), "swing_foot_mm": float(m.group(2)),
                               "start_s": args.start, "seconds": args.seconds}


def cmd_prepare(args) -> None:
    sel = takes(args)
    for robot in args.robots:
        todo = [t for t in sel if args.force or not ref_npz(args, robot, t[0], t[1]).exists()]
        print(f"[prepare] {robot}: {len(sel) - len(todo)} of {len(sel)} references exist, "
              f"retargeting {len(todo)} on {args.jobs} CPU workers")
        manifest_path = Path(args.refs) / robot / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        t0 = time.time()
        with ThreadPoolExecutor(args.jobs) as pool:
            for n, (name, info) in enumerate(pool.map(lambda t: _retarget(args, robot, *t), todo), 1):
                manifest[name] = info
                tag = (f"stance {info['stance_foot_mm']:.1f} mm, swing {info['swing_foot_mm']:.1f} mm"
                       if "error" not in info else f"FAILED {info['error']}")
                print(f"  [{n}/{len(todo)}] {name}: {tag}", flush=True)
        if todo:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            print(f"[prepare] {robot}: done in {time.time() - t0:.0f} s, {manifest_path}")


# --------------------------------------------------------------------------- track
def _config(robot) -> Path:
    """Every run uses the robot's release recipe unchanged, so scores compare across clips and robots."""
    return Path(get_example_path(f"{robot}_release_warp.yaml"))


def cmd_track(args) -> None:
    sel = takes(args)
    for robot in args.robots:
        cfg = _config(robot)
        ready = [t for t in sel if ref_npz(args, robot, t[0], t[1]).exists()]
        if len(ready) < len(sel):
            print(f"[track] {robot}: {len(sel) - len(ready)} references missing; run `prepare` first")
        todo = [t for t in ready if args.force or not is_done(run_dir(args, robot, t[0], t[1]))]
        print(f"[track] {robot}: {len(ready) - len(todo)} of {len(ready)} runs done, {len(todo)} to go "
              f"({cfg.name})")
        for n, (style, move, _) in enumerate(todo, 1):
            npz = ref_npz(args, robot, style, move)
            frames = _frames(npz)
            d = run_dir(args, robot, style, move)
            d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "dtmr.tracker", "--config", str(cfg), "--motion_path", str(npz),
                   "--output_dir", str(d), "--no_viz", "--no_ghost",
                   "--n_steps", str(math.ceil(args.headroom * frames))]
            t0 = time.time()
            with open(d / "run.log", "w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            row = _row(args, robot, style, move)
            msg = (f"MPJPE {row['mpjpe_world_cm']:.1f} cm, phase {row['end_phi']:.3f}"
                   if row.get("mpjpe_world_cm") is not None else f"FAILED (exit {rc}), see {d / 'run.log'}")
            print(f"  [{n}/{len(todo)}] {style}_{move}: {msg}, {time.time() - t0:.0f} s", flush=True)


def _frames(npz: Path) -> int:
    import numpy as np
    return int(np.load(npz)["timestamps"].shape[0])


# --------------------------------------------------------------------------- summary
END = re.compile(r"episode terminated at step (\d+) \(phi=([\d.]+)")
ELAPSED = re.compile(r"elapsed: ([\d.]+)s for (\d+) steps")


def _row(args, robot, style, move) -> dict:
    d = run_dir(args, robot, style, move)
    row = {"robot": robot, "clip": f"{style}_{move}", "style": style, "move": move}
    man = Path(args.refs) / robot / "manifest.json"
    info = json.loads(man.read_text()).get(f"{style}_{move}", {}) if man.exists() else {}
    row["stance_foot_mm"] = info.get("stance_foot_mm")
    phi = sorted(glob.glob(str(d / "*_phi_tracking_metrics.yaml")))
    dtw = sorted(glob.glob(str(d / "*_dtw_metrics.yaml")))
    if not phi:
        return row
    m = yaml.safe_load(open(phi[-1]))
    row.update(mpjpe_world_cm=100 * m["phi/mpjpe_world_m"], mpjpe_relcom_cm=100 * m["phi/mpjpe_relcom_m"],
               orient_deg=m["phi/mean_orient_err_deg"], joint_mae_rad=m["phi/qpos_joint_mae_rad"])
    if dtw:
        row["dtw_cost"] = yaml.safe_load(open(dtw[-1])).get("mean_cost")
    text = (d / "run.log").read_text(errors="ignore") if (d / "run.log").exists() else ""
    if (e := END.findall(text)):
        row["steps"], row["end_phi"] = int(e[-1][0]) + 1, float(e[-1][1])
    if (e := ELAPSED.findall(text)):
        row["wall_s"] = float(e[-1][0])
    row.setdefault("end_phi", float("nan"))
    return row


FIELDS = ("robot", "clip", "style", "move", "mpjpe_world_cm", "mpjpe_relcom_cm", "orient_deg", "joint_mae_rad",
          "dtw_cost", "end_phi", "steps", "wall_s", "stance_foot_mm")


def cmd_summary(args) -> None:
    sel = takes(args)
    for robot in args.robots:
        rows = [_row(args, robot, s, m) for s, m, _ in sel]
        done = [r for r in rows if r.get("mpjpe_world_cm") is not None]
        path = Path(args.out) / robot / "summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(done, key=lambda r: r["mpjpe_world_cm"]))
        print(f"[summary] {robot}: {len(done)} of {len(rows)} runs scored -> {path}")
        if not done:
            continue
        err = sorted(r["mpjpe_world_cm"] for r in done)
        unfinished = [r["clip"] for r in done if not r["end_phi"] >= 0.99]
        hours = sum(r.get("wall_s", 0) for r in done) / 3600
        print(f"  MPJPE (cm): median {err[len(err) // 2]:.1f}, best {err[0]:.1f}, worst {err[-1]:.1f}; "
              f"{len(done) - len(unfinished)} reached the end of the clip; {hours:.1f} GPU h")
        print("  worst five: " + ", ".join(f"{r['clip']} {r['mpjpe_world_cm']:.1f}"
                                            for r in sorted(done, key=lambda r: -r["mpjpe_world_cm"])[:5]))
        if unfinished:
            print(f"  stopped before the end: {', '.join(unfinished)}")


# --------------------------------------------------------------------------- render
def cmd_render(args) -> None:
    sel = takes(args)
    for robot in args.robots:
        todo = [(s, m) for s, m, _ in sel if is_done(run_dir(args, robot, s, m))
                and (args.force or not (run_dir(args, robot, s, m) / "rollout.mp4").exists())]
        print(f"[render] {robot}: {len(todo)} videos to render")
        for n, (style, move) in enumerate(todo, 1):
            d = run_dir(args, robot, style, move)
            subprocess.run([sys.executable, "-m", "dtmr.tools.render_rollout", "--run", str(d), "--robot", robot,
                            "--ref", str(ref_npz(args, robot, style, move)), "--out", str(d / "rollout.mp4")],
                           capture_output=True)
            print(f"  [{n}/{len(todo)}] {d / 'rollout.mp4'}", flush=True)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--robots", nargs="+", default=["unitree_g1"], choices=ROBOTS + ("all",),
                        help="robots to run (default unitree_g1; 'all' = the seven)")
    common.add_argument("--styles", nargs="+", default=None, help="style names (default: all 100)")
    common.add_argument("--moves", nargs="+", default=["FW"], choices=MOVES, help="movement types (default FW)")
    common.add_argument("--limit", type=int, default=None, help="only the first N takes of the selection")
    common.add_argument("--start", type=float, default=0.0, help="seconds into the trimmed take")
    common.add_argument("--seconds", type=float, default=10.0, help="window length (default 10 s)")
    common.add_argument("--data", default="data/100style/100STYLE", help="the unpacked archive")
    common.add_argument("--refs", default="data/100style_ref", help="where references go")
    common.add_argument("--out", default="output/100style", help="where runs go")
    common.add_argument("--force", action="store_true", help="redo work that already exists")
    sub.add_parser("list", parents=[common], help="print the selected takes")
    p = sub.add_parser("prepare", parents=[common], help="retarget the selection (CPU, parallel)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p = sub.add_parser("track", parents=[common], help="track every prepared reference with DTMR (GPU, one at a time)")
    p.add_argument("--headroom", type=float, default=1.6, help="n_steps = headroom x reference frames")
    sub.add_parser("summary", parents=[common], help="collect scores into summary.csv")
    sub.add_parser("render", parents=[common], help="render rollout.mp4 for every finished run")
    args = ap.parse_args()
    if "all" in args.robots:
        args.robots = list(ROBOTS)
    {"list": cmd_list, "prepare": cmd_prepare, "track": cmd_track,
     "summary": cmd_summary, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
