"""Read the 100STYLE locomotion dataset (BVH, 60 fps).

100STYLE (https://www.ianxmason.com/100style/) has 100 walking and running
styles, each in eight movement types (BR/BW/FR/FW: backward/forward run/walk,
SR/SW: sidestep run/walk, ID: idle, TR1-3: transitions), captured with Xsens on
one 28-joint skeleton. It is released under CC BY 4.0: credit "The 100STYLE
Dataset - Ian Mason" and cite Mason, Starke and Komura, "Real-Time Style
Modelling of Human Locomotion via Feature-Wise Transformations and Local Motion
Phases", ACM TOG / I3D 2022.

`fetch` downloads the BVH archive (1.4 GB) from Zenodo; `load_bvh` runs forward
kinematics and returns the motion with the bone names of `HumanMotion`,
so `keypoint_retarget.human_source` takes it unchanged. `Frame_Cuts.csv` in the
archive trims the calibration T-pose at both ends of every take; `load_style`
applies it.

    python -m dtmr.data.bvh_100style fetch --out data/100style
    python -m dtmr.data.keypoint_retarget data/100style/100STYLE/Zombie/Zombie_FW.bvh \\
        --robot unitree_g1 --out-dir data/100style_ref/unitree_g1 --start 0 --end 10
"""
from __future__ import annotations

import argparse
import csv
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np



@dataclass
class HumanMotion:
    start: dict[str, np.ndarray]   # bone -> (T, 3) start point (root: the root position)
    end: dict[str, np.ndarray]     # bone -> (T, 3) end point
    rot: dict[str, np.ndarray]     # bone -> (T, 3, 3) world rotation relative to the rest pose
    fps: float


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


ZENODO = "https://zenodo.org/api/records/8127870/files/100STYLE.zip/content"
UNIT = 0.01                       # BVH offsets are in centimetres
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
MOVES = ("BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3")
# bone -> (joint at the bone's start, joint at its end); a BVH joint sits at
# the start of the bone it rotates
BONES = {
    "root": ("Hips", "Hips"),
    "lowerback": ("Hips", "Chest"),
    "upperback": ("Chest", "Chest3"),
    "lowerneck": ("Neck", "Head"),
    **{f"{s}{bone}": (f"{S}{a}", f"{S}{b}") for s, S in (("l", "Left"), ("r", "Right"))
       for bone, a, b in (("femur", "Hip", "Knee"), ("tibia", "Knee", "Ankle"), ("foot", "Ankle", "Toe"),
                          ("toes", "Toe", "Toe"), ("humerus", "Shoulder", "Elbow"),
                          ("radius", "Elbow", "Wrist"), ("wrist", "Wrist", "Wrist"))},
}
# the bone whose rotation stands for each bone (the joint at its start)
ROT_JOINT = {b: start for b, (start, _) in BONES.items()}


def _parse(path: Path):
    """Joints (name, parent, offset, channels) in file order, frame time, frame data."""
    lines = Path(path).read_text().split("\n")
    joints, stack, i = [], [], 0
    while not lines[i].strip().startswith("MOTION"):
        tok = lines[i].split()
        if tok and tok[0] in ("ROOT", "JOINT"):
            joints.append({"name": tok[1], "parent": stack[-1] if stack else None, "channels": []})
            pending = joints[-1]["name"]
        elif tok and tok[0] == "End":
            pending = None
        elif tok and tok[0] == "{":
            stack.append(pending)
        elif tok and tok[0] == "}":
            stack.pop()
        elif tok and tok[0] == "OFFSET" and pending is not None:
            joints[-1]["offset"] = np.array([float(v) for v in tok[1:4]]) * UNIT
        elif tok and tok[0] == "CHANNELS":
            joints[-1]["channels"] = tok[2:]
        i += 1
    n = int(lines[i + 1].split()[1])
    dt = float(lines[i + 2].split()[2])
    data = np.loadtxt(lines[i + 3:i + 3 + n], ndmin=2)
    return joints, dt, data


def load_bvh(path: Path, start: int = 0, stop: int | None = None) -> HumanMotion:
    """Frames [start, stop) of a 100STYLE take as a `HumanMotion`."""
    joints, dt, data = _parse(path)
    data = data[start:stop]
    T = len(data)
    A = Y_UP_TO_Z_UP
    P, R, col = {}, {}, 0
    for j in joints:
        ch = j["channels"]
        vals = data[:, col:col + len(ch)]
        col += len(ch)
        Rl = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        for k, c in enumerate(ch):
            if c.endswith("rotation"):
                axis = c[0].lower()
                Rl = Rl @ np.stack([_rot(axis, a) for a in vals[:, k]])
        if j["parent"] is None:
            pos = vals[:, [ch.index(c) for c in ("Xposition", "Yposition", "Zposition")]] * UNIT
            P[j["name"]], R[j["name"]] = pos, Rl
        else:
            pp, Rp = P[j["parent"]], R[j["parent"]]
            P[j["name"]] = pp + Rp @ j["offset"]
            R[j["name"]] = Rp @ Rl
    z_up = lambda x: x @ A.T
    start_ = {b: z_up(P[a]) for b, (a, _) in BONES.items()}
    end = {b: z_up(P[e]) for b, (_, e) in BONES.items()}
    rot = {b: A @ R[ROT_JOINT[b]] @ A.T for b in BONES}
    return HumanMotion(start_, end, rot, 1.0 / dt)


def frame_cuts(root: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """(style, move) -> (start, stop) frames from the archive's Frame_Cuts.csv."""
    cuts = {}
    with open(Path(root) / "Frame_Cuts.csv") as f:
        for row in csv.DictReader(f):
            for m in MOVES:
                a, b = row.get(f"{m}_START"), row.get(f"{m}_STOP")
                if a and a != "N/A":
                    cuts[row["STYLE_NAME"], m] = (int(a), int(b))
    return cuts


def load_style(path: Path) -> HumanMotion:
    """A take with its calibration poses trimmed (Frame_Cuts.csv next to the style folders)."""
    path = Path(path)
    style, move = path.stem.rsplit("_", 1)
    a, b = frame_cuts(path.parent.parent).get((style, move), (0, None))
    return load_bvh(path, a, b)


def fetch(out: Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "100STYLE.zip"
    if not zpath.exists():
        print(f"[100style] downloading {ZENODO} (1.4 GB)")
        urllib.request.urlretrieve(ZENODO, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out, [n for n in z.namelist() if not n.startswith("__MACOSX")])
    print(f"[100style] {sum(1 for _ in (out / '100STYLE').rglob('*.bvh'))} takes in {out / '100STYLE'}")
    return out / "100STYLE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download and unpack the BVH archive")
    f.add_argument("--out", type=Path, default=Path("data/100style"))
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.out)


if __name__ == "__main__":
    main()
