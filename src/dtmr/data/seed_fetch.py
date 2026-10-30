"""Fetch individual motion clips from SEED, the Skeletal Everyday Embodiment
Dataset by Bones Studio.

SEED (https://huggingface.co/datasets/bones-studio/seed) is gated: accept its
license on the dataset page with your Hugging Face account, then either run
`huggingface-cli login` or export HF_TOKEN. Nothing from SEED is redistributed
with this repository. Motion Data by Bones Studio (https://bones.studio/): the
license requires that credit on anything published from the clips.

The BVH clips live in one 43 GB archive (`soma_uniform.tar.gz`). This tool
streams the archive over HTTP and writes only the clips you ask for, stopping as
soon as all of them are found, so the archive never touches your disk.

  # 1. find clips (needs `pip install pandas pyarrow`; or browse the dataset viewer)
  python -m dtmr.data.seed_fetch search --query jump --limit 20

  # 2. download them as BVH
  python -m dtmr.data.seed_fetch fetch \\
      --moves high_jump_R_001__A442 jog_ff_loop_180_R_slow_002__A447 --out data/seed_bvh

Clips are 120 fps on the SOMA "uniform" skeleton, which is what soma-retargeter
expects. The archive is not sorted: a clip can sit anywhere in the stream, so a
fetch may read from a few MB up to the whole archive.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO = "bones-studio/seed"
ARCHIVES = {"soma_uniform": "soma_uniform.tar.gz", "soma_proportional": "soma_proportional.tar.gz"}
METADATA = "metadata/seed_metadata_v004.parquet"
SEED_FPS = 120


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        path = Path("~/.cache/huggingface/token").expanduser()
        if path.exists():
            token = path.read_text().strip()
    if not token:
        sys.exit("No Hugging Face token found: run `huggingface-cli login` or export HF_TOKEN "
                 "(and accept the SEED license on the dataset page first).")
    return token


def _open(fname: str):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{fname}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    try:
        return urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            sys.exit(f"Access denied ({e.code}) for {fname}: accept the SEED license at "
                     f"https://huggingface.co/datasets/{REPO} with the account your token belongs to.")
        raise


class _Counting:
    def __init__(self, raw):
        self.raw, self.n = raw, 0

    def read(self, size=-1):
        chunk = self.raw.read(size)
        self.n += len(chunk)
        return chunk


def fetch(moves: list[str], out_dir: Path, variant: str, max_gb: float | None) -> int:
    wanted = {m[:-4] if m.endswith(".bvh") else m for m in moves}
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {m for m in wanted if (out_dir / f"{m}.bvh").exists()}
    for m in sorted(have):
        print(f"[seed] already present: {m}.bvh")
    todo = wanted - have
    if not todo:
        return 0
    stream = _Counting(_open(ARCHIVES[variant]))
    tar = tarfile.open(fileobj=stream, mode="r|gz")
    t0, last_gb, found = time.time(), 0.0, 0
    for member in tar:
        gb = stream.n / 1e9
        if gb - last_gb >= 1.0:
            last_gb = gb
            print(f"[seed] {gb:.0f} GB streamed, {found}/{len(todo)} found "
                  f"({stream.n / (time.time() - t0) / 1e6:.0f} MB/s)", file=sys.stderr)
        if max_gb and gb > max_gb:
            break
        if not member.isfile() or not member.name.endswith(".bvh"):
            continue
        stem = Path(member.name).stem
        if stem not in todo:
            continue
        (out_dir / f"{stem}.bvh").write_bytes(tar.extractfile(member).read())
        found += 1
        print(f"[seed] {found}/{len(todo)} {member.name} -> {out_dir / (stem + '.bvh')}")
        if found == len(todo):
            break
    missing = sorted(m for m in todo if not (out_dir / f"{m}.bvh").exists())
    print(f"[seed] done: {found}/{len(todo)} fetched, {stream.n / 1e9:.1f} GB streamed, "
          f"{time.time() - t0:.0f} s")
    if missing:
        print(f"[seed] NOT found: {missing} (check the move names against the metadata)", file=sys.stderr)
    return len(missing)


def search(query: str, limit: int, cache: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        sys.exit("`search` needs pandas + pyarrow (`pip install pandas pyarrow`), or browse "
                 f"https://huggingface.co/datasets/{REPO}/viewer instead.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"[seed] downloading metadata -> {cache}")
        cache.write_bytes(_open(METADATA).read())
    try:
        meta = pd.read_parquet(cache)
    except ImportError:
        sys.exit("`search` needs a parquet engine: `pip install pyarrow`.")
    text_cols = [c for c in meta.columns if c in ("move_name", "category", "package")
                 or c.startswith("content_")]
    hay = meta[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
    hits = meta[hay.str.contains(query.lower(), regex=False)]
    print(f"[seed] {len(hits)} of {len(meta)} moves match {query!r}; showing {min(limit, len(hits))}")
    for _, row in hits.head(limit).iterrows():
        desc = str(row.get("content_short_description", "") or row.get("content_natural_desc_1", ""))
        print(f"  {row['move_name']:<52} {row['move_duration_frames'] / SEED_FPS:6.1f} s  "
              f"{str(row.get('category', '')):<14} {desc[:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="stream-extract clips by move name")
    f.add_argument("--moves", nargs="+", required=True, help="move names (the BVH file stems)")
    f.add_argument("--out", type=Path, required=True, help="directory to write <move>.bvh into")
    f.add_argument("--variant", choices=sorted(ARCHIVES), default="soma_uniform",
                   help="skeleton variant; soma-retargeter expects soma_uniform (default)")
    f.add_argument("--max-gb", type=float, default=None, help="give up after streaming this much")
    s = sub.add_parser("search", help="search the metadata for move names")
    s.add_argument("--query", required=True, help="substring to look for in names and descriptions")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--cache", type=Path, default=Path("~/.cache/dtmr/seed_metadata_v004.parquet").expanduser())
    args = parser.parse_args()
    if args.cmd == "fetch":
        sys.exit(1 if fetch(args.moves, args.out, args.variant, args.max_gb) else 0)
    search(args.query, args.limit, args.cache)


if __name__ == "__main__":
    main()
