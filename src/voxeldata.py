"""Parse the minecraftbuildings dataset CSV into per-building voxel grids.

CSV format: "Description","Block Info"
- Block Info is a python-literal list of (x, y, z, block_id) tuples.
- block_id is the legacy numeric Minecraft block id (0-255), no data values.

Emits:
  out/buildings/bNNNNN.npz  dense uint8 voxel grid 'grid' indexed [x, y, z]
  out/meta.jsonl            one JSON record per kept building
  out/id_hist.json          block id histogram (sanity check of the id space)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time

import numpy as np

TUPLE_RE = re.compile(r"\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)")
csv.field_size_limit(min(__import__("sys").maxsize, 2**31 - 1))


def parse_blocks(text: str) -> np.ndarray | None:
    """Fast parse of the (x, y, z, id) tuple list into an (N, 4) int32 array."""
    m = TUPLE_RE.findall(text)
    if not m:
        return None
    return np.asarray(m, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/dataset.csv")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-dim", type=int, default=128,
                    help="skip buildings whose bbox exceeds this on any axis")
    ap.add_argument("--max-blocks", type=int, default=400_000)
    ap.add_argument("--min-blocks", type=int, default=300,
                    help="skip tiny builds (flowers, single items)")
    ap.add_argument("--max-buildings", type=int, default=3000)
    ap.add_argument("--max-voxels", type=int, default=3_000_000,
                    help="skip buildings with a larger dense bbox volume")
    args = ap.parse_args()

    bdir = os.path.join(args.out, "buildings")
    os.makedirs(bdir, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.jsonl")
    hist = np.zeros(256, dtype=np.int64)

    kept = skipped = 0
    t0 = time.time()
    with open(args.csv, newline="", encoding="utf-8") as fh, \
            open(meta_path, "w", encoding="utf-8") as meta:
        reader = csv.reader(fh)
        header = next(reader, None)
        print(f"[parse] header={header}", flush=True)
        for row_idx, row in enumerate(reader):
            if kept >= args.max_buildings:
                break
            if len(row) < 2:
                skipped += 1
                continue
            desc, blocks_text = row[0], row[1]
            arr = parse_blocks(blocks_text)
            if arr is None:
                skipped += 1
                continue
            n = arr.shape[0]
            if n < args.min_blocks or n > args.max_blocks:
                skipped += 1
                continue
            ids = arr[:, 3]
            if ids.min() < 0 or ids.max() > 255:
                # not the legacy id space we assumed; clip and count
                hist[0] += 1
                ids = np.clip(ids, 0, 255)
            xyz = arr[:, :3]
            lo = xyz.min(axis=0)
            hi = xyz.max(axis=0)
            dims = (hi - lo + 1).tolist()
            if max(dims) > args.max_dim or int(np.prod(dims)) > args.max_voxels:
                skipped += 1
                continue
            grid = np.zeros(tuple(dims), dtype=np.uint8)
            rel = xyz - lo
            grid[rel[:, 0], rel[:, 1], rel[:, 2]] = ids.astype(np.uint8)
            # canonical yaw: rotate so the long horizontal axis is x
            if grid.shape[2] > grid.shape[0]:
                grid = np.ascontiguousarray(np.rot90(grid, k=1, axes=(0, 2)))
                dims = list(grid.shape)
            hist += np.bincount(ids.astype(np.int64), minlength=256)

            fname = f"b{kept:05d}.npz"
            np.savez_compressed(os.path.join(bdir, fname), grid=grid)
            meta.write(json.dumps({
                "idx": kept,
                "file": f"buildings/{fname}",
                "desc": desc.strip()[:600],
                "dims": dims,
                "nblocks": int(n),
            }, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 50 == 0:
                rate = kept / (time.time() - t0 + 1e-9)
                print(f"[parse] kept={kept} skipped={skipped} "
                      f"({rate:.1f} buildings/s)", flush=True)

    np.save(os.path.join(args.out, "id_hist.npy"), hist)
    top = np.argsort(hist)[::-1][:20]
    with open(os.path.join(args.out, "id_hist.json"), "w") as fh:
        json.dump({int(i): int(hist[i]) for i in top}, fh, indent=2)
    print(f"[parse] done kept={kept} skipped={skipped} "
          f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"[parse] top block ids: "
          f"{ {int(i): int(hist[i]) for i in top[:10]} }", flush=True)


if __name__ == "__main__":
    main()
