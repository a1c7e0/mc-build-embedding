"""Build a building library from the rom1504 schematics tfrecords.

Emits (out_lib/ by default):
  buildings/bNNNNN.npz   uint8 grid [x, y, z] (0 = air), legacy ids
  meta.jsonl             idx, file, dims, nblocks, url, desc=""
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from schemlib import example_features, iter_tfrecord, nbt_parse, schematic_to_grid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tfrecords", nargs="+")
    ap.add_argument("--out", default="out_lib")
    ap.add_argument("--metadata", default="data/schematicsWithFinalUrl.json")
    ap.add_argument("--min-side", type=int, default=20,
                    help="min footprint max-side (standalone buildings)")
    ap.add_argument("--max-dim", type=int, default=128)
    ap.add_argument("--min-blocks", type=int, default=800)
    ap.add_argument("--max-blocks", type=int, default=900_000)
    args = ap.parse_args()

    bdir = os.path.join(args.out, "buildings")
    os.makedirs(bdir, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.jsonl")

    urls_meta = {}
    if os.path.exists(args.metadata):
        try:
            m = json.load(open(args.metadata, encoding="utf-8"))
            urls_meta = {e.get("url"): e for e in m if isinstance(e, dict)}
        except Exception as e:  # noqa: BLE001
            print(f"[lib] metadata unusable ({e})", flush=True)

    kept = skipped = bad = 0
    t0 = time.time()
    with open(meta_path, "w", encoding="utf-8") as meta_out:
        for tf_path in args.tfrecords:
            for record in iter_tfrecord(tf_path):
                try:
                    feats = example_features(record)
                    blob = feats.get("schematicData", [None])[0]
                    url = feats.get("url", [b""])[0].decode("utf-8", "replace")
                    if blob is None:
                        skipped += 1
                        continue
                    root = nbt_parse(blob)
                    grid, _pal = schematic_to_grid(root)
                except Exception:  # noqa: BLE001
                    bad += 1
                    continue

                dims = list(grid.shape)
                n = int((grid != 0).sum())
                if n < args.min_blocks or n > args.max_blocks:
                    skipped += 1
                    continue
                if max(dims) > args.max_dim or max(dims[0], dims[2]) < args.min_side:
                    skipped += 1
                    continue
                # trim to content bbox
                occ = np.nonzero(grid)
                if len(occ[0]) == 0:
                    skipped += 1
                    continue
                slices = tuple(slice(int(o.min()), int(o.max()) + 1) for o in occ)
                grid = np.ascontiguousarray(grid[slices])
                dims = list(grid.shape)
                if grid.shape[2] > grid.shape[0]:
                    grid = np.ascontiguousarray(np.rot90(grid, k=1, axes=(0, 2)))
                    dims = list(grid.shape)

                fname = f"b{kept:05d}.npz"
                np.savez_compressed(os.path.join(bdir, fname), grid=grid)
                rec = urls_meta.get(url, {})
                meta_out.write(json.dumps({
                    "idx": kept,
                    "file": f"buildings/{fname}",
                    "desc": rec.get("title") or rec.get("name") or "",
                    "dims": dims,
                    "nblocks": n,
                    "url": url,
                }, ensure_ascii=False) + "\n")
                kept += 1
                if kept % 200 == 0:
                    print(f"[lib] kept={kept} skipped={skipped} bad={bad} "
                          f"({kept / (time.time() - t0):.1f}/s)", flush=True)
            print(f"[lib] {os.path.basename(tf_path)} done: "
                  f"kept={kept} skipped={skipped} bad={bad}", flush=True)
    print(f"[lib] total kept={kept} skipped={skipped} bad={bad} "
          f"elapsed={time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
