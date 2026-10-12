"""Render skeleton-metric examples: sealed box, mid house, broken shell."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402

lib = "out_lib"
sk = json.load(open(f"{lib}/skeleton.json"))
vt = json.load(open(f"{lib}/vision_types.json"))
OUT = os.path.expanduser("~/.hermes/cache/scratch/skeleton_examples")
os.makedirs(OUT, exist_ok=True)

buildings = {k: v for k, v in sk.items()
             if vt.get(k, {}).get("type") == "building"}
by_enc = sorted(buildings, key=lambda k: buildings[k]["enclosure"])
picks = [("sealed", by_enc[-1]),
         ("rooms", by_enc[int(len(by_enc) * 0.75)]),
         ("mid", by_enc[int(len(by_enc) * 0.4)]),
         ("broken", by_enc[0])]
meta = {r["idx"]: r for r in (json.loads(l) for l in
                              open(f"{lib}/meta.jsonl", encoding="utf-8"))}
for tag, idx in picks:
    rec = meta[int(idx)]
    g = np.load(os.path.join(lib, rec["file"]))["grid"]
    render2(g, max_px=340).save(f"{OUT}/{tag}_{idx}.png")
    print(f"[ex] {tag} idx={idx} enc={buildings[idx]['enclosure']} "
          f"int={buildings[idx]['interior_voxels']}", flush=True)
