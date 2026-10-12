"""Analyze skeleton metrics by vision type + find best/worst examples."""
import collections
import json
import sys

import numpy as np

lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
sk = json.load(open(f"{lib}/skeleton.json"))
vt = json.load(open(f"{lib}/vision_types.json"))
groups = collections.defaultdict(list)
for k, v in sk.items():
    t = vt.get(k, {}).get("type", "?")
    groups[t].append((v["enclosure"], v.get("interior_voxels", 0)))
print(f"{'type':<12} {'n':>5} {'enc_med':>8} {'enc>0.05':>9} {'int>200':>8}")
for t, vals in sorted(groups.items()):
    enc = np.array([a for a, _ in vals])
    iv = np.array([b for _, b in vals])
    print(f"{t:<12} {len(vals):>5} {np.median(enc):>8.3f} "
          f"{100 * (enc > 0.05).mean():>8.0f}% {100 * (iv > 200).mean():>7.0f}%")
byi = {k: v for k, v in sk.items()}
lo = sorted(byi, key=lambda k: byi[k]["enclosure"])
print("\nmost closed (rooms):", [(k, byi[k]["enclosure"],
                                 byi[k]["interior_voxels"]) for k in lo[-5:]])
print("most open:", [(k, byi[k]["enclosure"]) for k in lo[:5]])
