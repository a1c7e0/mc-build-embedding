"""Data health-check for a generated world (floating blocks, column drift)."""
import json
import os
import sys

import numpy as np
from PIL import Image

wd = sys.argv[1] if len(sys.argv) > 1 else "out/world/valley_port"
W = np.load(os.path.join(wd, "world.npz"))["world"]
t = np.load(os.path.join(wd, "terrain.npz"))
Ht = t["height"].astype(int)
Wx, Hy, Dz = W.shape
print(f"world {W.shape} dtype={W.dtype}")

# per-column: highest non-air y vs terrain height
occ = W != 0
col_top = np.full((Wx, Dz), -1, dtype=int)
any_col = occ.any(axis=1)
ymax = np.argmax(occ[:, ::-1, :], axis=1)  # index from top
col_top[any_col] = Hy - 1 - ymax[any_col]

drift = col_top - Ht
bad = np.nonzero((np.abs(drift) > 1) & any_col)
print(f"columns with |top-Ht|>1: {len(bad[0])}")
if len(bad[0]):
    for k in range(min(8, len(bad[0]))):
        x, z = int(bad[0][k]), int(bad[1][k])
        print(f"  ({x},{z}) Ht={Ht[x,z]} top={col_top[x,z]} "
              f"col_ids={sorted(set(W[x, :min(col_top[x,z]+6, Hy), z].tolist()))[:10]}")

# floating voxels: air below a solid inside a column
below_solid = np.zeros_like(occ)
cs = np.cumsum(occ[:, ::-1, :], axis=1)[:, ::-1, :]
floating = occ & (cs > 1) & False  # placeholder
holes = 0
for x in range(0, Wx, 4):
    col_any = occ[x]
    for z in range(0, Dz, 4):
        col = col_any[:, z]
        if not col.any():
            continue
        ys = np.nonzero(col)[0]
        holes += (ys.max() - ys.min() + 1) - len(ys)
print(f"sampling columns (stride 4): interior holes total={holes}")

# aerial image stats
img = Image.open(os.path.join(wd, "aerial.png")).convert("RGB")
a = np.asarray(img)
print(f"aerial.png {img.size} mean_rgb={a.reshape(-1,3).mean(axis=0).round(1)} "
      f"std={a.reshape(-1,3).std(axis=0).round(1)}")
white_frac = float((a.min(axis=2) > 245).mean())
print(f"near-white pixel fraction: {white_frac:.3f}")

lay = json.load(open(os.path.join(wd, "layout.json")))
print(f"layout: slots={len(lay['slots'])} trees={len(lay['trees'])} "
      f"roads={len(lay['roads'])}")
