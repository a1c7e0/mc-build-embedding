"""Probe 4: isolated-pillar detection + rotated re-render disambiguation."""
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402

wd = sys.argv[1] if len(sys.argv) > 1 else "out/world/valley_port"
W = np.load(os.path.join(wd, "world.npz"))["world"]
t = np.load(os.path.join(wd, "terrain.npz"))
Ht = t["height"].astype(int)

# isolated pillar: column top >= 2 above every 8-neighbour top
nb = maximum_filter(Ht, size=3, mode="nearest")
mask = (Ht - nb) >= 2
n = int(mask.sum())
print(f"isolated pillar columns (top >=4 above all neighbours): {n}")
ys, zs = np.nonzero(mask)
for k in range(min(12, n)):
    x, z = int(ys[k]), int(zs[k])
    print(f"  ({x},{z}) top={Ht[x,z]} neighbours={Ht[max(x-1,0):x+2, max(z-1,0):z+2].tolist()}")

# region crop stats
x0, x1, z0, z1 = 90, 210, 205, 325
sub_mask = mask[x0:x1, z0:z1]
print(f"crop region ({x0}:{x1},{z0}:{z1}) pillar columns: {int(sub_mask.sum())}")

# rotated re-render of the same crop
sub = W[x0:x1, :, z0:z1]
occ = sub != 0
yz = np.nonzero(occ)[1]
y0, y1 = max(int(yz.min()) - 4, 0), min(int(yz.max()) + 5, W.shape[1])
sub = sub[:, y0:y1, :]
render(sub, max_px=760).save(os.path.join(wd, "qa_rot0.png"))
render(np.ascontiguousarray(np.rot90(sub, k=1, axes=(0, 2))),
       max_px=760).save(os.path.join(wd, "qa_rot90.png"))
print("saved qa_rot0.png / qa_rot90.png")
