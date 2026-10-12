"""Probe raw voxel data of specific builds (render QA support)."""
import json
import os

import numpy as np

meta = [json.loads(l) for l in open("out/meta.jsonl")]
for i in (548, 1385, 654):
    g = np.load(os.path.join("out", meta[i]["file"]))["grid"]
    dx, dy, dz = g.shape
    y0 = g[:, 0, :]
    ytop = g[:, -1, :]
    ids, cnt = np.unique(g[g != 0], return_counts=True)
    top = sorted(zip(cnt.tolist(), ids.tolist()), reverse=True)[:6]
    print(f"idx={i} dims={g.shape}")
    print(f"  y=0 layer: fill={float((y0 != 0).mean()):.2f} "
          f"ids={sorted(set(y0[y0 != 0].tolist()))[:8]}")
    print(f"  y=max layer: fill={float((ytop != 0).mean()):.2f}")
    print(f"  top ids (count, id): {top}")
