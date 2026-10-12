"""Probe 5: floating-island vertical profile (cone vs slab check)."""
import sys

import numpy as np

wd = sys.argv[1] if len(sys.argv) > 1 else "out/world/sky_archipelago"
cx = int(sys.argv[2]) if len(sys.argv) > 2 else 230
cz = int(sys.argv[3]) if len(sys.argv) > 3 else 204
W = np.load(f"{wd}/world_full.npz")["world"]

for d in (0, 16, 32, 44, 52):
    x, z = cx + d, cz
    col = W[x, :, z]
    nz = np.nonzero(col)[0]
    if len(nz):
        segs = []
        start = prev = int(nz[0])
        for v in nz[1:]:
            v = int(v)
            if v != prev + 1:
                segs.append((start, prev))
                start = v
            prev = v
        segs.append((start, prev))
        with_ids = [(a, b, int(col[a])) for a, b in segs]
        print(f"col ({x},{z}) d={d}: segments {with_ids}")
    else:
        print(f"col ({x},{z}) d={d}: EMPTY")
