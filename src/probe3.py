"""Probe 3: heightmap spike detection + aerial rgb diagnostics."""
import os
import sys

import numpy as np
from scipy.ndimage import median_filter

wd = sys.argv[1] if len(sys.argv) > 1 else "out/world/valley_port"
t = np.load(os.path.join(wd, "terrain.npz"))
H = t["height"].astype(float)
biome = t["biome"]
print(f"H: min={H.min()} max={H.max()} mean={H.mean():.1f} nan={np.isnan(H).sum()}")
print(f"biome unique={np.unique(biome, return_counts=True)}")

med = median_filter(H, size=9)
spike = H - med
n_sp = int((spike > 5).sum())
print(f"spike cells (H - median9 > 5): {n_sp}")
if n_sp:
    idx = np.nonzero(spike > 5)
    for k in range(min(10, len(idx[0]))):
        x, z = int(idx[0][k]), int(idx[1][k])
        print(f"  ({x},{z}) H={H[x,z]:.0f} med={med[x,z]:.0f} spike={spike[x,z]:.0f}")

gy, gx = np.gradient(H)
shade = np.clip(0.75 + (gx * 0.7 - gy * 0.7) / 6.0, 0.35, 1.25)
rgb = np.zeros(H.shape + (3,))
rgb[biome == 0] = (0.25, 0.45, 0.75)
rgb[biome == 1] = (0.86, 0.80, 0.58)
rgb[biome == 2] = (0.52, 0.72, 0.42)
rgb[biome == 3] = (0.28, 0.52, 0.30)
rgb[biome == 4] = (0.62, 0.60, 0.58)
print(f"rgb after biome paint: mean={rgb.mean():.3f} nan={np.isnan(rgb).sum()}")
rgb *= shade[..., None]
rgb *= (0.85 + 0.15 * (H / H.max()))[..., None]
print(f"rgb final: mean={rgb.mean():.3f} nan={np.isnan(rgb).sum()} "
      f"max={rgb.max():.3f}")
