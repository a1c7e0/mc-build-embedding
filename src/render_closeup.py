"""Close-up renders (iso + aerial crop) of world regions for visual QA.

usage: python src/render_closeup.py <world_dir> --spots "name:x:z:size,..."
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_dir")
    ap.add_argument("--spots", required=True)
    ap.add_argument("--ymin", type=int, default=0,
                    help="crop away everything below this height "
                         "(for floating-island shots)")
    args = ap.parse_args()

    wf = os.path.join(args.world_dir, "world_full.npz")
    if not os.path.exists(wf):
        wf = os.path.join(args.world_dir, "world.npz")
    W = np.load(wf)["world"]
    t = np.load(os.path.join(args.world_dir, "terrain.npz"))
    H, biome = t["height"].astype(float), t["biome"]
    Wx, Hy, Dz = W.shape

    for spot in args.spots.split(","):
        name, x, z, size = spot.split(":")
        x, z, size = int(x), int(z), int(size)
        x0, x1 = max(x - size // 2, 0), min(x + size // 2, Wx)
        z0, z1 = max(z - size // 2, 0), min(z + size // 2, Dz)
        sub = W[x0:x1, :, z0:z1]
        # trim vertical air for a tighter crop
        occ = sub != 0
        if occ.any():
            yz = np.nonzero(occ)[1]
            y0, y1 = max(int(yz.min()) - 4, args.ymin), min(int(yz.max()) + 5, Hy)
            sub = sub[:, y0:y1, :]
        img = render(sub, max_px=760)
        img.save(os.path.join(args.world_dir, f"closeup_{name}.png"))

        gy, gx = np.gradient(H[x0:x1, z0:z1])
        shade = np.clip(0.75 + (gx * 0.7 - gy * 0.7) / 6.0, 0.35, 1.25)
        rgb = np.zeros((x1 - x0, z1 - z0, 3))
        b = biome[x0:x1, z0:z1]
        rgb[b == 0] = (0.25, 0.45, 0.75)
        rgb[b == 1] = (0.86, 0.80, 0.58)
        rgb[b == 2] = (0.52, 0.72, 0.42)
        rgb[b == 3] = (0.28, 0.52, 0.30)
        rgb[b == 4] = (0.62, 0.60, 0.58)
        rgb *= shade[..., None]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.clip(rgb, 0, 1).transpose(1, 0, 2), origin="lower")
        ax.set_title(f"{name} ({x},{z})")
        fig.savefig(os.path.join(args.world_dir, f"crop_{name}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[closeup] {name} -> closeup_{name}.png / crop_{name}.png",
              flush=True)


if __name__ == "__main__":
    main()
