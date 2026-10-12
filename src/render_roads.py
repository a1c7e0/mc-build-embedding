"""Top-down view of a world region with road/path highlighting."""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROAD_BLOCK = 13  # gravel
PATH_BLOCK = 13  # same for now

COLORS = {
    0: (1, 1, 1), 1: (0.5, 0.5, 0.5), 2: (0.2, 0.8, 0.2),
    3: (0.6, 0.4, 0.2), 5: (0.9, 0.8, 0.6), 9: (0.2, 0.4, 0.9),
    13: (1.0, 0.5, 0.0), 17: (0.5, 0.3, 0.1), 18: (0.0, 0.5, 0.0),
    20: (0.7, 0.85, 1.0), 24: (0.9, 0.9, 0.5), 35: (1, 0, 0),
    85: (0.6, 0.3, 0.1), 89: (1, 1, 0), 98: (0.3, 0.3, 0.3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world_dir")
    ap.add_argument("--world", default="world_roads.npz")
    ap.add_argument("--x0", type=int, default=155)
    ap.add_argument("--x1", type=int, default=240)
    ap.add_argument("--z0", type=int, default=130)
    ap.add_argument("--z1", type=int, default=240)
    ap.add_argument("--out", default="roads_topview.png")
    args = ap.parse_args()

    world = np.load(os.path.join(args.world_dir, args.world))["world"]
    x0, x1, z0, z1 = args.x0, args.x1, args.z0, args.z1
    sub = world[x0:x1, :, z0:z1]
    H, D = sub.shape[0], sub.shape[2]

    top_block = np.zeros((H, D), dtype=int)
    for x in range(H):
        for z in range(D):
            col = sub[x, :, z]
            solid = np.nonzero(col)[0]
            if len(solid):
                top_block[x, z] = col[solid[-1]]

    rgb = np.ones((H, D, 3))
    for bid, c in COLORS.items():
        rgb[top_block == bid] = c

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_title("Roads (orange=gravel) + Buildings — market district")
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    plt.tight_layout()
    out = os.path.join(args.world_dir, args.out)
    plt.savefig(out, dpi=100)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
