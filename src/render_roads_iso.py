"""Quick closeup render of the road area from world_roads.npz."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render2


def main():
    wd = "out/world/valley_port"
    world = np.load(os.path.join(wd, "world_roads.npz"))["world"]
    # crop market area with roads
    x0, x1, z0, z1 = 155, 240, 130, 240
    sub = world[x0:x1, :, z0:z1]
    img = render2(sub, max_px=480, cutaway=False)
    out = os.path.join(wd, "roads_iso.png")
    img.save(out)
    print(f"saved {out}")
    # also a cutaway to show roads are at ground level
    img2 = render2(sub, max_px=480, cutaway=True)
    out2 = os.path.join(wd, "roads_cutaway.png")
    img2.save(out2)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
