"""Full-map isometric render (strided) of the assembled world."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_dir")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-px", type=int, default=1500)
    args = ap.parse_args()

    world = np.load(os.path.join(args.world_dir, "world_full.npz"))["world"]
    s = args.stride
    sub = world[::s, ::s, ::s]
    print(f"[iso] rendered grid {sub.shape} (stride {s})", flush=True)
    img = render(sub, max_px=args.max_px)
    dst = os.path.join(args.world_dir, "iso_full.png")
    img.save(dst)
    print(f"[iso] -> {dst}", flush=True)


if __name__ == "__main__":
    main()
