"""Grow a settlement from a seed: roads -> plots -> slots (logic-scale).

Lane-3 prototype of "logical magnification": instead of generating a whole
map at once, grow it hierarchically — a main road enters, branches, plots
subdivide along the roads, slots appear with center-biased types. Outputs
a growth.json compatible with assemble2's slot format and a plan render.

usage: python src/grow.py [out_dir] [seed]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

W = D = 400


def grow_roads(rng):  # noqa: ANN001
    """Road segments: meandering trunk + probabilistic branches."""
    roads = []
    queue = [((20, W // 2), 0.0, 2, 0)]       # pos, heading, width, depth
    while queue and len(roads) < 24:
        (x, z), h, wd, depth = queue.pop()
        seg = [(x, z)]
        length = rng.integers(40, 90)
        for _ in range(length):
            h += rng.normal(0, 0.08)
            x += np.cos(h) * 3
            z += np.sin(h) * 3
            if not (10 < x < W - 10 and 10 < z < D - 10):
                break
            seg.append((x, z))
            if depth < 2 and rng.random() < 0.02 and len(seg) > 15:
                queue.append(((x, z), h + rng.choice([-1.2, 1.2]),
                              max(wd - 1, 1), depth + 1))
        if len(seg) > 20:
            roads.append({"points": seg, "width": wd})
    return roads


def subdivide_plots(rng, roads):  # noqa: ANN001
    """Plots along both sides of roads; slots inset from the road."""
    slots = []
    plots = []
    for rd in roads:
        pts = rd["points"]
        step = 14
        for i in range(6, len(pts) - 6, step):
            (x0, z0), (x1, z1) = pts[i], pts[min(i + step, len(pts) - 1)]
            dx, dz = x1 - x0, z1 - z0
            L = np.hypot(dx, dz) + 1e-9
            nx, nz = -dz / L, dx / L
            for side in (1, -1):
                depth = rng.integers(10, 16)
                cx = x0 + nx * side * (depth + rd["width"])
                cz = z0 + nz * side * (depth + rd["width"])
                if not (20 < cx < W - 20 and 20 < cz < D - 20):
                    continue
                plots.append({"x": cx, "z": cz, "r": depth})
                slots.append({"district": "grown", "label": "grown plot",
                              "x": int(cx), "z": int(cz), "y": 0,
                              "clusters": [1, 3, 4, 6]})
    return plots, slots


def render_plan(roads, plots, path):  # noqa: ANN001
    im = Image.new("RGB", (W, D), (250, 250, 245))
    dr = ImageDraw.Draw(im)
    for rd in roads:
        dr.line([(x, z) for x, z in rd["points"]],
                fill=(200, 160, 110), width=max(rd["width"] * 3, 2))
    for p in plots:
        r = p["r"]
        dr.ellipse([p["x"] - r, p["z"] - r, p["x"] + r, p["z"] + r],
                   outline=(90, 130, 200))
    im.save(path)
    print(f"[grow] plan -> {path}", flush=True)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "out/grown_town"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    roads = grow_roads(rng)
    plots, slots = subdivide_plots(rng, roads)
    json.dump({"roads": [{"points": [[float(x), float(z)] for x, z in
                                     r["points"]][::4], "width": r["width"]}
                         for r in roads],
               "slots": slots,
               "plots": len(plots)},
              open(os.path.join(out, "growth.json"), "w"))
    render_plan(roads, plots, os.path.join(out, "plan.png"))
    print(f"[grow] roads={len(roads)} plots={len(plots)} "
          f"slots={len(slots)}", flush=True)


if __name__ == "__main__":
    main()
