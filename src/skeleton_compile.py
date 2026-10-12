"""Deterministic Skeleton-DSL compiler: JSON -> voxel shell.

No learning, no search. Walls (1-thick) + openings (carved) + floor slabs
+ room interiors + roof (flat/gable/shed). Legacy block ids via the
material map: mineral=1 wood=5 glass=20 organic=18.

usage: python src/skeleton_compile.py in.json [out.npz] [out.png]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402

MAT = {"mineral": 1, "wood": 5, "glass": 20, "organic": 18}


def compile_skeleton(sk: dict) -> np.ndarray:
    sh = int(sk.get("scale", {}).get("storey_height", 4))
    fp = sk["footprint"]
    W, D = int(fp["w"]), int(fp["d"])
    n_st = 1 + max((int(s) for r in sk.get("rooms", [])
                    for s in r.get("storeys", [0])), default=0)
    roof = sk.get("roof", {})
    pitch = int(roof.get("pitch", 1))
    kind0 = roof.get("kind", "flat")
    cross = (D if roof.get("ridge", "x") == "x" else W) if kind0 != "shed" \
        else (D if roof.get("ridge", "x") == "x" else W)
    if kind0 in ("gable", "hip"):
        rise = (cross // 2) // max(pitch, 1) + 2
    elif kind0 == "shed":
        rise = cross // max(pitch, 1) + 2
    else:
        rise = 1
    H = sh * n_st + rise
    g = np.zeros((W, H, D), dtype=np.uint16)
    wall_m = MAT[sk.get("material", {}).get("wall", "mineral")]
    floor_m = MAT[sk.get("material", {}).get("floor", "mineral")]
    roof_m = MAT[sk.get("material", {}).get("roof", "mineral")]

    for fl in sk.get("floors", []):
        y = int(fl["storey"]) * sh - 1 if int(fl["storey"]) > 0 else 0
        x0, z0, x1, z1 = [int(v) for v in fl["rect"]]
        if 0 <= y < H:
            g[x0:x1, y, z0:z1] = floor_m

    for r in sk.get("rooms", []):
        x0, z0, x1, z1 = [int(v) for v in r["rect"]]
        for st in r.get("storeys", [0]):
            y0, y1 = int(st) * sh + 1, (int(st) + 1) * sh
            g[x0:x1, y0:y1, z0:z1] = 0

    for w in sk.get("walls", []):
        (ax, az), (bx, bz) = w["from"], w["to"]
        ax, az, bx, bz = int(ax), int(az), int(bx), int(bz)
        steps = max(abs(bx - ax), abs(bz - az))
        for t in range(steps + 1):
            x = min(ax + (bx - ax) * t // max(steps, 1), W - 1)
            z = min(az + (bz - az) * t // max(steps, 1), D - 1)
            for st in w.get("storeys", list(range(n_st))):
                y0 = int(st) * sh
                for y in range(y0, min(y0 + sh, H)):
                    if 0 <= x < W and 0 <= z < D:
                        g[x, y, z] = wall_m
        for op in w.get("openings", []):
            at, wid, hei = int(op["at"]), int(op["width"]), int(op["height"])
            ys = int(w.get("storeys", [0])[0]) * sh + int(op.get("y", 0))
            for t in range(at, at + wid):
                x = min(ax + (bx - ax) * t // max(steps, 1), W - 1)
                z = min(az + (bz - az) * t // max(steps, 1), D - 1)
                for y in range(ys, min(ys + hei, H)):
                    if 0 <= x < W and 0 <= z < D:
                        g[x, y, z] = 0

    kind = roof.get("kind", "flat")
    oh = int(roof.get("overhang", 0))
    ytop = sh * n_st
    if kind == "flat":
        g[max(0, 0 - oh):W + oh, ytop, max(0, 0 - oh):D + oh] = 0
        g[0:W, ytop, 0:D] = roof_m
    else:
        axis = 0 if roof.get("ridge", "x") == "x" else 2   # ridge runs along
        span = (D if axis == 0 else W) // 2 + 1            # cross-axis half
        for i in range(span):
            y = ytop + i // max(pitch, 1)
            if y >= H:
                break
            if kind == "shed":
                if axis == 0:
                    g[0:W, y, i] = roof_m
                else:
                    g[i, y, 0:D] = roof_m
            elif axis == 0:
                g[0:W, y, i] = roof_m
                g[0:W, y, D - 1 - i] = roof_m
            else:
                g[i, y, 0:D] = roof_m
                g[W - 1 - i, y, 0:D] = roof_m
        # gable-end infill: close the triangle at both ends of the ridge
        for k in range(D if axis == 0 else W):
            run = min(k, (D - 1 - k) if axis == 0 else (W - 1 - k))
            ytop_k = ytop + run // max(pitch, 1)
            for y in range(ytop, min(ytop_k + 1, H)):
                if axis == 0:
                    g[0, y, k] = roof_m
                    g[W - 1, y, k] = roof_m
                else:
                    g[k, y, 0] = roof_m
                    g[k, y, D - 1] = roof_m
    return g


def main() -> None:
    src = sys.argv[1]
    sk = json.load(open(src, encoding="utf-8"))
    g = compile_skeleton(sk)
    out = sys.argv[2] if len(sys.argv) > 2 else src.replace(".json", ".npz")
    np.savez_compressed(out, grid=g)
    if len(sys.argv) > 3:
        render2(g, max_px=300, cutaway=True).save(sys.argv[3])
    nz = np.nonzero(g)
    print(f"[skel-compile] {os.path.basename(src)} -> {g.shape} "
          f"fill={float((g > 0).mean()):.3f} solid={len(nz[0])}", flush=True)


if __name__ == "__main__":
    main()
