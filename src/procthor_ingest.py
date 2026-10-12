"""ProcTHOR-10K ingestion: house JSON -> 6-class 32^3 voxel tensors.

Voxelizes at 1 block = 1 m: room floor polygons (slabs), wall quads
(1-thick, height from polygon), door/window holes carved in wall-local
(u, y) ranges, flat roof over the top layer. Materials map to the 6
classes by name keyword. Furniture is skipped (no bbox in JSON) — this
corpus teaches STRUCTURE. Output matches vq_data2 layout.

usage: python src/procthor_ingest.py [in.jsonl.gz] [out.npz] [limit]
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

SIZE = 32
FIT = 30


def mat_class(name: str | None) -> int:
    n = (name or "").lower()
    if "glass" in n:
        return 3
    if "wood" in n:
        return 2
    if any(k in n for k in ("plant", "leaf", "grass", "moss", "bamboo")):
        return 4
    return 1                      # mineral default (stone/brick/drywall/...)


def raster_poly(poly, y: int, g: np.ndarray, cls: int) -> None:
    """Fill a floor polygon (y-plane) into grid g as a 1-thick slab."""
    xs = [p["x"] for p in poly]
    zs = [p["z"] for p in poly]
    x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs))) + 1
    z0, z1 = int(np.floor(min(zs))), int(np.ceil(max(zs))) + 1
    if not (0 <= y < g.shape[1]):
        return
    # point-in-polygon for each cell centre (bbox scan; polygons are small)
    for x in range(max(x0, 0), min(x1, g.shape[0])):
        for z in range(max(z0, 0), min(z1, g.shape[2])):
            px, pz = x + 0.5, z + 0.5
            inside = False
            n = len(poly)
            for i in range(n):
                xA, zA = poly[i]["x"], poly[i]["z"]
                xB, zB = poly[(i + 1) % n]["x"], poly[(i + 1) % n]["z"]
                if (zA > pz) != (zB > pz) and \
                        px < (xB - xA) * (pz - zA) / (zB - zA + 1e-9) + xA:
                    inside = not inside
            if inside:
                g[x, y, z] = cls


def voxelize(h: dict) -> np.ndarray | None:
    rooms = h.get("rooms", [])
    walls = h.get("walls", [])
    if not rooms:
        return None
    pts = [p for r in rooms for p in r.get("floorPolygon", [])]
    if not pts:
        return None
    H = 4
    for w in walls:
        for p in w.get("polygon", []):
            H = max(H, int(np.ceil(p["y"])) + 1)
    W = int(np.ceil(max(p["x"] for p in pts))) + 2
    D = int(np.ceil(max(p["z"] for p in pts))) + 2
    g = np.zeros((max(W, 2), H + 2, max(D, 2)), dtype=np.uint8)

    for r in rooms:
        poly = r.get("floorPolygon", [])
        if not poly:
            continue
        ly = int(round(sum(p["y"] for p in poly) / len(poly)))
        raster_poly(poly, ly, g,
                    mat_class((r.get("floorMaterial") or {}).get("name")))

    hole_by_wall: dict = {}
    for key in ("doors", "windows"):
        for o in h.get(key, []):
            hp = o.get("holePolygon", [])
            if len(hp) < 2:
                continue
            u0, u1 = sorted((hp[0]["x"], hp[1]["x"]))
            y0, y1 = sorted((hp[0]["y"], hp[1]["y"]))
            for wk in ("wall0", "wall1"):
                hole_by_wall.setdefault(o.get(wk), []).append((u0, u1, y0, y1))

    for w in walls:
        poly = w.get("polygon", [])
        if len(poly) < 2:
            continue
        base = sorted((p for p in poly if p["y"] <= 0.01),
                      key=lambda p: (p["x"], p["z"]))
        if len(base) < 2:
            base = poly[:2]
        (ax, az), (bx, bz) = ((base[0]["x"], base[0]["z"]),
                              (base[1]["x"], base[1]["z"]))
        L = float(np.hypot(bx - ax, bz - az)) + 1e-9
        base_y = min(p["y"] for p in poly)
        hmax = max(p["y"] for p in poly)
        cls = mat_class((w.get("material") or {}).get("name"))
        solid = not w.get("empty", False)
        holes = hole_by_wall.get(w.get("id"), [])
        steps = max(int(L), 1)
        for t in range(steps + 1):
            u = L * t / steps
            x = ax + (bx - ax) * t / steps
            z = az + (bz - az) * t / steps
            xi, zi = int(round(x)), int(round(z))
            if not (0 <= xi < g.shape[0] and 0 <= zi < g.shape[2]):
                continue
            for y in range(int(round(base_y)),
                           min(int(round(hmax)) + 1, g.shape[1])):
                if any(hu0 - 0.5 <= u <= hu1 + 0.5 and
                       hy0 - 0.5 <= y - base_y <= hy1 + 0.5
                       for hu0, hu1, hy0, hy1 in holes):
                    continue                      # door/window hole
                if solid or any(hu0 <= u <= hu1 for hu0, hu1, _, _ in holes):
                    g[xi, y, zi] = cls

    # flat roof slab over the top-layer rooms (corpus had no roofs before)
    roof_y = 0
    for w in walls:
        for p in w.get("polygon", []):
            roof_y = max(roof_y, int(round(p["y"])))
    top_y = 0
    for r in rooms:
        poly = r.get("floorPolygon", [])
        if poly:
            top_y = max(top_y, int(round(sum(p["y"] for p in poly)
                                          / len(poly))))
    for r in rooms:
        poly = r.get("floorPolygon", [])
        if poly and int(round(sum(p["y"] for p in poly) / len(poly))) == top_y:
            raster_poly(poly, roof_y, g, 1)
    return g


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "out_lib/procthor/val.jsonl.gz"
    out = sys.argv[2] if len(sys.argv) > 2 else "out_lib/vq_data_pro.npz"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    xs, seen, n_bad = [], set(), 0
    with gzip.open(src, "rt") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            g = voxelize(json.loads(line))
            if g is None or not (g > 0).any():
                n_bad += 1
                continue
            nz = np.nonzero(g)
            crop = g[nz[0].min():nz[0].max() + 1,
                     nz[1].min():nz[1].max() + 1,
                     nz[2].min():nz[2].max() + 1]
            f30 = FIT / max(crop.shape)
            rz = zoom(crop, (f30, f30, f30), order=0)
            if rz.shape[0] > SIZE or rz.shape[1] > SIZE or rz.shape[2] > SIZE:
                continue
            out32 = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
            o = [(SIZE - s) // 2 for s in rz.shape]
            out32[o[0]:o[0] + rz.shape[0],
                  o[1]:o[1] + rz.shape[1],
                  o[2]:o[2] + rz.shape[2]] = rz
            hb = out32.tobytes()
            if hb in seen:
                continue
            seen.add(hb)
            xs.append(out32)
            if len(xs) % 1000 == 0:
                print(f"[procthor] {len(xs)} houses...", flush=True)
    X = np.stack(xs)
    styles = np.zeros(len(X), dtype=np.int32)          # all 'modern'
    np.savez_compressed(out, X=X, styles=styles)
    frac = (X > 0).mean()
    print(f"[procthor] {X.shape} -> {out} fill={frac:.3f} "
          f"skipped={n_bad}", flush=True)


if __name__ == "__main__":
    main()
