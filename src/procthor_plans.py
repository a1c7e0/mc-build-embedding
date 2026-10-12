"""ProcTHOR plan-skeleton extraction: house JSON -> plan ground truth.

Extracts the PLAN layer per house: rooms (type, exact polygon, rect
approximation, area), adjacency graph (via door room0/room1), openings
per wall, storey (floor y), wall Manhattan-ness. Also measures how well
the rect-based Skeleton DSL v0 can express real plans:
  rect_exact  = room polygon area == its bbox area (lossless for DSL)
Output: out_lib/plans/procthor_plans.jsonl + stats on stdout.

usage: python src/procthor_plans.py [in.jsonl.gz] [out.jsonl] [limit]
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter


def poly_area(poly) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, z1 = poly[i]["x"], poly[i]["z"]
        x2, z2 = poly[(i + 1) % n]["x"], poly[(i + 1) % n]["z"]
        s += x1 * z2 - x2 * z1
    return abs(s) / 2


def extract(h: dict) -> dict | None:
    rooms = h.get("rooms", [])
    if not rooms:
        return None
    out_rooms = []
    for r in rooms:
        poly = [(p["x"], p["z"]) for p in r.get("floorPolygon", [])]
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        rect = [min(xs), min(zs), max(xs), max(zs)]
        a_poly = poly_area([{"x": x, "z": z} for x, z in poly])
        a_rect = (rect[2] - rect[0]) * (rect[3] - rect[1])
        out_rooms.append({
            "id": r["id"],
            "type": r.get("roomType", "?"),
            "poly": [[round(x, 3), round(z, 3)] for x, z in poly],
            "rect": [round(v, 3) for v in rect],
            "y": round(sum(p["y"] for p in r["floorPolygon"]) /
                      max(len(r["floorPolygon"]), 1), 2),
            "a_poly": round(a_poly, 2),
            "a_rect": round(a_rect, 2),
            "rect_exact": abs(a_poly - a_rect) < 0.01 * max(a_poly, 1),
            "floor_material": (r.get("floorMaterial") or {}).get("name"),
        })
    edges = []
    for dk in ("doors", "windows"):
        for o in h.get(dk, []):
            r0, r1 = o.get("room0"), o.get("room1")
            if r0 and r1:
                edges.append([r0, r1, dk[:-1]])
    walls = []
    for w in h.get("walls", []):
        poly = [(p["x"], p["z"]) for p in w.get("polygon", [])]
        manh = True
        for i in range(len(poly) - 1):
            if abs(poly[i][0] - poly[i + 1][0]) > 0.01 and \
                    abs(poly[i][1] - poly[i + 1][1]) > 0.01:
                manh = False
        walls.append({"id": w["id"], "empty": bool(w.get("empty")),
                      "manhattan": manh,
                      "material": (w.get("material") or {}).get("name")})
    return {
        "rooms": out_rooms,
        "edges": edges,
        "walls": walls,
        "n_windows": len(h.get("windows", [])),
        "storeys": sorted({r["y"] for r in out_rooms}),
    }


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "out_lib/procthor/train.jsonl.gz"
    out = sys.argv[2] if len(sys.argv) > 2 else \
        "out_lib/plans/procthor_plans.jsonl"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    types = Counter()
    rooms_exact = rooms_all = 0
    houses_exact = 0
    n_rooms_hist = Counter()
    manh = Counter()
    with gzip.open(src, "rt") as fin, open(out, "w", encoding="utf-8") as fo:
        for i, line in enumerate(fin):
            if limit and i >= limit:
                break
            rec = extract(json.loads(line))
            if rec is None:
                continue
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            types.update(r["type"] for r in rec["rooms"])
            ok = all(r["rect_exact"] for r in rec["rooms"])
            houses_exact += int(ok)
            for r in rec["rooms"]:
                rooms_all += 1
                rooms_exact += int(r["rect_exact"])
            n_rooms_hist[len(rec["rooms"])] += 1
            manh.update(w["manhattan"] for w in rec["walls"])
    print(f"[plans] {n} houses -> {out}")
    print(f"[plans] rect-exact rooms: {rooms_exact}/{rooms_all} "
          f"({rooms_exact / max(rooms_all, 1):.1%}), "
          f"rect-exact houses: {houses_exact}/{n} "
          f"({houses_exact / max(n, 1):.1%})")
    print(f"[plans] Manhattan walls: {manh[True]}/{manh[True] + manh[False]} "
          f"({manh[True] / max(manh[True] + manh[False], 1):.1%})")
    print(f"[plans] room types: {dict(types.most_common())}")
    print(f"[plans] rooms/house: min {min(n_rooms_hist)} max "
          f"{max(n_rooms_hist)} mode "
          f"{n_rooms_hist.most_common(1)[0][0]}")


if __name__ == "__main__":
    main()
