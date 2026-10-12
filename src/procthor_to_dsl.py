"""ProcTHOR house -> Skeleton DSL v0 -> compiled shell (end-to-end demo).

Reads one house from the raw jsonl, converts plan+walls+openings into
skeleton DSL JSON (metres rounded to blocks), then calls the deterministic
compiler and renders a cutaway + a top-down plan diagram.

usage: python src/procthor_to_dsl.py in.jsonl.gz line_index out_dir
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402
from skeleton_compile import compile_skeleton  # noqa: E402

TYPE_MAP = {"Bedroom": "bedroom", "Bathroom": "bathroom",
            "LivingRoom": "living", "Kitchen": "kitchen"}


def main() -> None:
    src, idx, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(src, "rt") as f:
        for i, line in enumerate(f):
            if i == idx:
                h = json.loads(line)
                break
    pts = [p for r in h["rooms"] for p in r["floorPolygon"]]
    W = int(np.ceil(max(p["x"] for p in pts))) + 1
    D = int(np.ceil(max(p["z"] for p in pts))) + 1
    wall_h = max(p["y"] for w in h["walls"] for p in w["polygon"])
    sh = max(3, int(round(wall_h)))

    rooms = []
    for r in h["rooms"]:
        poly = r["floorPolygon"]
        xs = [p["x"] for p in poly]
        zs = [p["z"] for p in poly]
        rooms.append({"id": r["id"],
                      "type": TYPE_MAP.get(r.get("roomType"), "room"),
                      "rect": [int(round(min(xs))), int(round(min(zs))),
                               int(round(max(xs))), int(round(max(zs)))],
                      "storeys": [0]})

    holes = {}
    for dk in ("doors", "windows"):
        for o in h.get(dk, []):
            hp = o.get("holePolygon", [])
            if len(hp) < 2:
                continue
            u0, u1 = sorted((hp[0]["x"], hp[1]["x"]))
            y0, y1 = sorted((hp[0]["y"], hp[1]["y"]))
            holes.setdefault(o.get("wall0"), []).append(
                (u0, u1, y0, y1, dk[:-1]))
            holes.setdefault(o.get("wall1"), []).append(
                (u0, u1, y0, y1, dk[:-1]))

    walls = []
    for w in h["walls"]:
        poly = [(p["x"], p["z"]) for p in w["polygon"]]
        base = [p for p in w["polygon"] if p["y"] <= 0.01]
        if len(base) < 2:
            base = w["polygon"][:2]
        base = base[:2]                      # keep polygon order: u origin
        (ax, az), (bx, bz) = ((base[0]["x"], base[0]["z"]),
                              (base[1]["x"], base[1]["z"]))
        ops = []
        for u0, u1, y0, y1, kind in holes.get(w["id"], []):
            ops.append({"kind": "door" if kind == "door" else "window",
                        "at": max(0, int(round(u0))),
                        "width": max(1, int(round(u1 - u0))),
                        "height": max(1, int(round(y1 - y0))),
                        "y": int(round(y0))})
        walls.append({"from": [int(round(ax)), int(round(az))],
                      "to": [int(round(bx)), int(round(bz))],
                      "storeys": [0], "openings": ops})

    sk = {"id": f"procthor_{idx}",
          "scale": {"unit": "block", "storey_height": sh},
          "footprint": {"w": W, "d": D},
          "rooms": rooms, "walls": walls,
          "floors": [{"storey": 0, "rect": [0, 0, W, D]}],
          "roof": {"kind": "flat", "pitch": 1, "ridge": "x"},
          "material": {"wall": "mineral", "floor": "wood",
                       "roof": "mineral", "frame": "wood"}}
    jf = os.path.join(out_dir, f"procthor_{idx}.json")
    json.dump(sk, open(jf, "w"), indent=1)

    g = compile_skeleton(sk)
    np.savez_compressed(jf.replace(".json", ".npz"), grid=g)
    render2(g, max_px=320, cutaway=True).save(jf.replace(".json", ".png"))

    im = Image.new("RGB", (W * 10 + 20, D * 10 + 20), (250, 250, 245))
    dr = ImageDraw.Draw(im)
    colors = {"bedroom": (120, 160, 220), "bathroom": (120, 200, 200),
              "living": (220, 180, 120), "kitchen": (180, 220, 120),
              "room": (200, 200, 200)}
    for r in rooms:
        x0, z0, x1, z1 = r["rect"]
        dr.rectangle([10 + x0 * 10, 10 + z0 * 10, 10 + x1 * 10, 10 + z1 * 10],
                     fill=colors.get(r["type"], (200, 200, 200)),
                     outline=(60, 60, 60))
        dr.text((12 + x0 * 10, 12 + z0 * 10), r["type"], fill=(30, 30, 30))
    im.save(os.path.join(out_dir, f"procthor_{idx}_plan.png"))
    print(f"[to-dsl] house {idx}: {len(rooms)} rooms "
          f"{[r['type'] for r in rooms]} {W}x{D} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
