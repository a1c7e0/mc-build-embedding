"""Bridge generated plans to the Skeleton DSL v0 compiler.

Uses only what a generated plan provides: the room rects, an exterior wall
shell, and a flat roof -- compiled with the existing deterministic
src/skeleton_compile.py, rendered with src/render.py.

Rooms are normalized [0,1] coords; the plan canvas becomes an SxS block
footprint (S=size, default 12 blocks ~= one median house).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from render import render2  # noqa: E402
from skeleton_compile import compile_skeleton  # noqa: E402

TYPE_MAP = {"Bedroom": "bedroom", "Bathroom": "bathroom",
            "LivingRoom": "living", "Kitchen": "kitchen"}


def _derive_walls(irooms: list, W: int, D: int) -> list:
    """Interior partition walls from shared room-rect edges, with door
    openings centered on each shared edge. Exterior walls get windows."""
    walls = []
    # exterior shell with window openings (one per side, centered)
    for (fx, fz, tx, tz) in [(0, 0, W, 0), (W, 0, W, D), (W, D, 0, D),
                             (0, D, 0, 0)]:
        length = max(abs(tx - fx), abs(tz - fz))
        ops = []
        if length >= 6:
            ops.append({"kind": "window", "at": length // 2 - 1,
                        "width": 2, "height": 2, "y": 1})
        walls.append({"from": [fx, fz], "to": [tx, tz], "storeys": [0],
                      "openings": ops})
    # interior walls: shared edge between two room rects
    n = len(irooms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = irooms[i]["rect"], irooms[j]["rect"]
            # vertical shared edge (a.x1 == b.x0 or b.x1 == a.x0)
            for (ra, rb) in [(a, b), (b, a)]:
                if ra[2] == rb[0]:  # ra.x1 == rb.x0
                    z0 = max(ra[1], rb[1])
                    z1 = min(ra[3], rb[3])
                    if z1 - z0 >= 2:
                        cx = ra[2]
                        wall_len = z1 - z0
                        door_at = max(0, wall_len // 2 - 1)
                        walls.append({
                            "from": [cx, z0], "to": [cx, z1],
                            "storeys": [0],
                            "openings": [{"kind": "door", "at": door_at,
                                          "width": 2, "height": 3, "y": 0}]})
                if ra[3] == rb[1]:  # ra.z1 == rb.z0
                    x0 = max(ra[0], rb[0])
                    x1 = min(ra[2], rb[2])
                    if x1 - x0 >= 2:
                        cz = ra[3]
                        wall_len = x1 - x0
                        door_at = max(0, wall_len // 2 - 1)
                        walls.append({
                            "from": [x0, cz], "to": [x1, cz],
                            "storeys": [0],
                            "openings": [{"kind": "door", "at": door_at,
                                          "width": 2, "height": 3, "y": 0}]})
    return walls


def plan_to_skeleton(rooms, house_id: str, size: int = 12,
                     storey_height: int = 4) -> tuple[dict, int]:
    """rooms: [(type, x0, z0, x1, z1)] normalized -> (skeleton dict, n_clamped)."""
    W = D = int(size)
    irooms = []
    clamped = 0
    for k, (typ, x0, z0, x1, z1) in enumerate(rooms):
        # Use round() consistently so shared boundaries snap to the same
        # integer (floor/ceil mismatch creates a 1-block gap and interior
        # wall detection fails).
        a0 = int(round(x0 * size))
        b0 = int(round(z0 * size))
        a1 = int(round(x1 * size))
        b1 = int(round(z1 * size))
        na0, nb0 = max(0, min(W - 1, a0)), max(0, min(D - 1, b0))
        na1, nb1 = max(na0 + 1, min(W, a1)), max(nb0 + 1, min(D, b1))
        if (na0, nb0, na1, nb1) != (a0, b0, a1, b1):
            clamped += 1
        irooms.append({"id": f"room{k}",
                       "type": TYPE_MAP.get(typ, "room"),
                       "rect": [na0, nb0, na1, nb1], "storeys": [0]})
    walls = _derive_walls(irooms, W, D)
    sk = {"id": house_id,
          "scale": {"unit": "block", "storey_height": storey_height},
          "footprint": {"w": W, "d": D},
          "rooms": irooms, "walls": walls,
          "floors": [{"storey": 0, "rect": [0, 0, W, D]}],
          "roof": {"kind": "flat", "pitch": 1, "ridge": "x"},
          "material": {"wall": "mineral", "floor": "wood",
                       "roof": "mineral", "frame": "wood"}}
    return sk, clamped


def export_and_compile(rooms, out_dir: str, house_id: str,
                       size: int = 12) -> dict:
    """Write skeleton json + compiled npz + cutaway png; return summary."""
    os.makedirs(out_dir, exist_ok=True)
    sk, clamped = plan_to_skeleton(rooms, house_id, size=size)
    jf = os.path.join(out_dir, f"{house_id}.json")
    json.dump(sk, open(jf, "w"), indent=1)

    g = compile_skeleton(sk)
    np.savez_compressed(jf.replace(".json", ".npz"), grid=g)
    render2(g, max_px=320, cutaway=True).save(jf.replace(".json", "_cutaway.png"))
    return {"skeleton_json": jf, "grid_shape": list(g.shape),
            "nonzero_voxels": int(np.count_nonzero(g)),
            "n_rooms": len(sk["rooms"]), "n_clamped_rects": clamped,
            "footprint": [sk["footprint"]["w"], sk["footprint"]["d"]]}
