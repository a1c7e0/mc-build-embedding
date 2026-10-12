"""japanese_house style grammar -> Skeleton-DSL.

Signature features (construction mapping, see src/stylegen/README.md):

* 低矮坡顶: gable roof with a shallow rise (pitch 2-3), wide eaves (walls
  inset under the full-grid roof). The compiler's ``shed`` roof only
  rasterises half the cross span (src/skeleton_compile.py span = cross//2),
  so the low-slope look is taken from gable pitch instead of shed.
* 缘侧 engawa: a veranda band on the front (1-2 deep) with a colonnade of
  1-cell wood piers at the platform edge under the eave.
* 障子窗: dense regular arrays of small 1x2 windows (2-block spacing) on
  every facade and on every storey.
* 木构架: wall/floor/roof all wood (timber frame look).
* 小入口: small 1-2 wide, 2-3 tall door placed off-centre on the veranda
  facade (never symmetrically centred like the Chinese gate).
"""
from __future__ import annotations

import random

from .dsl import build, column, opening, room, roof, slab, wall

NAME = "japanese_house"


def generate(seed: int):
    rng = random.Random(seed)
    sh = rng.choice([4, 5])
    n_st = rng.choice([1, 1, 2])
    engawa = rng.randint(1, 2)          # front veranda depth
    m = 1                               # other eave margins
    hw = rng.choice([7, 9, 9, 11])      # hall outer width (x)
    hd = rng.choice([5, 7, 7])          # hall outer depth (z)
    pitch = rng.choice([2, 3])
    ridge = rng.choice(["x", "x", "x", "z"])
    door_w = rng.choice([1, 1, 2])
    door_h = rng.choice([2, 3])
    win_w, win_h = 1, 2
    win_gap = rng.choice([2, 2, 3])

    W = hw + 2 * m
    D = hd + engawa + m
    x0, x1 = m, W - 1 - m               # hall wall ring
    z0, z1 = engawa, D - 1 - m
    door_h = min(door_h, sh - 1)

    storeys = list(range(n_st))
    walls = []

    # off-centre entrance: veranda facade, never mirrored
    door_at = rng.choice([1, max(1, hw // 3), max(1, 2 * hw // 3),
                          max(1, hw - 2)])
    door_at = min(door_at, hw - door_w)

    def window_ops(st: int, steps: int, door: bool):
        ops = []
        if door:
            ops.append(opening("door", door_at, door_w, door_h, 0))
        at = 1
        while at + win_w <= steps + 1:
            if not (door and at < door_at + door_w
                    and at + win_w > door_at):
                ops.append(opening("window", at, win_w, win_h, 1))
            at += win_gap
        return ops

    for st in storeys:
        walls.append(wall(x0, z0, x1, z0, [st], window_ops(st, hw - 1, True)))
        walls.append(wall(x1, z0, x1, z1, [st], window_ops(st, hd - 1, False)))
        walls.append(wall(x1, z1, x0, z1, [st], window_ops(st, hw - 1, False)))
        walls.append(wall(x0, z1, x0, z0, [st], window_ops(st, hd - 1, False)))

    # engawa colonnade at the platform edge
    cols = set()
    n_col = max(3, hw // 2)
    for k in range(n_col):
        cx = x0 + k * (hw - 1) // (n_col - 1)
        cols.add((cx, 0))
    for cx, cz in sorted(cols):
        walls.append(column(cx, cz, storeys))

    rooms = [room("room", "house", (x0 + 1, z0 + 1, x1, z1), storeys)]
    floors = [slab(0, (0, 0, W, D))] + [slab(s, (0, 0, W, D))
                                        for s in range(1, n_st)]
    sk = build(f"japanese_house_s{seed}", sh, W, D, rooms, walls, floors,
               roof("gable", pitch, ridge, overhang=engawa),
               {"wall": "wood", "floor": "wood", "roof": "wood",
                "frame": "wood"})
    params = {"seed": seed, "storey_height": sh, "storeys": n_st,
              "engawa": engawa, "roof": "gable", "pitch": pitch,
              "ridge": ridge, "door": [door_w, door_h, door_at],
              "window": [win_w, win_h], "window_gap": win_gap,
              "footprint": [W, D], "columns": len(cols)}
    return sk, params
