"""chinese_temple style grammar -> Skeleton-DSL.

Signature features (construction mapping, see src/stylegen/README.md):

* 台基 podium: whole platform raised 1-2 blocks (fill rows) + a stepped
  stair in front of the gate, descending toward the platform edge.
* 红柱 colonnade: wood-material 1-cell piers at the platform edge
  (front/back rows + side rows), spaced on symmetric bays (对称开间).
* 灰瓦坡顶: mineral gable/hip roof; the walls are inset from the grid
  edge so the full-grid roof reads as a deep overhanging eave (大出檐).
* 叠檐 tier lines: an eave-lip band at every storey top inside the eave
  margin (one tier per storey, so 2-storey variants read as double-eave).
* 宽大门: wide centred gate (3-5 blocks), symmetric window bays around it.
"""
from __future__ import annotations

import random

from .dsl import build, column, fill_rect, opening, room, roof, slab, wall

NAME = "chinese_temple"


def generate(seed: int):
    rng = random.Random(seed)
    bays = rng.choice([3, 5, 5, 7])
    bay = rng.randint(2, 3)
    eave = rng.choice([1, 2, 2, 3])
    podium_h = rng.choice([2, 2, 3])
    sh = rng.choice([6, 7, 8])
    n_st = rng.choice([1, 1, 2])
    roof_kind = rng.choice(["gable", "hip"])
    pitch = rng.choice([1, 2])
    door_w = rng.choice([3, 3, 5])
    win_w = rng.choice([1, 2])
    win_h = rng.choice([2, 3])

    hw = bays * bay + 2                       # hall outer width (incl walls)
    hd = rng.choice([9, 11, 13])              # hall outer depth
    W = hw + 2 * eave
    D = hd + 2 * eave
    x0, x1 = eave, W - 1 - eave               # hall wall ring
    z0, z1 = eave, D - 1 - eave

    door_at = (hw - door_w) // 2
    door_h = min(sh - 1, 6)
    sill = podium_h + 1
    win_h = min(win_h, sh - sill)

    storeys = list(range(n_st))
    walls = []

    # ---- fills: podium + eave tier lines + front stair -----------------
    ring = [(s * sh - 2, s * sh) for s in range(1, n_st + 1)]
    podium = [(0, podium_h)]
    inner = podium + [(s * sh - 1, s * sh) for s in range(1, n_st)]
    walls += fill_rect((x0, z0, x1 + 1, z1 + 1), inner, sh)
    marg = podium + ring
    walls += fill_rect((0, 0, W, eave), marg, sh)             # front band
    walls += fill_rect((0, D - eave, W, D), marg, sh)         # back band
    walls += fill_rect((0, eave, eave, D - eave), marg, sh)   # left band
    walls += fill_rect((W - eave, eave, W, D - eave), marg, sh)  # right band

    dx0 = x0 + door_at
    dx1 = dx0 + door_w - 1
    for j in range(eave):                     # stair: descend toward z=0
        top = max(1, podium_h - eave + j)     # blocks y<top (ground top = 1)
        walls += fill_rect((max(dx0 - 1, 0), j, min(dx1 + 2, W), j + 1),
                           [(0, top)] + ring, sh)

    # ---- walls: symmetric bays ---------------------------------------
    def front_ops(st: int):
        ops = []
        if st == 0:
            ops.append(opening("door", door_at, door_w, door_h, 0))
        for k in range(bays):
            at = hw * (2 * k + 1) // (2 * bays) - win_w // 2
            if st == 0 and at < door_at + door_w and at + win_w > door_at:
                continue                      # gate takes this bay
            if 0 <= at and at + win_w <= hw:
                sy = sill if st == 0 else 2
                ops.append(opening("window", at, win_w, win_h, sy))
        return ops

    def side_ops(st: int, n: int):
        ops = []
        for k in range(n):
            at = hd * (2 * k + 1) // (2 * n) - win_w // 2
            if 0 <= at and at + win_w <= hd:
                sy = sill if st == 0 else 2
                ops.append(opening("window", at, win_w, win_h, sy))
        return ops

    for st in storeys:
        walls.append(wall(x0, z0, x1, z0, [st], front_ops(st)))
        walls.append(wall(x1, z0, x1, z1, [st], side_ops(st, 2)))
        walls.append(wall(x1, z1, x0, z1, [st], front_ops(st)))
        walls.append(wall(x0, z1, x0, z0, [st], side_ops(st, 2)))

    # ---- colonnade at the platform edge ------------------------------
    cols = set()
    for k in range(bays + 1):
        cx = x0 + k * (hw - 1) // bays
        cols.add((cx, 0))
        cols.add((cx, D - 1))
    for k in range(3):
        cz = z0 + k * (hd - 1) // 2
        cols.add((0, cz))
        cols.add((W - 1, cz))
    for cx, cz in sorted(cols):
        walls.append(column(cx, cz, storeys))

    rooms = [room("hall", "temple", (x0 + 1, z0 + 1, x1, z1), storeys)]
    floors = [slab(0, (0, 0, W, D))] + [slab(s, (0, 0, W, D))
                                        for s in range(1, n_st)]
    sk = build(f"chinese_temple_s{seed}", sh, W, D, rooms, walls, floors,
               roof(roof_kind, pitch, "x", overhang=eave),
               {"wall": "wood", "floor": "mineral", "roof": "mineral",
                "frame": "wood"})
    params = {"seed": seed, "bays": bays, "bay_width": bay, "eave": eave,
              "podium_height": podium_h, "storey_height": sh,
              "storeys": n_st, "roof": roof_kind, "pitch": pitch,
              "door_width": door_w, "window": [win_w, win_h],
              "footprint": [W, D], "columns": len(cols)}
    return sk, params
