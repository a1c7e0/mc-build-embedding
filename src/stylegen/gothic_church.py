"""gothic_church style grammar -> Skeleton-DSL.

Signature features (construction mapping, see src/stylegen/README.md):

* 高耸窄体量: long narrow footprint (21-29 x 13-17) with tall storeys
  (sh 10-12 single volume, or two sh 5-6 storeys = clerestory).
* 尖顶: gable roof with pitch 1-2 (pitch is inverse: small pitch = steep),
  ridge along the nave so the cross-axis rise reads as a spire-like roof.
* 高窗: arrays of narrow tall lancets (2 wide x 5-8 high) between bays on
  both long walls and on the ends.
* 扶壁: 1-cell-wide full-height piers bridging the 2-deep buttress slot
  between the long walls and the grid edge, one per bay.
* 十字形平面: the rooms tile a cross (nave / crossing / chancel / two
  transept arms / four corner chapels) and two internal transept walls
  with tall arch openings cut the cruciform aisle line.
"""
from __future__ import annotations

import random

from .dsl import (build, column, fill_rect, opening, room, roof, slab, wall)

NAME = "gothic_church"


def generate(seed: int):
    rng = random.Random(seed)
    n_st = rng.choice([1, 1, 1, 1, 2])
    sh = rng.choice([11, 12]) if n_st == 1 else 6
    L = rng.choice([25, 27, 29, 31])
    Wd = rng.choice([13, 15])
    pitch = rng.choice([1, 1, 1, 2])
    m = 2                                # buttress slot depth
    n_bays = rng.randint(5, 7)
    win_w = rng.choice([1, 2])
    door_w = rng.choice([3, 5])
    nw = rng.choice([2, 3])              # nave half width in cells

    W, D = L, Wd
    zN, zS = m, D - 1 - m                # long wall lines
    storeys = list(range(n_st))
    walls = []
    total_h = n_st * sh

    # cruciform layout: nave strip along x, transept strip across
    iz0, iz1 = zN + 1, zS                # interior z span (exclusive)
    half = max(1, min(nw, (iz1 - iz0 - 2) // 2))   # keep both arms >=1 deep
    nz0 = max(iz0 + 1, D // 2 - half)
    nz1 = min(iz1 - 1, D // 2 + half)
    tx0, tx1 = W // 3, 2 * W // 3

    # ---- fills: stone plinth (+ inter-storey slabs survive the fill) --
    plinth = [(0, 1)] + [(s * sh - 1, s * sh) for s in range(1, n_st)]
    walls += fill_rect((0, 0, W, D), plinth, sh)

    # ---- buttress bays ----------------------------------------------
    xb = [2 + k * (W - 5) // n_bays for k in range(n_bays + 1)]
    win_h = sh - 4 if n_st == 1 else sh - 3

    def bay_windows(st: int):
        ops = []
        sill = 2 if n_st == 1 else 1
        for k in range(len(xb) - 1):
            cx = (xb[k] + xb[k + 1]) // 2
            at = cx - win_w // 2
            if at + win_w <= W:
                ops.append(opening("window", at, win_w, win_h, sill))
        return ops

    for st in storeys:
        walls.append(wall(0, zN, W - 1, zN, [st], bay_windows(st)))
        walls.append(wall(0, zS, W - 1, zS, [st], bay_windows(st)))

    # west portal (gable end): one entry across all storeys = tall arch
    portal = [opening("door", (D - door_w) // 2, door_w, total_h - 2, 0)]
    walls.append(wall(0, 0, 0, D - 1, storeys, portal))
    # east end lancets on every level
    east = []
    for st in storeys:
        sill = 2 if n_st == 1 else 1
        for off in (-3, 3):
            at = (D - win_w) // 2 + off
            if 0 <= at and at + win_w <= D:
                east.append(opening("window", at, win_w, win_h,
                                    st * sh + sill))
    walls.append(wall(W - 1, 0, W - 1, D - 1, storeys, east))

    # ---- internal transept walls with tall arches (cruciform cut) ----
    arch_w = rng.choice([3, 3, 5])
    at = (D - 2 * m - arch_w) // 2
    for tx in (tx0, tx1):
        ops = [opening("door", at, arch_w, total_h - 4, 1)]
        walls.append(wall(tx, zN, tx, zS, storeys, ops))
    # nave flanks: arcade walls closing the cross arms off the chapels
    for z in (nz0, nz1):
        for xa, xbnd in ((1, tx0), (tx1, W - 2)):
            steps = xbnd - xa
            ops, off = [], 2
            while off + 2 <= steps + 1:
                ops.append(opening("door", off, 2, total_h - 4, 1))
                off += 4
            walls.append(wall(xa, z, xbnd, z, storeys, ops))

    # ---- buttresses + corner piers -----------------------------------
    for k in xb:
        for st in storeys:
            walls.append(wall(k, 0, k, zN, [st]))
            walls.append(wall(k, zS, k, D - 1, [st]))
    for cx, cz in ((0, zN), (0, zS), (W - 1, zN), (W - 1, zS)):
        walls.append(column(cx, cz, storeys))

    # ---- cruciform room tiling --------------------------------------
    rooms = [
        room("nave", "nave", (1, nz0, tx0, nz1), storeys),
        room("crossing", "crossing", (tx0, nz0, tx1, nz1), storeys),
        room("chancel", "chancel", (tx1, nz0, W - 1, nz1), storeys),
        room("transept_n", "transept", (tx0, iz0, tx1, nz0), storeys),
        room("transept_s", "transept", (tx0, nz1, tx1, iz1), storeys),
        room("chapel_nw", "chapel", (1, iz0, tx0, nz0), storeys),
        room("chapel_sw", "chapel", (1, nz1, tx0, iz1), storeys),
        room("chapel_ne", "chapel", (tx1, iz0, W - 1, nz0), storeys),
        room("chapel_se", "chapel", (tx1, nz1, W - 1, iz1), storeys),
    ]
    floors = [slab(0, (0, 0, W, D))] + [slab(s, (0, 0, W, D))
                                        for s in range(1, n_st)]
    sk = build(f"gothic_church_s{seed}", sh, W, D, rooms, walls, floors,
               roof("gable", pitch, "x", overhang=1),
               {"wall": "mineral", "floor": "mineral", "roof": "mineral",
                "frame": "mineral"})
    params = {"seed": seed, "storey_height": sh, "storeys": n_st,
              "length": L, "width": Wd, "roof": "gable", "pitch": pitch,
              "bays": n_bays,
              "door_width": door_w, "window": [win_w, win_h],
              "arch_width": arch_w, "nave_half_width": half,
              "transept_walls": [tx0, tx1], "footprint": [W, D]}
    return sk, params
