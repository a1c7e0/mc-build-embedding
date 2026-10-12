"""nautical_dock style grammar -> Skeleton-DSL.

Signature features (construction mapping, see src/stylegen/README.md):

* 栈桥 pier: the whole wharf is a raised wooden deck (fill rows at
  deck_h=2-3) standing on rows of pilings, with loading doors opening at
  deck level.
* 仓房 warehouse: a big storage box with 3-5 wide loading doors and small
  high windows; plan ``end_warehouse`` puts it at one end of the pier,
  plan ``shed_strip`` runs it as a long transit shed along one side.
* shed 顶: plan ``shed_strip`` uses roof.kind="shed" (the compiler
  rasterises the shed over the low half of the cross span, which is
  exactly the covered strip of the transit shed; the far half is the open
  loading deck). ``end_warehouse`` uses a low-pitch gable over the whole
  pier. The compiler's gable-end infill always sits at x=0/W-1, so the
  end walls are placed on the grid edge and the open deck is closed by
  deck-height railings that meet the infill.
* 大坡道: a stepped gangway of 2-deep strips descending from the deck to
  ground level on the open side (centred notch on end_warehouse, full
  width on shed_strip).
* 木平台: wall/floor/roof all wood.

Write order matters (the compiler rasterises wall entries in list order,
each entry rewriting its cells): fills first, then main walls, then
piers; a later fill must carry a superset of the earlier bands at any
shared cell (posts/railings keep the deck block).
"""
from __future__ import annotations

import random

from .dsl import build, fill_cell, fill_rect, opening, room, roof, wall

NAME = "nautical_dock"


def generate(seed: int):
    rng = random.Random(seed)
    n_st = rng.choice([1, 1, 1, 2])
    sh = rng.choice([7, 8]) if n_st == 1 else rng.choice([6, 7])
    L = rng.choice([21, 23, 25, 27, 29])
    Wd = rng.choice([11, 13, 15])
    deck_h = rng.choice([2, 3])
    pitch = rng.choice([2, 3])
    door_w = rng.choice([3, 5])
    plan = rng.choice(["end_warehouse", "shed_strip", "shed_strip"])

    W, D = L, Wd
    storeys = list(range(n_st))
    ytop = sh * n_st
    walls, fills = [], []
    ramp_len = 2 * deck_h
    ramp_z0 = D - ramp_len
    door_y = deck_h + 1
    door_h = min(4, sh - door_y)
    win_y, win_h = deck_h + 2, 2
    deck_bands = [(deck_h, deck_h + 1)] + \
        [(s * sh - 1, s * sh) for s in range(1, n_st)]

    def win_ops(st: int, steps: int, gap: int = 3, wid: int = 2):
        ops = []
        at = 2
        while at + wid <= steps:
            y = win_y if st == 0 else 2
            h = win_h if st == 0 else 2
            if y + h <= sh:
                ops.append(opening("window", at, wid, h, y))
            at += gap
        return ops

    if plan == "end_warehouse":
        kind = "gable"
        wh_w = int(W * rng.uniform(0.3, 0.45))
        rw = min(rng.choice([5, 7, 9]), W - 6)
        rx0, rx1 = (W - rw) // 2, (W - rw) // 2 + rw

        for st in storeys:
            walls.append(wall(0, 0, 0, D - 1, [st], win_ops(st, D - 1)))
            walls.append(wall(W - 1, 0, W - 1, D - 1, [st],
                              win_ops(st, D - 1)))
            walls.append(wall(0, 1, wh_w, 1, [st], win_ops(st, wh_w)))
            walls.append(wall(0, D - 2, wh_w, D - 2, [st], win_ops(st, wh_w)))
            ops = [opening("door", (D - 3 - door_w) // 2, door_w,
                           door_h, door_y)]
            walls.append(wall(wh_w, 1, wh_w, D - 2, [st], ops))

        fills += fill_rect((0, 0, W, ramp_z0), deck_bands, sh)
        fills += fill_rect((0, ramp_z0, rx0, D), deck_bands, sh)
        fills += fill_rect((rx1, ramp_z0, W, D), deck_bands, sh)
        for j in range(deck_h):
            z0 = ramp_z0 + 2 * j
            fills += fill_rect((rx0, z0, rx1, z0 + 2),
                               [(0, deck_h - j)], sh)
        # south deck edge: railing runs broken by full-height mooring posts
        for x in range(0, W):
            if rx0 <= x < rx1:
                continue
            bands = [(0, deck_h + 3)] if x % 3 == 0 else [(deck_h, deck_h + 3)]
            fills += fill_cell(x, ramp_z0 - 1, bands, sh)
        # roof-bearing colonnade over the pier + pilings below the deck
        for px in range(wh_w + 1, W - 1, rng.choice([3, 4])):
            for pz in (1, D - 2):
                if pz >= ramp_z0 and rx0 <= px < rx1:
                    continue
                walls.append(wall(px, pz, px, pz, storeys))
            fills += fill_cell(px, D // 2, [(0, deck_h + 1)], sh)
        rooms = [room("hold", "warehouse", (1, 2, wh_w, D - 2), storeys)]
    else:
        kind = "shed"
        mid = D // 2
        mid_w = mid - 1                       # transit shed's deck-side wall
        for st in storeys:
            walls.append(wall(0, 0, 0, mid_w, [st], win_ops(st, mid_w)))
            walls.append(wall(W - 1, 0, W - 1, mid_w, [st],
                              win_ops(st, mid_w)))
            walls.append(wall(0, 1, W - 1, 1, [st], win_ops(st, W - 1)))
            ops = [opening("door", at, door_w, door_h, door_y)
                   for at in (3, W - 6 - door_w) if at >= 0]
            walls.append(wall(0, mid_w, W - 1, mid_w, [st], ops))

        fills += fill_rect((1, 0, W - 1, ramp_z0), deck_bands, sh)
        # deck-height railings close the open deck against the gable infill
        fills += fill_rect((0, mid, 1, D), [(deck_h, ytop)], sh)
        fills += fill_rect((W - 1, mid, W, D), [(deck_h, ytop)], sh)
        for j in range(deck_h):
            z0 = ramp_z0 + 2 * j
            fills += fill_rect((1, z0, W - 1, z0 + 2), [(0, deck_h - j)], sh)
        for px in range(2, W - 2, 3):
            fills += fill_cell(px, ramp_z0 - 1, [(0, deck_h + 1)], sh)
        rooms = [room("hold", "warehouse", (1, 2, W - 1, mid_w), storeys)]

    # the raised wharf deck IS the ground floor (emitted as fill rows);
    # only inter-storey slabs are declared as DSL floors.
    floors = [{"storey": s, "rect": [0, 0, W, D]}
              for s in range(1, n_st)]
    sk = build(f"nautical_dock_s{seed}", sh, W, D, rooms, fills + walls,
               floors, roof(kind, pitch, "x", overhang=1),
               {"wall": "wood", "floor": "wood", "roof": "wood",
                "frame": "wood"})
    params = {"seed": seed, "plan": plan, "storey_height": sh,
              "storeys": n_st, "deck_height": deck_h, "roof": kind,
              "pitch": pitch, "door_width": door_w, "ramp_steps": deck_h,
              "footprint": [W, D]}
    return sk, params
