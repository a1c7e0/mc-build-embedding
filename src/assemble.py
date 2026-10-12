"""Assembly pass v2: buildings, streets, trees, docks, landmarks.

- buildings are stamped from the library (PCA-cluster retrieval per district)
  with a stone skirt so they never float on rolling terrain
- streets: per-district MST over the building slots, gravel, width 2
- docks: plank piers into the water near the industry district
- landmarks first (so trees never grow on top of them)

Outputs: out/world/<name>/world_full.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

AIR, STONE, GRASS, DIRT, SAND, WATER = 0, 1, 2, 3, 12, 9
LOG, LEAVES, PLANKS = 17, 18, 5
COBBLE, STONE_BRICK, GLOWSTONE, FENCE, BRICK = 4, 98, 89, 85, 45
QUARTZ, RED_WOOL, GLASS, GRAVEL = 155, 35, 20, 13


def clip_stamp(world, g, x, y, z):
    W, Hy, D = world.shape
    dx, dy, dz = g.shape
    x0, x1 = x - dx // 2, x - dx // 2 + dx
    z0, z1 = z - dz // 2, z - dz // 2 + dz
    y1 = y + dy
    if x0 < 0 or z0 < 0 or x1 > W or z1 > D or y < 0 or y1 > Hy:
        return False
    sub = world[x0:x1, y:y1, z0:z1]
    m = g != 0
    sub[m] = g[m]
    return True


# ------------------------------------------------------------- builders ---
def oak(rng):
    h = int(rng.integers(4, 7))
    g = np.zeros((9, h + 5, 9), np.uint16)
    for ly in range(h - 2, h + 2):
        w = 3 if ly < h else 2
        g[4 - w:5 + w, ly, 4 - w:5 + w] = LEAVES
    g[4, :h, 4] = LOG
    return g


def pine(rng):
    h = int(rng.integers(7, 11))
    g = np.zeros((7, h + 2, 7), np.uint16)
    g[3, :h, 3] = LOG
    y, w = h - 5, 3
    while y < h + 1 and w >= 1:
        g[3 - w:4 + w, y, 3 - w:4 + w] = LEAVES
        y += 1
        w -= 1
    g[3, :h, 3] = LOG
    return g


def dead_tree(rng):
    h = int(rng.integers(3, 6))
    g = np.zeros((5, h + 1, 5), np.uint16)
    g[2, :h, 2] = LOG
    if rng.random() < 0.5:
        g[2:4, h - 2, 2] = LOG
    return g


def lighthouse():
    g = np.zeros((11, 26, 11), np.uint16)
    for y in range(24):
        w = 4 if y < 6 else (3 if y < 18 else 2)
        ring = np.ones((2 * w + 1, 2 * w + 1), bool)
        ring[1:-1, 1:-1] = False
        col = QUARTZ if (y // 3) % 2 == 0 else RED_WOOL
        g[5 - w:6 + w, y, 5 - w:6 + w][ring] = col
        if y == 23:
            g[5 - w:6 + w, y, 5 - w:6 + w][:] = GLOWSTONE
    g[3:8, 24, 3:8] = FENCE
    g[5, 25, 5] = GLOWSTONE
    g[4:7, 0, 4:7] = STONE_BRICK
    return g


def watchtower():
    g = np.zeros((9, 18, 9), np.uint16)
    for (lx, lz) in ((1, 1), (7, 1), (1, 7), (7, 7)):
        g[lx, :12, lz] = LOG
    g[1:8, 12, 1:8] = PLANKS
    g[1:8, 13, 1] = PLANKS
    g[1:8, 13, 7] = PLANKS
    g[1, 13, 1:8] = PLANKS
    g[7, 13, 1:8] = PLANKS
    g[1:8, 14, 1:8] = PLANKS
    g[0:9, 15, 0:9] = COBBLE
    g[1:8, 16, 1:8] = COBBLE
    g[3:6, 17, 3:6] = COBBLE
    g[2, 12, 2] = GLOWSTONE
    g[6, 12, 6] = GLOWSTONE
    return g


# ------------------------------------------------------------------ main ---
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_dir")
    ap.add_argument("--data", default="out")
    args = ap.parse_args()
    rng = np.random.default_rng(1234)

    world = np.load(os.path.join(args.world_dir, "world.npz"))["world"]
    t = np.load(os.path.join(args.world_dir, "terrain.npz"))
    Ht = t["height"].astype(int)
    biome = t["biome"]
    layout = json.load(open(os.path.join(args.world_dir, "layout.json"),
                            encoding="utf-8"))
    W, Hy, D = world.shape
    wl = layout["spec"]["water_level"]

    meta = [json.loads(l) for l in
            open(os.path.join(args.data, "meta.jsonl"), encoding="utf-8")]
    cid = np.load(os.path.join(args.data, "embeddings.npz"))["cid"]
    by_cluster: dict[int, list[int]] = {}
    for i, c in enumerate(cid.tolist()):
        dims = meta[i]["dims"]
        if max(dims[0], dims[2]) <= 30 and dims[1] <= 34:
            by_cluster.setdefault(int(c), []).append(i)

    occ = np.zeros((W, D), dtype=bool)

    # ---- landmarks first (nothing may grow on them) ---------------------
    for lm in layout["landmarks"]:
        g = lighthouse() if lm["type"] == "lighthouse" else watchtower()
        r = 6
        occ[max(lm["x"] - r, 0):lm["x"] + r,
            max(lm["z"] - r, 0):lm["z"] + r] = True
        if clip_stamp(world, g, lm["x"], max(int(lm["y"]) - 1, 2), lm["z"]):
            print(f"[assemble] landmark {lm['type']} at ({lm['x']},{lm['z']})",
                  flush=True)

    # ---- buildings with a stone skirt ----------------------------------
    placed = 0
    slot_pos = []
    for slot in layout["slots"]:
        clusters = [c for c in slot["clusters"] if by_cluster.get(c)]
        if not clusters:
            continue
        c = int(rng.choice(clusters))
        i = int(rng.choice(by_cluster[c]))
        g = np.load(os.path.join(args.data, meta[i]["file"]))["grid"]
        g = np.rot90(g, k=int(rng.integers(0, 4)), axes=(0, 2))
        dx, dy, dz = g.shape
        x, z = slot["x"], slot["z"]
        x0, x1 = x - dx // 2, x - dx // 2 + dx
        z0, z1 = z - dz // 2, z - dz // 2 + dz
        if x0 < 3 or z0 < 3 or x1 > W - 3 or z1 > D - 3:
            continue
        if occ[x0 - 1:x1 + 1, z0 - 1:z1 + 1].mean() > 0.15:
            continue
        base = max(int(np.median(Ht[x0:x1, z0:z1])) - 2, 2)
        if base + dy > Hy - 2:
            continue
        for fx in range(x0 - 1, x1 + 1):        # skirt ring included
            for fz in range(z0 - 1, z1 + 1):
                if Ht[fx, fz] < base:
                    world[fx, Ht[fx, fz] + 1:base + 1, fz] = STONE
                elif Ht[fx, fz] > base:
                    world[fx, base + 1:Ht[fx, fz] + 1, fz] = AIR
                occ[fx, fz] = True
        clip_stamp(world, g, x, base, z)
        slot_pos.append((slot, x, z))
        placed += 1
    print(f"[assemble] buildings placed: {placed}", flush=True)

    # ---- streets: MST over slots per district ---------------------------
    by_dist: dict[str, list[tuple[int, int]]] = {}
    for slot, x, z in slot_pos:
        by_dist.setdefault(slot["district"], []).append((x, z))
    streets = 0
    for pts in by_dist.values():
        if len(pts) < 2:
            continue
        connected = [pts[0]]
        rest = pts[1:]
        while rest:
            best = min(((abs(a - c) + abs(b - d), k)
                        for k, (c, d) in enumerate(rest)
                        for (a, b) in connected), default=None)
            if best is None:
                break
            _, k = best
            (c, d) = rest.pop(k)
            # L-path with a jittered corner
            corner = (c, connected[0][1]) if rng.random() < 0.5 else (connected[0][0], d)
            path = [(c, d), corner]
            for (px, pz) in [(int(np.round(a)), int(np.round(b)))
                             for a, b in np.linspace(path[0], path[1], 12)]:
                for ox in (-1, 0, 1):
                    for oz in (-1, 0, 1):
                        xx, zz = px + ox, pz + oz
                        if 0 <= xx < W and 0 <= zz < D and Ht[xx, zz] > wl:
                            world[xx, Ht[xx, zz], zz] = GRAVEL
                            occ[xx, zz] = True
                            streets += 1
            connected.append((c, d))
    print(f"[assemble] street cells: {streets}", flush=True)

    # ---- docks near the industry district -------------------------------
    dock_dir = None
    for dist in layout["spec"]["districts"]:
        if dist["type"] == "industry":
            dock_dir = dist
    if dock_dir:
        cx, cz = (np.asarray(dock_dir["center"]) *
                  np.asarray(layout["spec"]["size"])).astype(int)
        best = None
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            for r in range(10, 140, 4):
                px = int(cx + np.cos(ang) * r)
                pz = int(cz + np.sin(ang) * r)
                if not (0 < px < W - 2 and 0 < pz < D - 2):
                    continue
                if Ht[px, pz] < wl:          # found water
                    best = (px, pz, ang)
                    break
            if best:
                break
        if best:
            px, pz, ang = best
            # pier: deck at wl+1, posts below, 18 long into the water
            dx_, dz_ = np.cos(ang), np.sin(ang)
            for s in range(18):
                qx = int(px + dx_ * s)
                qz = int(pz + dz_ * s)
                if not (2 < qx < W - 3 and 2 < qz < D - 3):
                    break
                for ox, oz in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    xx, zz = qx + ox, qz + oz
                    if Ht[xx, zz] < wl:      # only over water/low ground
                        world[xx, wl + 1, zz] = PLANKS
                        world[xx, wl, zz] = LOG
                        if s % 5 == 0:
                            world[xx, wl + 2, zz] = FENCE
            # working-dock details: crates and barrels ON the deck, plus a
            # small moored boat alongside
            for s in range(4, 17, 4):
                qx = int(px + dx_ * s)
                qz = int(pz + dz_ * s)
                if 2 < qx < W - 3 and 2 < qz < D - 3:
                    world[qx, wl + 2, qz] = (PLANKS, COBBLE, BRICK)[s % 3]
                    if s % 8 == 4:
                        world[qx, wl + 3, qz] = COBBLE
            bx = int(px + dx_ * 10 - dz_ * 3)
            bz = int(pz + dz_ * 10 + dx_ * 3)
            if 3 < bx < W - 4 and 3 < bz < D - 4:
                for a in range(4):
                    world[bx + a, wl + 1, bz] = PLANKS
                    world[bx + a, wl, bz] = PLANKS
                world[bx + 1, wl + 2, bz] = FENCE
            print(f"[assemble] dock pier from ({px},{pz})", flush=True)

    # ---- trees ----------------------------------------------------------
    tree_builders = [oak, pine, dead_tree]
    planted = skipped = 0
    for (tx, tz, ty) in layout["trees"]:
        if occ[tx, tz]:
            continue
        if occ[max(tx - 4, 0):tx + 5, max(tz - 4, 0):tz + 5].any():
            skipped += 1
            continue
        if rng.random() < 0.25:
            skipped += 1
            continue
        pick = tree_builders[0 if biome[tx, tz] == 3 else
                             (1 if rng.random() < 0.3 else 2)]
        g = pick(rng)
        if clip_stamp(world, g, tx, int(ty), tz):
            planted += 1
    print(f"[assemble] trees planted: {planted} (skipped {skipped})",
          flush=True)

    np.savez_compressed(os.path.join(args.world_dir, "world_full.npz"),
                        world=world)
    layout["assembled"] = {"buildings": placed, "trees": planted,
                           "street_cells": streets}
    json.dump(layout, open(os.path.join(args.world_dir, "layout.json"), "w",
                           encoding="utf-8"), ensure_ascii=False)
    print("[assemble] saved world_full.npz", flush=True)


if __name__ == "__main__":
    main()
