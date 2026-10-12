"""Assembly v3: style-anchored districts (author portfolios) + type roles +
material palette normalisation, on the rom1504 building library.

- each district draws buildings from its anchor author(s) -> style unity
- type roles: buildings at slots, vehicles near water/docks, statues at
  plazas, vegetation items replace part of the generic trees
- palette normalisation: remap material classes to the district's palette

Outputs: out/world/<name>/world_full.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from assemble import (AIR, COBBLE, FENCE, GRAVEL, GRASS, LEAVES, LOG,  # noqa: E402,F401
                      PLANKS, STONE, WATER, clip_stamp, dead_tree,
                      lighthouse, oak, pine, watchtower)
from features import class_of  # noqa: E402

CLASS_ORDER = ["wood", "metal", "plant", "glass", "colourful", "liquid",
               "mineral"]


def grade_ring(world, Ht, x0, x1, z0, z1, base, rng):  # noqa: ANN001
    """Cut-and-fill apron: terrain ramps to the building base within 3 cells,
    topped with grass and occasional shrubs so the contact reads as graded
    ground instead of a hard foundation edge."""
    W, Hy, D = world.shape
    for x in range(x0 - 3, x1 + 3):
        for z in range(z0 - 3, z1 + 3):
            if not (0 <= x < W and 0 <= z < D):
                continue
            if x0 <= x < x1 and z0 <= z < z1:
                continue
            d = max(x0 - x, x - x1 + 1, z0 - z, z - z1 + 1, 1)
            h = int(Ht[x, z])
            nh = min(max(h, base - d), base + d)
            if nh > h:
                world[x, h + 1:nh + 1, z] = STONE
            elif nh < h:
                world[x, nh + 1:h + 1, z] = AIR
            if nh != h:
                Ht[x, z] = nh
                world[x, nh, z] = GRASS
                if h - nh >= 3 and rng.random() < 0.35:
                    world[x, min(nh + 1, Hy - 1), z] = LEAVES


def apply_palette(grid: np.ndarray, palette: dict[str, int]) -> np.ndarray:
    out = grid.copy()
    for bid in np.unique(grid):
        if bid == 0:
            continue
        cls = class_of(int(bid))
        if cls in palette:
            out[grid == bid] = palette[cls]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_dir")
    ap.add_argument("--lib", default="out_lib")
    args = ap.parse_args()
    rng = np.random.default_rng(2024)

    world = np.load(os.path.join(args.world_dir, "world.npz"))["world"]
    t = np.load(os.path.join(args.world_dir, "terrain.npz"))
    Ht = t["height"].astype(int)
    biome = t["biome"]
    layout = json.load(open(os.path.join(args.world_dir, "layout.json"),
                            encoding="utf-8"))
    W, Hy, D = world.shape
    wl = layout["spec"]["water_level"]

    meta = [json.loads(l) for l in
            open(os.path.join(args.lib, "meta.jsonl"), encoding="utf-8")]
    types = json.load(open(os.path.join(args.lib, "types2.json")))
    vt_path = os.path.join(args.lib, "vision_types.json")
    vtypes = json.load(open(vt_path)) if os.path.exists(vt_path) else {}
    skel_path = os.path.join(args.lib, "skeleton.json")
    skel = json.load(open(skel_path)) if os.path.exists(skel_path) else {}
    ok_styles = {"modern", "medieval", "asian", "scifi", "western", "nautical",
                 "other"}
    style_of = {int(k): (v.get("style") if v.get("style") in ok_styles
                         else "other")
                for k, v in vtypes.items()}
    ov_path = os.path.join(args.lib, "authors", "type_overrides.json")
    overrides = json.load(open(ov_path)) if os.path.exists(ov_path) else {}
    tags_of: dict[int, str] = {}
    for k, v in overrides.items():
        if ":" in v:
            t, tag = v.split(":", 1)
            tags_of[int(k)] = tag
        else:
            t = v
        types[k] = t
    emb = np.load(os.path.join(args.lib, "embeddings.npz"))["Z32"]
    Zn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    by_url = {e["url"]: e for e in
              json.load(open("data/schematicsWithFinalUrl.json",
                             encoding="utf-8")) if isinstance(e, dict)}
    user_of = {r["idx"]: by_url.get(r.get("url", ""), {}).get("user", "")
               for r in meta}

    def pool_for(dist, kind):  # noqa: ANN001, ANN202
        authors = set(dist.get("authors", []))
        styles = set(dist.get("styles", []))
        out = []
        for r in meta:
            if types[str(r["idx"])] != kind:
                continue
            if authors and user_of[r["idx"]] not in authors:
                continue
            # style steering: drop labelled mismatches, keep unlabelled
            if (styles and r["idx"] in style_of
                    and style_of[r["idx"]] not in styles):
                continue
            dx, dy, dz = r["dims"]
            if kind == "building" and (max(dx, dz) > 60 or dy > 48):
                continue
            if kind in ("vehicle", "statue", "vegetation") and max(dx, dz) > 30:
                continue
            out.append(r["idx"])
        return out

    occ = np.zeros((W, D), dtype=bool)

    # ---- landmarks ------------------------------------------------------
    for lm in layout["landmarks"]:
        g = lighthouse() if lm["type"] == "lighthouse" else watchtower()
        occ[max(lm["x"] - 6, 0):lm["x"] + 6, max(lm["z"] - 6, 0):lm["z"] + 6] = True
        clip_stamp(world, g, lm["x"], max(int(lm["y"]) - 1, 2), lm["z"])

    # ---- buildings (style-anchored, MMR-diverse) -----------------------
    placed = 0
    slot_pos = []
    used: set[int] = set()
    for slot in layout["slots"]:
        dist = next((d for d in layout["spec"]["districts"]
                     if d["type"] == slot["district"]), None)
        if dist is None:
            continue
        pool = [i for i in pool_for(dist, "building") if i not in used]
        if not pool:
            # author pool exhausted -> style-respecting fallback pool
            styles = set(dist.get("styles", []))
            pool = [r["idx"] for r in meta
                    if types[str(r["idx"])] == "building"
                    and r["idx"] not in used
                    and max(r["dims"][0], r["dims"][2]) <= 60
                    and r["dims"][1] <= 48
                    and (not styles or r["idx"] not in style_of
                         or style_of[r["idx"]] in styles)][:80]
        if not pool:
            continue
        # smaller buildings for island tops
        if slot.get("on_island"):
            pool = [i for i in pool
                    if max(meta[i]["dims"][0], meta[i]["dims"][2]) <= 50] or pool
        # preferred-tag items first (e.g. actual temples in the temple quarter)
        want = set(dist.get("tags", []))
        pref = [i for i in pool if tags_of.get(i) in want] or pool
        # then style-matched (vision labels)
        styles = set(dist.get("styles", []))
        if styles:
            pref = [i for i in pref if style_of.get(i) in styles] or pref
        # then enclosure preference (skeleton composite signal:
        # real rooms = sized interior + mostly continuous walls)
        enc_pref = dist.get("enclosure")

        def closed_ok(i):  # noqa: ANN001
            s = skel.get(str(i), {})
            return (s.get("interior_voxels", 0) >= 200
                    and s.get("wall_gap", 1.0) <= 0.6)
        if enc_pref == "closed":
            pref = [i for i in pref if closed_ok(i)] or pref
        elif enc_pref == "open":
            pref = [i for i in pref if not closed_ok(i)] or pref
        # MMR: farthest from what this district already placed
        placed_here = [i2 for s2, x2, z2, i2 in slot_pos
                       if s2["district"] == slot["district"]]
        if placed_here:
            sims = (Zn[pref] @ Zn[placed_here].T).max(axis=1)
            order = np.argsort(sims)
            i = int(pref[order[int(rng.integers(0, min(3, len(order))))]])
        else:
            i = int(rng.choice(pref))
        used.add(i)
        g = np.load(os.path.join(args.lib, meta[i]["file"]))["grid"]
        g = apply_palette(g, dist.get("palette", {}))
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
        if slot.get("on_island"):
            base = max(int(slot["y"]), 2)
        if base + dy > Hy - 2:
            continue
        for fx in range(x0 - 1, x1 + 1):
            for fz in range(z0 - 1, z1 + 1):
                if slot.get("on_island"):
                    occ[fx, fz] = True
                    continue
                # skirt capped at 10 deep: steep terrain must not grow pillars
                fill_from = max(int(Ht[fx, fz]) + 1, base - 10)
                if fill_from < base + 1:
                    world[fx, fill_from:base + 1, fz] = STONE
                elif Ht[fx, fz] > base:
                    world[fx, base + 1:Ht[fx, fz] + 1, fz] = AIR
                occ[fx, fz] = True
        clip_stamp(world, g, x, base, z)
        if not slot.get("on_island"):
            grade_ring(world, Ht, x0 - 1, x1 + 1, z0 - 1, z1 + 1, base, rng)
        slot_pos.append((slot, x, z, i))
        placed += 1
    print(f"[a3] buildings placed: {placed}", flush=True)

    # ---- streets (MST per district) ------------------------------------
    by_dist: dict[str, list[tuple[int, int]]] = {}
    for slot, x, z, _bi in slot_pos:
        by_dist.setdefault(slot["district"], []).append((x, z))
    streets = 0
    for pts in by_dist.values():
        if len(pts) < 2:
            continue
        connected, rest = [pts[0]], pts[1:]
        while rest:
            best = min(((abs(a - c) + abs(b - d), k)
                        for k, (c, d) in enumerate(rest)
                        for (a, b) in connected), default=None)
            if best is None:
                break
            _, k = best
            c, d = rest.pop(k)
            corner = (c, connected[0][1]) if rng.random() < 0.5 else (connected[0][0], d)
            for (px, pz) in [(int(round(a)), int(round(b)))
                             for a, b in np.linspace((c, d), corner, 12)]:
                for ox in (-1, 0, 1):
                    for oz in (-1, 0, 1):
                        xx, zz = px + ox, pz + oz
                        if 0 <= xx < W and 0 <= zz < D and Ht[xx, zz] > wl:
                            world[xx, Ht[xx, zz], zz] = GRAVEL
                            occ[xx, zz] = True
                            streets += 1
            connected.append((c, d))

    # ---- props: vehicles at water, statues at plazas --------------------
    props = 0
    ind = next((d for d in layout["spec"]["districts"]
                if d["type"] == "industry"),
               layout["spec"]["districts"][0])
    vp = pool_for(ind, "vehicle") or pool_for(ind, "building")
    water_idx = np.nonzero((Ht < wl))
    for k in range(6):
        if not vp:
            break
        j = int(rng.integers(0, len(water_idx[0])))
        x, z = int(water_idx[0][j]), int(water_idx[1][j])
        i = int(rng.choice(vp))
        g = np.load(os.path.join(args.lib, meta[i]["file"]))["grid"]
        g = apply_palette(g, ind.get("palette", {}))
        g = np.rot90(g, k=int(rng.integers(0, 4)), axes=(0, 2))
        if clip_stamp(world, g, x, max(wl - 2, 2), z):
            props += 1

    for dist in layout["spec"]["districts"]:
        sp = pool_for(dist, "statue")
        if not sp:
            continue
        cx, cz = (np.asarray(dist["center"]) *
                  np.asarray(layout["spec"]["size"])).astype(int)
        for k in range(3):
            x = int(np.clip(cx + rng.integers(-40, 40), 4, W - 5))
            z = int(np.clip(cz + rng.integers(-40, 40), 4, D - 5))
            if Ht[x, z] <= wl or occ[x, z]:
                continue
            i = int(rng.choice(sp))
            g = np.load(os.path.join(args.lib, meta[i]["file"]))["grid"]
            g = apply_palette(g, dist.get("palette", {}))
            if clip_stamp(world, g, x, int(Ht[x, z]), z):
                occ[max(x - 4, 0):x + 5, max(z - 4, 0):z + 5] = True
                props += 1
    print(f"[a3] props placed: {props} (street cells {streets})", flush=True)

    # ---- trees + vegetation assets -------------------------------------
    planted = 0
    veg_pool = [r["idx"] for r in meta
                if types[str(r["idx"])] == "vegetation"
                and max(r["dims"][0], r["dims"][2]) <= 18]
    centers = [np.asarray(d["center"]) * np.asarray(layout["spec"]["size"])
               for d in layout["spec"]["districts"]]
    radii = [d["radius"] * layout["spec"]["size"][0] * 1.25
             for d in layout["spec"]["districts"]]
    for (tx, tz, ty) in layout["trees"]:
        if occ[tx, tz]:
            continue
        if occ[max(tx - 7, 0):tx + 8, max(tz - 7, 0):tz + 8].any():
            continue
        in_district = any((tx - c[0]) ** 2 + (tz - c[1]) ** 2 < r * r
                          for c, r in zip(centers, radii))
        if rng.random() < (0.75 if in_district else 0.3):
            continue
        if veg_pool and rng.random() < 0.05:
            i = int(rng.choice(veg_pool))
            g = np.load(os.path.join(args.lib, meta[i]["file"]))["grid"]
            g = np.rot90(g, k=int(rng.integers(0, 4)), axes=(0, 2))
        else:
            g = (oak if biome[tx, tz] == 3 else
                 (pine if rng.random() < 0.15 else dead_tree))(rng)
        if clip_stamp(world, g, tx, int(ty), tz):
            planted += 1
    print(f"[a3] trees planted: {planted}", flush=True)

    np.savez_compressed(os.path.join(args.world_dir, "world_full.npz"),
                        world=world)
    layout["assembled_v3"] = {"buildings": placed, "props": props,
                              "trees": planted}
    json.dump(layout, open(os.path.join(args.world_dir, "layout.json"), "w",
                           encoding="utf-8"), ensure_ascii=False)
    print("[a3] saved world_full.npz", flush=True)


if __name__ == "__main__":
    main()
