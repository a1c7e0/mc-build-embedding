"""Minimal road + path overlay on an existing generated world.

Post-processes world.npz + terrain.npz to add:
- Roads: straight segments, flat (terrain-following via clamp)
- Paths: building-to-road connectors, width 2

No RoadGraph, no topology, no auto-routing. Coordinates are explicit.

usage: python src/roads.py <world_dir> [--config roads.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

# legacy block IDs
AIR, STONE, GRASS, DIRT = 0, 1, 2, 3
GRAVEL, SANDSTONE = 13, 24
WATER = 9
PLANKS, LOG = 5, 17
PATH_BLOCK = GRAVEL  # road surface material


def add_road_segment(world, Ht, x0, z0, x1, z1, width=3, water_level=28):
    """Lay a road segment along a straight line (Bresenham-style).
    Road surface = PATH_BLOCK at terrain height (clamped to water_level+1).
    Skips columns where buildings already exist (non-terrain blocks).
    Returns number of blocks placed."""
    n = 0
    dx, dz = abs(x1 - x0), abs(z1 - z0)
    steps = max(dx, dz, 1)
    for i in range(steps + 1):
        t = i / steps
        cx = int(round(x0 + (x1 - x0) * t))
        cz = int(round(z0 + (z1 - z0) * t))
        for wx in range(cx - width // 2, cx + width // 2 + 1):
            for wz in range(cz - width // 2, cz + width // 2 + 1):
                if 0 <= wx < world.shape[0] and 0 <= wz < world.shape[2]:
                    # Skip if building already exists here (non-terrain block above ground)
                    h = int(Ht[wx, wz])
                    if h < water_level:
                        h = water_level
                    # Check for building blocks above terrain (non-air, non-terrain)
                    # Terrain blocks: GRASS(2), DIRT(3), STONE(1), SANDSTONE(24), etc.
                    # Building blocks: PLANKS(5), LOG(17), LEAVES(18), GLASS(20), COBBLE(98), etc.
                    has_building = False
                    for y in range(h + 1, min(h + 30, world.shape[1])):
                        bid = world[wx, y, wz]
                        if bid not in (0, 1, 2, 3, 9, 12, 13, 24, 60):  # terrain IDs
                            has_building = True
                            break
                    if has_building:
                        continue  # skip this column - building here
                    # Place road
                    world[wx, h + 1:, wz] = AIR
                    world[wx, h, wz] = PATH_BLOCK
                    for dy in range(1, 3):
                        if h - dy >= 0 and world[wx, h - dy, wz] == AIR:
                            world[wx, h - dy, wz] = DIRT
                    n += 1
    return n


def add_path_segment(world, Ht, x0, z0, x1, z1, width=2, water_level=28):
    """Lay a path (narrower than road) connecting building to road."""
    return add_road_segment(world, Ht, x0, z0, x1, z1,
                            width=width, water_level=water_level)


def world_hash(world):
    """Deterministic hash of world grid for reproducibility check."""
    return hashlib.sha256(world.tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_dir")
    ap.add_argument("--config", default=None,
                    help="JSON config with roads and paths (default: auto-generate)")
    ap.add_argument("--out", default=None, help="output suffix (default: overwrite)")
    args = ap.parse_args()

    wd = args.world_dir
    # Use world_full.npz (with buildings) if available, else world.npz
    world_file = os.path.join(wd, "world_full.npz")
    if not os.path.exists(world_file):
        world_file = os.path.join(wd, "world.npz")
    terrain_file = os.path.join(wd, "terrain.npz")
    world = np.load(world_file)["world"].copy()
    t = np.load(terrain_file)
    Ht = t["height"].astype(int).copy()
    layout = json.load(open(os.path.join(wd, "layout.json"), encoding="utf-8"))
    wl = layout["spec"]["water_level"]

    W, Hy, D = world.shape

    # Load or auto-generate road config
    if args.config and os.path.exists(args.config):
        config = json.load(open(args.config, encoding="utf-8"))
    else:
        # Auto-generate: pick 6 market buildings, draw a main road + branches
        slots = [s for s in layout["slots"] if s["district"] == "market"][:6]
        if len(slots) < 3:
            slots = layout["slots"][:6]
        # Main road: horizontal through the market area
        xs = [s["x"] for s in slots]
        zs = [s["z"] for s in slots]
        road_z = int(np.median(zs))
        road_x0, road_x1 = min(xs) - 20, max(xs) + 20
        roads = [{"from": [road_x0, road_z], "to": [road_x1, road_z],
                  "width": 3, "type": "main"}]
        # Branch roads: vertical from each building to main road
        paths = []
        for i, s in enumerate(slots):
            paths.append({"building": i, "bx": s["x"], "bz": s["z"],
                          "to_road_z": road_z, "width": 2})
        config = {"roads": roads, "paths": paths,
                  "buildings": [{"x": s["x"], "z": s["z"]} for s in slots]}
        # Save auto-generated config
        cfg_path = os.path.join(wd, "roads_config.json")
        json.dump(config, open(cfg_path, "w"), indent=1)
        print(f"[roads] auto-generated config: {cfg_path}")
        print(f"[roads] {len(roads)} roads, {len(paths)} paths, "
              f"{len(config['buildings'])} buildings")

    # Place roads
    n_road = 0
    for r in config["roads"]:
        x0, z0 = r["from"]
        x1, z1 = r["to"]
        w = r.get("width", 3)
        n = add_road_segment(world, Ht, x0, z0, x1, z1, width=w, water_level=wl)
        n_road += n
        print(f"[roads] road ({x0},{z0})->({x1},{z1}) w={w}: {n} blocks")

    # Place paths
    n_path = 0
    for p in config.get("paths", []):
        bx, bz = p["bx"], p["bz"]
        rz = p.get("to_road_z", bz)
        w = p.get("width", 2)
        n = add_path_segment(world, Ht, bx, bz, bx, rz, width=w, water_level=wl)
        n_path += n
        print(f"[roads] path ({bx},{bz})->({bx},{rz}) w={w}: {n} blocks")

    # Save
    out_suffix = args.out or ""
    out_world = os.path.join(wd, f"world_roads{out_suffix}.npz")
    np.savez_compressed(out_world, world=world)
    h = world_hash(world)
    print(f"[roads] total: {n_road} road + {n_path} path blocks")
    print(f"[roads] world hash: {h}")
    print(f"[roads] saved: {out_world}")

    # Also save hash for reproducibility check
    hash_file = os.path.join(wd, f"world_hash{out_suffix}.txt")
    with open(hash_file, "w") as f:
        f.write(h)
    return h


if __name__ == "__main__":
    main()
