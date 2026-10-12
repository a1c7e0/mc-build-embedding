"""Rule-based decoration: place furniture from ProcTHOR data into a
compiled skeleton shell.

Maps ProcTHOR assetId patterns to Minecraft block configurations,
then stamps them at the recorded positions (scaled to block grid).

usage: python src/decorate.py <skeleton.json> <furniture.jsonl line> <out.npz>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402

# legacy block IDs
AIR, STONE, GRASS, DIRT = 0, 1, 2, 3
PLANKS, LOG, LEAVES = 5, 17, 18
GLASS, SANDSTONE = 20, 24
WOOL_WHITE, WOOL_RED, WOOL_BLUE = 35, 35, 35  # simplified
QUARTZ = 155  # placeholder
IRON = 42
GOLD = 41
HAY = 170
WATER = 9

# Furniture pattern → (block_id, width, height, depth)
# Matches assetId prefix (case-insensitive)
FURNITURE_MAP = {
    "fridge":        (IRON,       1, 2, 1),
    "countertop":    (STONE,      2, 1, 1),
    "kitchencounter": (STONE,     2, 1, 1),
    "sink":          (STONE,      1, 1, 1),
    "stove":         (STONE,      1, 1, 1),
    "oven":          (STONE,      1, 1, 1),
    "microwave":     (IRON,       1, 1, 1),
    "dishwasher":    (IRON,       1, 1, 1),
    "houseplant":    (LEAVES,     1, 2, 1),
    "plant":         (LEAVES,     1, 2, 1),
    "sofa":          (PLANKS,     3, 1, 2),
    "couch":         (PLANKS,     3, 1, 2),
    "chair":         (PLANKS,     1, 1, 1),
    "table":         (PLANKS,     2, 1, 2),
    "desk":          (PLANKS,     2, 1, 1),
    "bed":           (PLANKS,     2, 1, 3),
    "toilet":        (STONE,      1, 1, 1),
    "bathtub":       (STONE,      1, 1, 2),
    "tv":            (GLASS,      2, 1, 1),
    "television":    (GLASS,      2, 1, 1),
    "bookshelf":     (PLANKS,     1, 2, 1),
    "shelf":         (PLANKS,     1, 2, 1),
    "cabinet":       (PLANKS,     1, 2, 1),
    "wardrobe":      (PLANKS,     1, 2, 1),
    "dresser":       (PLANKS,     2, 1, 1),
    "nightstand":    (PLANKS,     1, 1, 1),
    "lamp":          (GOLD,       1, 1, 1),
    "fan":           (IRON,       1, 1, 1),
    "curtain":       (PLANKS,     1, 2, 1),
    "rug":           (PLANKS,     2, 1, 2),
    "painting":      (PLANKS,     1, 1, 1),
    "mirror":        (GLASS,      1, 1, 1),
    "washer":        (IRON,       1, 1, 1),
    "dryer":         (IRON,       1, 1, 1),
    "washer/dryer":  (IRON,       1, 1, 1),
    "coffee":        (PLANKS,     1, 1, 1),
    "ottoman":       (PLANKS,     1, 1, 1),
    "stool":         (PLANKS,     1, 1, 1),
    "piano":         (PLANKS,     2, 1, 2),
    "fireplace":     (STONE,      2, 2, 1),
    "heater":        (IRON,       1, 1, 1),
    "ac":            (IRON,       1, 1, 1),
    "printer":       (IRON,       1, 1, 1),
    "computer":      (GLASS,      1, 1, 1),
    "monitor":       (GLASS,      1, 1, 1),
}


def match_furniture(asset_id: str) -> tuple[int, int, int, int]:
    """Match assetId to a block config. Returns (block_id, w, h, d)."""
    al = asset_id.lower()
    for key, cfg in FURNITURE_MAP.items():
        if key in al:
            return cfg
    # default: small wooden block
    return (PLANKS, 1, 1, 1)


def place_furniture(world, Ht, item, water_level=28):
    """Place one furniture item into the world grid."""
    bid, fw, fh, fd = match_furniture(item["assetId"])
    x = int(round(item["position"]["x"]))
    z = int(round(item["position"]["z"]))
    # For standalone building grids: find floor level (lowest solid layer)
    # For worlds: use terrain height
    W, Hy, D = world.shape
    if not (0 <= x < W and 0 <= z < D):
        return 0
    # Find the floor: the lowest y where there's a solid block in this column
    col = world[x, :, z]
    solid_ys = np.nonzero(col)[0]
    if len(solid_ys) == 0:
        y_base = water_level
    else:
        # Place furniture ON TOP of the floor slab (first solid layer)
        y_base = solid_ys[0]
    # If the column has walls (taller than 2), use floor instead of roof
    if len(solid_ys) > 2:
        y_base = solid_ys[0]  # floor slab

    placed = 0
    for dx in range(fw):
        for dy in range(fh):
            for dz in range(fd):
                px, py, pz = x + dx, y_base + 1 + dy, z + dz
                if 0 <= px < W and 0 <= py < Hy and 0 <= pz < D:
                    if world[px, py, pz] == AIR:
                        world[px, py, pz] = bid
                        placed += 1
    return placed


def main():
    if len(sys.argv) < 4:
        print("usage: python src/decorate.py <world.npz> <furniture.jsonl> <out.npz>")
        print("  furniture.jsonl: one house per line, each with 'objects' array")
        sys.exit(1)

    world_file = sys.argv[1]
    furn_file = sys.argv[2]
    out_file = sys.argv[3]

    # Load world grid (try 'world' then 'grid' key)
    npz = np.load(world_file)
    world = (npz["world"] if "world" in npz else npz["grid"]).copy()
    # Load terrain height
    terrain_file = os.path.join(os.path.dirname(world_file), "terrain.npz")
    if os.path.exists(terrain_file):
        Ht = np.load(terrain_file)["height"].astype(int)
    else:
        Ht = np.zeros(world.shape[:1] + world.shape[2:], dtype=int)
        for x in range(world.shape[0]):
            for z in range(world.shape[2]):
                col = world[x, :, z]
                solid = np.nonzero(col)[0]
                Ht[x, z] = solid[-1] if len(solid) else 0

    # Load furniture (one house)
    with open(furn_file, encoding="utf-8") as f:
        furn = json.loads(f.readline())

    items = furn.get("objects", [])
    print(f"[decorate] {len(items)} furniture items")

    total = 0
    placed_types = {}
    for item in items:
        n = place_furniture(world, Ht, item)
        total += n
        bid, _, _, _ = match_furniture(item["assetId"])
        placed_types[bid] = placed_types.get(bid, 0) + n

    print(f"[decorate] placed {total} blocks")
    print(f"[decorate] by type: {placed_types}")

    np.savez_compressed(out_file, world=world)
    print(f"[decorate] saved: {out_file}")


if __name__ == "__main__":
    main()
