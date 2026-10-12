"""Room-aware decoration: place furniture motifs based on room type.

Combines:
- MC motifs (245 multi-block furniture patterns)
- ProcTHOR room→furniture rules (bedroom→beds, kitchen→fridges)
- ProcTHOR placement rules (53% floor, 28% wall, 19% table)

For skeleton DSL output (with room rects), place room-appropriate furniture.

usage: python src/decorate_rooms.py <grid.npz> <skeleton.json> <style> <seed> <out.npz>
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402
from apply_motifs import load_motifs, place_motif  # noqa: E402

MOTIF = 3

# Room type → furniture categories (from ProcTHOR analysis)
ROOM_FURNITURE = {
    "Bedroom": ["Bed", "Dresser", "Chair", "Television", "Wardrobe",
                "Nightstand", "Table", "Houseplant", "Shelving"],
    "Kitchen": ["Countertop", "Fridge", "Stool", "Chair", "Shelving",
                "Dining", "Table", "GarbageBag", "bin"],
    "Bathroom": ["Toilet", "Sink", "Plunger", "bin", "Laundry",
                 "Washing", "Dresser", "Mirror"],
    "LivingRoom": ["Sofa", "Dining", "TV", "Television", "Chair",
                   "Armchair", "Houseplant", "Shelving", "Table", "Floor"],
    "room": ["Table", "Chair", "Houseplant", "Shelving", "Floor"],
}

# Furniture category → motif block types to look for
# (map ProcTHOR categories to our motif block IDs)
FURNITURE_BLOCK_HINTS = {
    "Bed": {35, 5, 17},           # wool + planks + log
    "Dresser": {5, 17, 47},       # planks + log + bookshelf
    "Chair": {5, 17, 53},         # planks + log + stairs
    "Table": {5, 17, 85},         # planks + log + fence
    "Sofa": {35, 5},              # wool + planks
    "Countertop": {1, 4, 98},     # stone + cobble + stonebrick
    "Fridge": {42, 1},            # iron + stone
    "Toilet": {1, 155},           # stone + quartz
    "Sink": {1, 155, 42},         # stone + quartz + iron
    "TV": {42, 20, 55},           # iron + glass + redstone
    "Television": {42, 20, 55},
    "Houseplant": {18, 2, 81},    # leaves + grass + cactus
    "Shelving": {5, 47, 17},      # planks + bookshelf + log
    "Lighting": {89, 50, 41},     # glowstone + torch + gold
    "Wall": {35, 20, 102},        # wool + glass + glasspane
    "Floor": {35, 24, 171},       # wool + sandstone + carpet
}


def get_room_at(x, z, rooms):
    """Find which room contains position (x,z)."""
    for room in rooms:
        rect = room.get("rect")
        if not rect:
            continue
        x0, z0, x1, z1 = rect
        if x0 <= x <= x1 and z0 <= z <= z1:
            return room.get("type", "room")
    return "room"


def filter_motifs_for_room(room_type, all_motifs):
    """Filter motifs that match the room's furniture categories.
    Returns motifs sorted by relevance (furniture block match score)."""
    categories = ROOM_FURNITURE.get(room_type, ROOM_FURNITURE["room"])
    target_blocks = set()
    for cat in categories:
        target_blocks.update(FURNITURE_BLOCK_HINTS.get(cat, set()))

    # Score each motif by how many target blocks it contains
    scored = []
    for motif in all_motifs:
        blocks = set(motif) - {0}
        score = len(blocks & target_blocks)
        scored.append((score, motif))

    # Sort by score (descending), keep those with at least 1 match
    scored.sort(key=lambda x: -x[0])
    relevant = [m for s, m in scored if s > 0]
    return relevant if relevant else all_motifs[:50]  # fallback


def find_interior_in_room(grid, room_type, rooms, room_rect):
    """Find interior air positions within a specific room."""
    x0, z0, x1, z1 = room_rect
    W, H, D = grid.shape
    positions = []
    for x in range(max(1, x0), min(W - 1, x1)):
        for y in range(1, H - 1):
            for z in range(max(1, z0), min(D - 1, z1)):
                if grid[x, y, z] != 0:
                    continue
                neighbors = sum(1 for dx, dy, dz in
                    [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                    if grid[x+dx, y+dy, z+dz] > 0)
                if neighbors >= 3:
                    positions.append((x, y, z))
    return positions


def main():
    if len(sys.argv) < 6:
        print("usage: python src/decorate_rooms.py <grid.npz> <skeleton.json> <style> <seed> <out.npz>")
        sys.exit(1)

    grid_file = sys.argv[1]
    skel_file = sys.argv[2]
    style = sys.argv[3]
    seed = int(sys.argv[4])
    out_file = sys.argv[5]

    rng = random.Random(seed)
    npz = np.load(grid_file)
    grid = (npz["world"] if "world" in npz else npz["grid"]).copy()
    g_orig = grid.copy()

    skel = json.load(open(skel_file, encoding="utf-8"))
    rooms = skel.get("rooms", [])

    by_style, all_furn = load_motifs()
    style_motifs = by_style.get(style, all_furn)
    if not style_motifs:
        style_motifs = all_furn

    print(f"[deco_rooms] grid {grid.shape}, style={style}, "
          f"{len(rooms)} rooms, {len(style_motifs)} style motifs")

    total_placed = 0
    for room in rooms:
        room_type = room.get("type", "room")
        rect = room.get("rect")
        if not rect:
            continue

        # Filter motifs for this room type
        room_motifs = filter_motifs_for_room(room_type, style_motifs)

        # Find interior positions in this room
        interior = find_interior_in_room(grid, room_type, rooms, rect)
        if not interior:
            continue

        # Place 2-4 furniture items per room
        n_target = min(4, max(1, len(interior) // 20))
        placed_in_room = 0

        for x, y, z in interior:
            if placed_in_room >= n_target:
                break
            if rng.random() > 0.35:
                continue
            motif = room_motifs[rng.randint(0, min(len(room_motifs) - 1, 30))]
            mx = max(0, min(x - MOTIF // 2, grid.shape[0] - MOTIF))
            my = max(0, min(y - MOTIF // 2, grid.shape[1] - MOTIF))
            mz = max(0, min(z - MOTIF // 2, grid.shape[2] - MOTIF))
            if place_motif(grid, motif, mx, my, mz):
                placed_in_room += 1

        total_placed += placed_in_room
        print(f"  {room_type:12s}: {placed_in_room} items in {len(interior)} positions")

    print(f"[deco_rooms] total placed: {total_placed}")
    np.savez_compressed(out_file, grid=grid)

    # Comparison render
    from PIL import Image
    img0 = render2(g_orig, max_px=320, cutaway=True)
    img1 = render2(grid, max_px=320, cutaway=True)
    w = img0.width + img1.width + 10
    h = max(img0.height, img1.height)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    sheet.paste(img0, (0, 0))
    sheet.paste(img1, (img0.width + 10, 0))
    cmp = out_file.replace(".npz", "_compare.png")
    sheet.save(cmp)
    print(f"[deco_rooms] saved: {out_file}, comparison: {cmp}")


if __name__ == "__main__":
    main()
