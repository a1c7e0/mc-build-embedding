"""Complete decoration placement system.

Combines:
1. MC motif vocabulary (HOW to build furniture = multi-block patterns)
2. ProcTHOR placement rules (WHERE to put furniture = position distribution)

Applies motifs to a skeleton shell following ProcTHOR learned placement:
- 53% floor, 28% wall, 19% table-height
- 55% center, 25% near wall, 20% mid
- Uniform orientation

usage: python src/decorate_full.py <grid.npz> <style> <seed> <out.npz>
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402
from apply_motifs import load_motifs, place_motif, find_interior_air  # noqa: E402

MOTIF = 3


def classify_position(x, y, z, grid):
    """Classify a position as floor/table/wall/center/mid/near_wall."""
    W, H, D = grid.shape
    # Height band
    if y <= 2:
        height = "floor"
    elif y <= 4:
        height = "table"
    else:
        height = "wall"

    # Room position (distance from nearest wall)
    dist_x = min(x, W - 1 - x)
    dist_z = min(z, D - 1 - z)
    dist = min(dist_x, dist_z)
    if dist <= 2:
        room_pos = "near_wall"
    elif dist >= 5:
        room_pos = "center"
    else:
        room_pos = "mid"

    return height, room_pos


def get_placement_targets(grid, max_items=15):
    """Generate furniture placement targets following ProcTHOR distribution."""
    W, H, D = grid.shape
    interior = find_interior_air(grid)
    if not interior:
        return []

    rng_targets = []
    # ProcTHOR: 53% floor, 28% wall, 19% table
    n_floor = int(max_items * 0.53)
    n_wall = int(max_items * 0.28)
    n_table = max_items - n_floor - n_wall

    # Filter interior by height
    floor_pos = [(x, y, z) for x, y, z in interior if y <= 2]
    table_pos = [(x, y, z) for x, y, z in interior if 2 < y <= 4]
    wall_pos = [(x, y, z) for x, y, z in interior if y > 4]

    # Sample according to distribution
    targets = []
    for pos_list, n in [(floor_pos, n_floor), (table_pos, n_table), (wall_pos, n_wall)]:
        if pos_list and n > 0:
            sampled = random.sample(pos_list, min(n, len(pos_list)))
            targets.extend(sampled)

    return targets


def main():
    if len(sys.argv) < 5:
        print("usage: python src/decorate_full.py <grid.npz> <style> <seed> <out.npz>")
        sys.exit(1)

    grid_file = sys.argv[1]
    style = sys.argv[2]
    seed = int(sys.argv[3])
    out_file = sys.argv[4]

    rng = random.Random(seed)
    npz = np.load(grid_file)
    grid = (npz["world"] if "world" in npz else npz["grid"]).copy()
    g_orig = grid.copy()

    by_style, all_furn = load_motifs()
    motifs = by_style.get(style, all_furn)
    if not motifs:
        motifs = all_furn

    targets = get_placement_targets(grid, max_items=15)
    print(f"[deco_full] {grid.shape}, style={style}, "
          f"{len(motifs)} motifs, {len(targets)} targets")

    placed = 0
    for x, y, z in targets:
        if placed >= 12:
            break
        motif = motifs[rng.randint(0, len(motifs) - 1)]
        mx = max(0, min(x - MOTIF // 2, grid.shape[0] - MOTIF))
        my = max(0, min(y - MOTIF // 2, grid.shape[1] - MOTIF))
        mz = max(0, min(z - MOTIF // 2, grid.shape[2] - MOTIF))
        if place_motif(grid, motif, mx, my, mz):
            placed += 1

    print(f"[deco_full] placed {placed} furniture motifs")
    np.savez_compressed(out_file, grid=grid)

    # Render comparison
    from PIL import Image
    img0 = render2(g_orig, max_px=300, cutaway=True)
    img1 = render2(grid, max_px=300, cutaway=True)
    w = img0.width + img1.width + 10
    h = max(img0.height, img1.height)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    sheet.paste(img0, (0, 0))
    sheet.paste(img1, (img0.width + 10, 0))
    cmp = out_file.replace(".npz", "_compare.png")
    sheet.save(cmp)
    print(f"[deco_full] saved {out_file}, comparison {cmp}")


if __name__ == "__main__":
    main()
